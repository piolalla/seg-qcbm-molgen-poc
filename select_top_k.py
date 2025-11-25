#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select top-K molecules by Reward and export files for generator fine-tuning.

Inputs:
    data/reward.csv
        Required columns:
            - 'Reward'
            - at least one of: 'smiles', 'canonical_smiles'

Outputs (with prefix OUT_PREFIX, default: data/finetune):
    - {OUT_PREFIX}_topk.csv
        Full table of selected molecules with all original columns.
    - {OUT_PREFIX}_topk.selfies.txt
        One SELFIES string per line (for language-model fine-tuning).
    - {OUT_PREFIX}_topk.jsonl
        JSONL with {"selfies": ..., "reward": ...} per line
        (for RL, ranking fine-tuning, etc.).
"""

import os
import json
import argparse
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import selfies as sf

# optional: silence RDKit noise
RDLogger.DisableLog('rdApp.*')

DEFAULT_INPUT   = "data/reward.csv"
DEFAULT_PREFIX  = "data/finetune"

# ---------------------- helpers ---------------------- #

def to_canonical_smiles(smi: str):
    """Return canonical isomeric SMILES or None if invalid."""
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        can = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return can
    except Exception:
        return None

def smiles_to_selfies_safe(smi: str):
    """Convert SMILES to canonical SELFIES, return None on failure."""
    can = to_canonical_smiles(smi)
    if can is None:
        return None
    try:
        return sf.encoder(can)
    except Exception:
        return None

# ---------------------- main ---------------------- #

def main(args):
    input_csv = args.input
    out_prefix = args.out_prefix
    top_k = args.top_k
    min_reward = args.min_reward

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"[OK] Loaded {len(df)} rows from {input_csv}")

    # ensure Reward column exists
    if "Reward" not in df.columns:
        raise ValueError("Column 'Reward' is missing in input CSV.")

    # decide which SMILES column to use
    smiles_col = None
    if "smiles" in df.columns:
        smiles_col = "smiles"
    elif "canonical_smiles" in df.columns:
        smiles_col = "canonical_smiles"
    else:
        raise ValueError("No 'smiles' or 'canonical_smiles' column in input CSV.")

    # optional filtering by minimum reward
    if min_reward is not None:
        before = len(df)
        df = df[df["Reward"] >= min_reward].reset_index(drop=True)
        print(f"[INFO] Filtered by min_reward={min_reward}: {before} -> {len(df)} rows")

    if df.empty:
        raise RuntimeError("No rows left after filtering. Check your min_reward or input file.")

    # sort by Reward descending
    df = df.sort_values(by="Reward", ascending=False).reset_index(drop=True)

    # canonicalize SMILES & drop duplicates
    print("[INFO] Canonicalizing SMILES and dropping duplicates...")
    df["canonical_for_dedup"] = df[smiles_col].apply(to_canonical_smiles)
    df = df.dropna(subset=["canonical_for_dedup"])
    before_dup = len(df)
    df = df.drop_duplicates(subset=["canonical_for_dedup"]).reset_index(drop=True)
    print(f"[INFO] Dropped duplicates: {before_dup} -> {len(df)} rows")

    # select top-K
    if top_k is not None and top_k > 0:
        df_top = df.head(top_k).copy()
    else:
        df_top = df.copy()

    print(f"[OK] Selected {len(df_top)} rows for fine-tuning.")

    # compute SELFIES for selected molecules
    print("[INFO] Converting SMILES to SELFIES...")
    selfies_list = []
    valid_idx = []

    for idx, smi in enumerate(df_top["canonical_for_dedup"].tolist()):
        s = smiles_to_selfies_safe(smi)
        if s is None:
            continue
        selfies_list.append(s)
        valid_idx.append(idx)

    if not selfies_list:
        raise RuntimeError("No valid SELFIES could be generated from selected molecules.")

    # keep only rows for which we have valid SELFIES
    df_top = df_top.iloc[valid_idx].reset_index(drop=True)
    selfies_list = selfies_list[:len(df_top)]

    df_top["SELFIES"] = selfies_list

    # ---------------------- outputs ---------------------- #
    csv_out   = f"{out_prefix}_topk.csv"
    txt_out   = f"{out_prefix}_topk.selfies.txt"
    jsonl_out = f"{out_prefix}_topk.jsonl"

    # 1) full CSV
    df_top.to_csv(csv_out, index=False)
    print(f"[OK] Saved CSV selection to: {csv_out}")

    # 2) plain SELFIES TXT (one per line)
    with open(txt_out, "w", encoding="utf-8") as f:
        for s in selfies_list:
            f.write(s + "\n")
    print(f"[OK] Saved SELFIES to: {txt_out}")

    # 3) JSONL: {selfies, reward}
    with open(jsonl_out, "w", encoding="utf-8") as f:
        for s, r in zip(selfies_list, df_top["Reward"].tolist()):
            obj = {"selfies": s, "reward": float(r)}
            f.write(json.dumps(obj) + "\n")
    print(f"[OK] Saved JSONL to: {jsonl_out}")

    print("[DONE] Top-K selection and export finished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Path to reward CSV (default: data/reward.csv)"
    )
    ap.add_argument(
        "--out_prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Output prefix (default: data/finetune)"
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=256,
        help="Number of top molecules to select (default: 256)"
    )
    ap.add_argument(
        "--min_reward",
        type=float,
        default=None,
        help="Optional minimum Reward threshold; set None to disable."
    )
    args = ap.parse_args()
    main(args)
