#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze scored generator outputs (classic baseline or others).

Features:
  - Global stats over all rows
  - Top-K-by-reward stats (for PPT / comparison)
Usage:
  python analyze_classic_baseline.py \
    --scored data/gen_classic_round1_scored.csv \
    --topk 50
"""

import argparse
import numpy as np
import pandas as pd


def describe_series(name, s, prefix=""):
    s = pd.to_numeric(s, errors="coerce")
    s = s.dropna()
    if len(s) == 0:
        print(f"{prefix}{name}: (no valid values)")
        return
    print(f"{prefix}{name}:")
    print(f"{prefix}  count = {len(s)}")
    print(f"{prefix}  mean  = {s.mean():.4f}")
    print(f"{prefix}  std   = {s.std():.4f}")
    qs = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"{prefix}  quantiles:")
    for q, v in qs.items():
        print(f"{prefix}    q{int(q*100):2d} = {v:.4f}")


def analyze_block(df, label, topk=None):
    print(f"\n========== {label} ==========")
    if topk is not None:
        print(f"[INFO] Rows in this block: {len(df)} (Top-{topk} by reward)")
    else:
        print(f"[INFO] Rows in this block: {len(df)} (all samples)")

    # （validity / uniqueness）
    if "smiles" in df.columns:
        unique_smiles = df["smiles"].dropna().unique()
        n_unique = len(unique_smiles)
        print(f"[BASIC] Unique SMILES = {n_unique}")
        if len(df) > 0:
            print(f"[BASIC] Uniqueness ratio = {n_unique / len(df):.4f}")
    else:
        print("[WARN] No 'smiles' column found.")

    print("\n[STATS] Distribution summary:")
    for col in ["reward", "QED", "SA", "sim_g12d", "novelty_raw", "molwt"]:
        if col in df.columns:
            describe_series(col, df[col], prefix="  ")
        else:
            print(f"  {col}: (column not found)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze scored generator results (classic baseline, quantum, etc.)."
    )
    parser.add_argument(
        "--scored",
        type=str,
        default="data/gen_classic_round1_scored.csv",
        help="Path to scored CSV (must contain at least 'smiles' and 'reward').",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Top-K by reward for the second analysis block.",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading scored file: {args.scored}")
    df = pd.read_csv(args.scored)
    n_total = len(df)
    print(f"[INFO] Total rows in file = {n_total}")

    if "reward" not in df.columns:
        raise ValueError("'reward' column not found; cannot rank Top-K.")

    analyze_block(df, label="GLOBAL (ALL SAMPLES)")

    # Top-K by reward
    topk = min(args.topk, len(df))
    df_top = df.sort_values("reward", ascending=False).head(topk).reset_index(drop=True)
    analyze_block(df_top, label=f"TOP-{topk} BY REWARD", topk=topk)


    print(f"\n[TOP-{topk}] rows by reward (compact view):")
    cols_to_show = [c for c in ["smiles", "QED", "SA", "sim_g12d", "novelty_raw", "molwt", "reward"]
                    if c in df.columns]
    with pd.option_context("display.max_rows", topk, "display.max_colwidth", 80):
        print(df_top[cols_to_show])


if __name__ == "__main__":
    main()
