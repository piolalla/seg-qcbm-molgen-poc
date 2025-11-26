#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze generator outputs INCLUDING seed-related behavior.

Adds QCBM-specific analysis:
  - seed_idx frequency
  - mean seed_reward_qed_mw
  - Top contributing seeds
  - correlation between seed_reward and final reward
  - collapse detection (entropy)
Usage:
  python analyze_qcbm_with_seed.py \
      --scored data/gen_qcbm_round1_scored.csv \
      --seed-csv data/clean_kras_g12d.csv \
      --seed-smiles-col canonical_smiles \
      --topk 50
"""

import argparse
import numpy as np
import pandas as pd


# --------------------- General stats ---------------------
def describe_series(name, s, prefix=""):
    s = pd.to_numeric(s, errors="coerce").dropna()
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
    if topk:
        print(f"[INFO] Block size = {len(df)} (Top-{topk} by reward)")
    else:
        print(f"[INFO] Block size = {len(df)}")

    if "smiles" in df.columns:
        uniq = df["smiles"].dropna().unique()
        print(f"[BASIC] Unique SMILES = {len(uniq)}")
        print(f"[BASIC] Uniqueness ratio = {len(uniq)/len(df):.4f}")

    print("\n[STATS] Distribution summary:")
    for col in ["reward", "QED", "SA", "sim_g12d", "novelty_raw", "molwt"]:
        if col in df.columns:
            describe_series(col, df[col], prefix="  ")
        else:
            print(f"  {col}: missing")


# --------------------- Seed-level analysis ---------------------
def analyze_seed_behavior(df, df_seeds, seed_col):
    print("\n========== SEED-LEVEL ANALYSIS ==========")

    if "seed_idx" not in df.columns:
        print("[WARN] No seed_idx column; cannot perform seed analysis.")
        return

    seed_counts = df["seed_idx"].value_counts().sort_index()
    print(f"[INFO] Number of unique seeds used = {len(seed_counts)}")
    print(f"[INFO] Most frequent seeds (top 10):")
    print(seed_counts.head(10))

    # Normalize to probabilities
    p = seed_counts / seed_counts.sum()
    entropy = -np.sum(p * np.log2(p))
    print(f"\n[SEED ENTROPY] = {entropy:.4f} bits (max={np.log2(len(seed_counts)):.2f})")

    # Join seed reward
    df_sr = df.groupby("seed_idx")["seed_reward_qed_mw"].mean()
    df_join = pd.merge(
        seed_counts.rename("count"),
        df_sr.rename("mean_seed_reward"),
        left_index=True, right_index=True
    )

    print("\n[SEED CONTRIBUTION TABLE] top 20:")
    print(df_join.head(20))

    # Correlation: seed_reward vs. usage frequency
    corr = df_join["count"].corr(df_join["mean_seed_reward"])
    print(f"\n[CORRELATION] seed_reward vs seed_usage count = {corr:.4f}")

    # Correlation: final reward vs seed reward over rows
    if "reward" in df.columns:
        corr2 = df["reward"].corr(df["seed_reward_qed_mw"])
        print(f"[CORRELATION] row-level: reward vs seed_reward = {corr2:.4f}")


# --------------------- Main ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=str, required=True)
    parser.add_argument("--seed-csv", type=str, required=True)
    parser.add_argument("--seed-smiles-col", type=str, default="canonical_smiles")
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    print(f"[INFO] Loading: {args.scored}")
    df = pd.read_csv(args.scored)

    print(f"[INFO] Loading seeds: {args.seed_csv}")
    df_seeds = pd.read_csv(args.seed_csv)

    # Global
    analyze_block(df, label="GLOBAL (ALL SAMPLES)")

    # Top-K
    topk = min(args.topk, len(df))
    df_top = df.sort_values("reward", ascending=False).head(topk)
    analyze_block(df_top, label=f"TOP-{topk} BY REWARD", topk=topk)

    # Seed behavior
    analyze_seed_behavior(df, df_seeds, args.seed_smiles_col)

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()
