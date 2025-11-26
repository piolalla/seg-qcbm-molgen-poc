#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze scored generator outputs (classic / quantum / segmented).

Features:
  - Global stats over all rows
  - Top-K-by-reward stats
  - Optional seed-level analysis (if seed_idx / seed_reward_qed_mw exist)
  - Optional segment-level analysis (if segment_id exists)
  - Explicit Global vs Top-K comparison for:
      * novelty_raw
      * physchem_score
      * (optionally) f_physchem

Usage:
  python analyze_qcbm_with_seed.py \
    --scored data/gen_qcbm_segmented_round1_scored.csv \
    --topk 50
"""

import argparse
import numpy as np
import pandas as pd


# ========= Basic numeric summary =========

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
        print(f"[INFO] Block size = {len(df)} (Top-{topk} by reward)")
    else:
        print(f"[INFO] Block size = {len(df)}")

    # 基本唯一性 & 有效性
    if "smiles" in df.columns:
        uniq = df["smiles"].dropna().unique()
        n_unique = len(uniq)
        print(f"[BASIC] Unique SMILES = {n_unique}")
        if len(df) > 0:
            print(f"[BASIC] Uniqueness ratio = {n_unique / len(df):.4f}")
    else:
        print("[WARN] No 'smiles' column found.")

    print("\n[STATS] Distribution summary:")
    # 在这里补上 physchem_score / f_physchem
    cols_to_summarize = [
        "reward",
        "QED",
        "SA",
        "sim_g12d",
        "novelty_raw",
        "molwt",
        "physchem_score",
        "f_physchem",
    ]
    for col in cols_to_summarize:
        if col in df.columns:
            describe_series(col, df[col], prefix="  ")
        else:
            # 对缺少的列简单提示即可
            print(f"  {col}: (column not found)")


# ========= Seed-level analysis =========

def analyze_seed_level(df):
    """Analyze seed usage and relation to seed_reward_qed_mw."""
    if "seed_idx" not in df.columns or "seed_reward_qed_mw" not in df.columns:
        print("\n========== SEED-LEVEL ANALYSIS ==========")
        print("[INFO] seed_idx / seed_reward_qed_mw not found; skip seed-level analysis.")
        return

    print("\n========== SEED-LEVEL ANALYSIS ==========")
    seed_idx = df["seed_idx"].astype(int)
    seed_reward = pd.to_numeric(df["seed_reward_qed_mw"], errors="coerce")

    # 使用频率
    vc = seed_idx.value_counts().sort_index()
    n_seeds_used = len(vc)
    print(f"[INFO] Number of unique seeds used = {n_seeds_used}")
    print("[INFO] Most frequent seeds (top 10):")
    print(vc.head(10))

    # 种子使用熵 H(seed_idx)
    p = vc / vc.sum()
    entropy = -(p * np.log2(p)).sum()
    max_entropy = np.log2(len(vc)) if len(vc) > 0 else 0.0
    print(f"\n[SEED ENTROPY] = {entropy:.4f} bits (max={max_entropy:.2f})")

    # 每个 seed 的平均 seed_reward_qed_mw
    df_seed = pd.DataFrame({
        "seed_idx": seed_idx,
        "seed_reward_qed_mw": seed_reward
    }).dropna()
    grouped = df_seed.groupby("seed_idx")
    seed_mean_reward = grouped["seed_reward_qed_mw"].mean()
    seed_usage = grouped.size()

    seed_table = pd.DataFrame({
        "count": seed_usage,
        "mean_seed_reward": seed_mean_reward
    }).sort_values("count", ascending=False)

    print("\n[SEED CONTRIBUTION TABLE] top 20:")
    print(seed_table.head(20))

    # 相关性：种子 reward vs 使用次数
    if len(seed_table) > 1:
        corr_seed = seed_table["mean_seed_reward"].corr(seed_table["count"])
        print(f"\n[CORRELATION] seed_reward vs seed_usage count = {corr_seed:.4f}")
    else:
        print("\n[CORRELATION] Not enough seeds to compute correlation.")

    # 行级相关性：最终 reward vs seed_reward_qed_mw
    if "reward" in df.columns:
        df_row = df[["reward", "seed_reward_qed_mw"]].copy()
        df_row = df_row.apply(pd.to_numeric, errors="coerce").dropna()
        if len(df_row) > 1:
            corr_row = df_row["reward"].corr(df_row["seed_reward_qed_mw"])
            print(f"[CORRELATION] row-level: reward vs seed_reward = {corr_row:.4f}")
        else:
            print("[CORRELATION] Not enough rows to compute row-level correlation.")
    else:
        print("[CORRELATION] 'reward' column not found; skip row-level correlation.")


# ========= Segment-level analysis (for segmented QCBM) =========

def analyze_segment_level(df):
    """Analyze segment_id usage and per-segment stats (only if segment_id exists)."""
    if "segment_id" not in df.columns:
        print("\n========== SEGMENT-LEVEL ANALYSIS ==========")
        print("[INFO] segment_id not found; skip segment-level analysis.")
        return

    print("\n========== SEGMENT-LEVEL ANALYSIS ==========")
    seg = df["segment_id"].astype(int)
    vc = seg.value_counts().sort_index()

    n_seg = len(vc)
    print(f"[INFO] Number of segments used = {n_seg}")
    print("[INFO] Segment usage counts:")
    print(vc)

    # segment 熵 H(segment_id)
    p = vc / vc.sum()
    entropy = -(p * np.log2(p)).sum()
    max_entropy = np.log2(len(vc)) if len(vc) > 0 else 0.0
    print(f"\n[SEGMENT ENTROPY] = {entropy:.4f} bits (max={max_entropy:.2f})")

    # 每个 segment 的平均 reward / QED / sim / novelty / physchem
    cols = [
        "reward",
        "QED",
        "SA",
        "sim_g12d",
        "novelty_raw",
        "molwt",
        "physchem_score",
        "f_physchem",
    ]
    avail_cols = [c for c in cols if c in df.columns]
    if not avail_cols:
        print("\n[SEGMENT STATS] No numeric columns (reward/QED/SA/...) found; skip.")
        return

    df_seg = df.copy()
    for c in avail_cols:
        df_seg[c] = pd.to_numeric(df_seg[c], errors="coerce")

    grouped = df_seg.groupby("segment_id")

    seg_stats = grouped[avail_cols].mean()
    seg_stats["count"] = grouped.size()

    # 方便看，把 count 放前面
    cols_order = ["count"] + avail_cols
    seg_stats = seg_stats[cols_order].sort_values("count", ascending=False)

    print("\n[SEGMENT STATS] (mean over each segment):")
    print(seg_stats)


# ========= Global vs Top-K comparison for novelty / physchem =========

def compare_global_topk(df_global, df_top, topk):
    print("\n========== GLOBAL vs TOP-{0} COMPARISON ==========".format(topk))

    def safe_mean(series):
        series = pd.to_numeric(series, errors="coerce").dropna()
        if len(series) == 0:
            return None
        return float(series.mean())

    metrics = [
        ("novelty_raw",       "Novelty (novelty_raw)"),
        ("physchem_score",    "Physchem score (physchem_score)"),
        ("f_physchem",        "Physchem weight (f_physchem)"),
    ]

    for col, desc in metrics:
        if col not in df_global.columns:
            print(f"[{desc}] column '{col}' not found; skip.")
            continue

        g_mean = safe_mean(df_global[col])
        t_mean = safe_mean(df_top[col])

        if g_mean is None or t_mean is None:
            print(f"[{desc}] insufficient data; skip.")
            continue

        print(f"[{desc}]")
        print(f"  Global mean  = {g_mean:.4f}")
        print(f"  Top-{topk} mean = {t_mean:.4f}")
        print(f"  Δ(topk - global) = {t_mean - g_mean:+.4f}")


# ========= Main =========

def main():
    parser = argparse.ArgumentParser(
        description="Analyze scored generator results (classic / quantum / segmented)."
    )
    parser.add_argument(
        "--scored",
        type=str,
        required=True,
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

    # 全局
    analyze_block(df, label="GLOBAL (ALL SAMPLES)")

    # Top-K by reward
    topk = min(args.topk, len(df))
    df_top = df.sort_values("reward", ascending=False).head(topk).reset_index(drop=True)
    analyze_block(df_top, label=f"TOP-{topk} BY REWARD", topk=topk)

    # 种子级分析（如果有）
    analyze_seed_level(df)

    # 分段级分析（如果有 segment_id）
    analyze_segment_level(df)

    # Global vs Top-K 对比 (novelty_raw / physchem_score / f_physchem)
    compare_global_topk(df, df_top, topk)

    # 最后给一个紧凑视图（方便 eyeball top rows）
    print("\n[TOP-{0}] rows by reward (compact view):".format(topk))
    cols_to_show = [c for c in [
        "smiles",
        "segment_id",
        "seed_idx",
        "QED",
        "SA",
        "sim_g12d",
        "novelty_raw",
        "physchem_score",
        "molwt",
        "reward",
    ] if c in df.columns]

    with pd.option_context("display.max_rows", topk,
                           "display.max_colwidth", 80):
        print(df_top[cols_to_show])

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()
