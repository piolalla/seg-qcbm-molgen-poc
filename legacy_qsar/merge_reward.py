#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

# ============================================================
# Config: input & output
# ============================================================
INPUT_CSV  = "data/expanded_with_novelty.csv"  # now expects R_novel
OUTPUT_CSV = "data/reward.csv"

# ============================================================
# Config: reward weights (YOU CAN TUNE THESE)
# ============================================================
W_ACTIVITY = 0.55   # QSAR predicted activity
W_QED      = 0.20   # drug-likeness
W_SA       = 0.20   # synthetic accessibility
W_NOVEL    = 0.05   # novelty/diversity
# Docking ignored for now

# ============================================================
# Satisfaction shaping functions
# ============================================================

def f_activity(p):
    """Reward only for p >= 0.5; map [0.5,1.0] -> [0,1]."""
    if pd.isna(p):
        return 0.0
    if p <= 0.5:
        return 0.0
    return (p - 0.5) / 0.5

def f_qed(q):
    """
    QED ideal range ~[0.4, 0.9].
    Hard zero for very low/high values.
    """
    if pd.isna(q):
        return 0.0
    if q < 0.2 or q > 0.95:
        return 0.0
    if 0.4 <= q <= 0.9:
        return 1.0
    if q < 0.4:
        return (q - 0.2) / (0.4 - 0.2)
    else:
        return (0.95 - q) / (0.95 - 0.9)

def f_sa(sa):
    """
    SA (1-10): lower is better.
    <=4 ideal, >=8 unacceptable.
    """
    if pd.isna(sa):
        return 0.0
    if sa >= 8:
        return 0.0
    if sa <= 4:
        return 1.0
    return (8 - sa) / (8 - 4)

def f_novel(r):
    """
    Novelty reward: only for r >= 0.2.
    """
    if pd.isna(r):
        return 0.0
    if r <= 0.2:
        return 0.0
    return (r - 0.2) / 0.8

# ============================================================
# Load data
# ============================================================

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print(f"[OK] Loaded {len(df)} rows from {INPUT_CSV}")

for col in ["P_pred", "QED", "SA", "R_novel"]:
    if col not in df.columns:
        raise ValueError(f"ERROR: Required column '{col}' is missing in input CSV.")

# ============================================================
# Compute satisfaction scores
# ============================================================

df["s_act"]   = df["P_pred"].apply(f_activity)
df["s_qed"]   = df["QED"].apply(f_qed)
df["s_sa"]    = df["SA"].apply(f_sa)
df["s_novel"] = df["R_novel"].apply(f_novel)

# ============================================================
# Final Reward
# ============================================================

df["Reward"] = (
      W_ACTIVITY * df["s_act"]
    + W_QED      * df["s_qed"]
    + W_SA       * df["s_sa"]
    + W_NOVEL    * df["s_novel"]
)

df.to_csv(OUTPUT_CSV, index=False)
print(f"[OK] Saved reward file to: {OUTPUT_CSV}")
