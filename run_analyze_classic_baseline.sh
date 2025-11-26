#!/bin/bash
set -e

SCORING_FILE="data/gen_classic_round1_scored.csv"
TOPK=50

echo "============================================"
echo " Running Classic Baseline Analysis"
echo "============================================"
echo "[INFO] Scored file: $SCORING_FILE"
echo "[INFO] Top-K: $TOPK"
echo

python analyze_classic_baseline.py \
    --scored "$SCORING_FILE" \
    --topk $TOPK

echo
echo "============================================"
echo " Analysis complete."
echo "============================================"
