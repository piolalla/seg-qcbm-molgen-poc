#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f "/home/sun143/miniconda3/etc/profile.d/conda.sh" ]; then
  source /home/sun143/miniconda3/etc/profile.d/conda.sh
  conda activate qsar_env
fi

SCORING_FILE="data/gen_classic_round1_scored.csv"
TOPK=50

echo "============================================"
echo " Running Classic Baseline Analysis"
echo "============================================"
echo "[INFO] Scored file: $SCORING_FILE"
echo "[INFO] Top-K: $TOPK"
echo

python src/analyze_classic_baseline.py \
  --scored "$SCORING_FILE" \
  --topk "$TOPK"

echo
echo "============================================"
echo " Analysis complete."
echo "============================================"
