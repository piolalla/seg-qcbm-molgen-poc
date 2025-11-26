#!/usr/bin/env bash
set -euo pipefail

# activate conda
source /home/sun143/miniconda3/etc/profile.d/conda.sh
conda activate qsar_env

SCORED="${1:-data/gen_qcbm_round1_scored.csv}"
SEEDCSV="${2:-data/clean_kras_g12d.csv}"
TOPK="${3:-50}"

echo "[INFO] Scored file: $SCORED"
echo "[INFO] Seed file:   $SEEDCSV"
echo "[INFO] TopK:        $TOPK"

python analyze_qcbm_with_seed.py \
    --scored "$SCORED" \
    --seed-csv "$SEEDCSV" \
    --topk "$TOPK"
