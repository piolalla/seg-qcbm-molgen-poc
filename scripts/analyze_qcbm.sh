#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f "/home/sun143/miniconda3/etc/profile.d/conda.sh" ]; then
  source /home/sun143/miniconda3/etc/profile.d/conda.sh
  conda activate qsar_env
fi

SCORED="${1:-data/gen_qcbm_round1_scored.csv}"
SEEDCSV="${2:-data/clean_kras_g12d.csv}"
TOPK="${3:-50}"

echo "[INFO] Scored file: $SCORED"
echo "[INFO] Seed file:   $SEEDCSV"
echo "[INFO] TopK:        $TOPK"

python src/analyze_qcbm_with_seed.py \
  --scored "$SCORED" \
  --seed-csv "$SEEDCSV" \
  --topk "$TOPK"
