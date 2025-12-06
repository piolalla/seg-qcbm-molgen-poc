#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f "/home/sun143/miniconda3/etc/profile.d/conda.sh" ]; then
  source /home/sun143/miniconda3/etc/profile.d/conda.sh
  conda activate qsar_env
fi

OUTDIR="models/classic_lm"
mkdir -p "${OUTDIR}"

python src/classic_selfies_lm.py \
  --mode train \
  --selfies-files data/clean_kras_g12d.selfies.txt \
  --outdir "${OUTDIR}" \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --emb-dim 128 \
  --hid-dim 256 \
  --num-layers 2 \
  --max-len 120
