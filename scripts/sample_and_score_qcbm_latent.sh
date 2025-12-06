#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f "/home/sun143/miniconda3/etc/profile.d/conda.sh" ]; then
  source /home/sun143/miniconda3/etc/profile.d/conda.sh
  conda activate qsar_env
fi

DATADIR="data"

python src/qgen_qcbm_latent.py \
  --mode sample \
  --seed-csv "${DATADIR}/clean_kras_g12d.csv" \
  --seed-smiles-col canonical_smiles \
  --model-dir models/qcbm_latent \
  --n-samples 1000 \
  --n-shots 512 \
  --num-mutations 1 \
  --out-csv "${DATADIR}/gen_qcbm_round1.csv"

python src/score_generated_with_rule_based.py \
  --train "${DATADIR}/clean_kras_g12d.csv" \
  --generated "${DATADIR}/gen_qcbm_round1.csv" \
  --out "${DATADIR}/gen_qcbm_round1_scored.csv"
