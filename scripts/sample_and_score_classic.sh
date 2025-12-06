#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f "/home/sun143/miniconda3/etc/profile.d/conda.sh" ]; then
  source /home/sun143/miniconda3/etc/profile.d/conda.sh
  conda activate qsar_env
fi

MODEL_DIR="${1:-models/classic_lm}"
DATADIR="${2:-data}"
N_SAMPLES="${3:-1000}"

GEN_CSV="${DATADIR}/gen_classic_round1.csv"
SCORED_CSV="${DATADIR}/gen_classic_round1_scored.csv"

echo "== Sampling from classic generator =="
python src/classic_selfies_lm.py \
  --mode sample \
  --model-dir "${MODEL_DIR}" \
  --n-samples "${N_SAMPLES}" \
  --out-csv "${GEN_CSV}" \
  --temperature 1.0

echo "== Scoring with rule-based oracle =="
python src/score_generated_with_rule_based.py \
  --train "${DATADIR}/clean_kras_g12d.csv" \
  --generated "${GEN_CSV}" \
  --out "${SCORED_CSV}"

echo "[OK] Classic baseline completed."
echo "Generated: ${GEN_CSV}"
echo "Scored:    ${SCORED_CSV}"
