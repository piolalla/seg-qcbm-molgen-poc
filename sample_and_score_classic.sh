#!/usr/bin/env bash
set -e

# 使用经典生成器采样并用 rule-based oracle 打分

MODEL_DIR="${1:-models/classic_lm}"
DATADIR="${2:-data}"
N_SAMPLES="${3:-1000}"

GEN_CSV="${DATADIR}/gen_classic_round1.csv"
SCORED_CSV="${DATADIR}/gen_classic_round1_scored.csv"

echo "== Sampling from classic generator =="
python classic_selfies_lm.py \
  --mode sample \
  --model-dir "${MODEL_DIR}" \
  --n-samples "${N_SAMPLES}" \
  --out-csv "${GEN_CSV}" \
  --temperature 1.0

echo "== Scoring with rule-based oracle =="
python score_generated_with_rule_based.py \
  --train "${DATADIR}/clean_kras_g12d.csv" \
  --generated "${GEN_CSV}" \
  --out "${SCORED_CSV}"

echo "[OK] Classic baseline completed."
echo "Generated: ${GEN_CSV}"
echo "Scored:    ${SCORED_CSV}"
