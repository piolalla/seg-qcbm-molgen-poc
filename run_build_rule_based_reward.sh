#!/usr/bin/env bash
set -e

#   prep_kras_g12d.py  -> 得到 data/clean_kras_g12d.csv
#   并生成 data/expanded_candidates_scored.csv

TRAIN="${1:-data/clean_kras_g12d.csv}"
EXPANDED="${2:-data/expanded_candidates_scored.csv}"
OUT="${3:-data/rule_based_reward.csv}"

echo "Building rule-based reward..."
echo "  Train:    ${TRAIN}"
echo "  Expanded: ${EXPANDED}"
echo "  Out:      ${OUT}"

python build_rule_based_reward.py \
  --train "${TRAIN}" \
  --expanded "${EXPANDED}" \
  --out "${OUT}" \
  --w-qed 1.2 \
  --w-sa 0.8 \
  --w-sim 1.2 \
  --w-nov 0.3 \
  --w-phys 0.1
