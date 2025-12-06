#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f "/home/sun143/miniconda3/etc/profile.d/conda.sh" ]; then
  source /home/sun143/miniconda3/etc/profile.d/conda.sh
  conda activate qsar_env
fi

TRAIN="${1:-data/clean_kras_g12d.csv}"
EXPANDED="${2:-data/expanded_candidates_scored.csv}"
OUT="${3:-data/rule_based_reward.csv}"

echo "Building rule-based reward..."
echo "  Train:    ${TRAIN}"
echo "  Expanded: ${EXPANDED}"
echo "  Out:      ${OUT}"

python src/build_rule_based_reward.py \
  --train "${TRAIN}" \
  --expanded "${EXPANDED}" \
  --out "${OUT}" \
  --w-qed 1.2 \
  --w-sa 0.8 \
  --w-sim 1.2 \
  --w-nov 0.3 \
  --w-phys 0.1
