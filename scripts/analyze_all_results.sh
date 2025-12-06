#!/usr/bin/env bash
set -euo pipefail

# ---------- locate project root ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# ---------- (optional) activate conda env ----------
if [ -f "/home/sun143/miniconda3/etc/profile.d/conda.sh" ]; then
  source /home/sun143/miniconda3/etc/profile.d/conda.sh
  conda activate qsar_env
fi

PY_ANALYZER="src/analyze_qcbm_with_seed.py"
TOPK=50

if [ ! -f "$PY_ANALYZER" ]; then
  echo "[ERROR] Python analyzer '$PY_ANALYZER' not found."
  exit 1
fi

run_analysis() {
  local label="$1"
  local csv_path="$2"

  if [ -f "$csv_path" ]; then
    echo "============================================================"
    echo " ANALYZING: $label"
    echo " File: $csv_path"
    echo "============================================================"
    python "$PY_ANALYZER" --scored "$csv_path" --topk "$TOPK"
    echo
  else
    echo "[WARN] $label file not found: $csv_path  (skip)"
    echo
  fi
}

run_analysis "CLASSIC BASELINE"   "data/gen_classic_round1_scored.csv"
run_analysis "NON-SEGMENTED QCBM" "data/gen_qcbm_round1_scored.csv"
run_analysis "SEGMENTED QCBM"     "data/gen_qcbm_segmented_round1_scored.csv"

echo "[DONE] All available result files have been analyzed."
