#!/usr/bin/env bash
set -euo pipefail

# ===== 0. Activate conda env =====
source /home/sun143/miniconda3/etc/profile.d/conda.sh
conda activate qsar_env

PY_ANALYZER="analyze_qcbm_with_seed.py"
TOPK=50
DATADIR="data"

if [ ! -f "$PY_ANALYZER" ]; then
  echo "[ERROR] Python analyzer '$PY_ANALYZER' not found in current directory."
  echo "        Please make sure analyze_qcbm_with_seed.py is here."
  exit 1
fi

run_analysis () {
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

# ===== 1. Classic baseline =====
#run_analysis "CLASSIC BASELINE" "data/gen_classic_round1_scored.csv"

# ===== 2. Non-segmented QCBM latent =====
#run_analysis "NON-SEGMENTED QCBM" "data/gen_qcbm_round1_scored.csv"

# ===== 3. Segmented QCBM latent =====
python score_generated_with_rule_based.py \
  --train "${DATADIR}/clean_kras_g12d.csv" \
  --generated "${DATADIR}/gen_qcbm_segmented_round1.csv" \
  --out "${DATADIR}/gen_qcbm_segmented_round1_scored.csv"

run_analysis "SEGMENTED QCBM" "data/gen_qcbm_segmented_round1_scored.csv"

echo "[DONE] All available result files have been analyzed."
