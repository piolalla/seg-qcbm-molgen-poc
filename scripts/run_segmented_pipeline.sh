#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f "/home/sun143/miniconda3/etc/profile.d/conda.sh" ]; then
  source /home/sun143/miniconda3/etc/profile.d/conda.sh
  conda activate qsar_env
fi

echo "======================"
echo "STEP 1: Segmenting seeds..."
echo "======================"

python src/preprocess_segments.py \
  --seed-csv data/clean_kras_g12d.csv \
  --smiles-col canonical_smiles \
  --n-segments 3 \
  --out-csv data/clean_kras_g12d_segmented.csv

echo "[OK] Segmented seeds written to data/clean_kras_g12d_segmented.csv"
echo

echo "======================"
echo "STEP 2: Training segmented QCBM..."
echo "======================"

python src/qgen_qcbm_segmented.py --mode train \
  --seed-csv data/clean_kras_g12d_segmented.csv \
  --seed-smiles-col canonical_smiles \
  --outdir models/qcbm_segmented \
  --min-qubits 3 \
  --max-qubits 6 \
  --n-layers 3 \
  --n-iters 30 \
  --n-shots 512

echo "[OK] Models saved to models/qcbm_segmented/"
echo

echo "======================"
echo "STEP 3: Sampling from segmented QCBM..."
echo "======================"

python src/qgen_qcbm_segmented.py --mode sample \
  --seed-csv data/clean_kras_g12d_segmented.csv \
  --model-dir models/qcbm_segmented \
  --out-csv data/gen_qcbm_segmented_round1.csv \
  --n-samples 1000 \
  --max-batches 50 \
  --num-mutations 1 \
  --n-shots 512

echo
echo "[DONE] Generated molecules saved to data/gen_qcbm_segmented_round1.csv"
echo "==============================================="
