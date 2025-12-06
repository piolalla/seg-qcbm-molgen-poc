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
  --mode train \
  --seed-csv "${DATADIR}/clean_kras_g12d.csv" \
  --seed-smiles-col canonical_smiles \
  --n-qubits 10 \
  --n-layers 3 \
  --n-shots 256 \
  --n-iters 30 \
  --init normal \
  --outdir models/qcbm_latent
