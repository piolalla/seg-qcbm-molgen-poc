#!/usr/bin/env bash
set -e

DATADIR="data"

python qgen_qcbm_latent.py \
  --mode train \
  --seed-csv "${DATADIR}/clean_kras_g12d.csv" \
  --seed-smiles-col canonical_smiles \
  --n-qubits 10 \
  --n-layers 3 \
  --n-shots 256 \
  --n-iters 30 \
  --init normal \
  --outdir models/qcbm_latent
