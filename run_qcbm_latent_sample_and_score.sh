#!/usr/bin/env bash
set -e

DATADIR="data"

# 1) sample by QCBM latent generator
python qgen_qcbm_latent.py \
  --mode sample \
  --seed-csv "${DATADIR}/clean_kras_g12d.csv" \
  --seed-smiles-col canonical_smiles \
  --model-dir models/qcbm_latent \
  --n-samples 1000 \
  --n-shots 512 \
  --num-mutations 1 \
  --out-csv "${DATADIR}/gen_qcbm_round1.csv"

# 2) score with your existing rule-based oracle
python score_generated_with_rule_based.py \
  --train "${DATADIR}/clean_kras_g12d.csv" \
  --generated "${DATADIR}/gen_qcbm_round1.csv" \
  --out "${DATADIR}/gen_qcbm_round1_scored.csv"