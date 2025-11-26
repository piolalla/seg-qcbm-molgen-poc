#!/usr/bin/env bash
set -e

# 训练经典 SELFIES-LSTM baseline
# 默认只用 KRAS G12D 的 clean 数据；如果你以后想加 MOSES，可以在这里追加一个 .selfies.txt

OUTDIR="models/classic_lm"

mkdir -p "${OUTDIR}"

python classic_selfies_lm.py \
  --mode train \
  --selfies-files data/clean_kras_g12d.selfies.txt \
  --outdir "${OUTDIR}" \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --emb-dim 128 \
  --hid-dim 256 \
  --num-layers 2 \
  --max-len 120
