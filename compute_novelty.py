#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Compute novelty scores (R_novel) for expanded molecules.
# R_novel = 1 - max Tanimoto similarity to any training molecule (ECFP4).
#
# Input:
#   data/clean_kras_g12d.csv          (training set)
#   data/expanded_with_p.csv          (expanded set with P_pred, QED, SA)
#
# Output:
#   data/expanded_with_novelty.csv    (same as expanded_with_p + R_novel)

import os
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Disable most RDKit logging noise
RDLogger.DisableLog('rdApp.*')

DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "clean_kras_g12d.csv")
EXP_CSV   = os.path.join(DATA_DIR, "expanded_with_p.csv")
OUT_CSV   = os.path.join(DATA_DIR, "expanded_with_novelty.csv")

# ----------------------------------------------------
# ECFP4 fingerprint
# ----------------------------------------------------
def ecfp4_fp(smiles, nBits=2048, radius=2):
    """Return RDKit ExplicitBitVect for ECFP4 fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV}")
if not os.path.exists(EXP_CSV):
    raise FileNotFoundError(f"Expanded CSV not found: {EXP_CSV}")

df_train = pd.read_csv(TRAIN_CSV)
df_exp   = pd.read_csv(EXP_CSV)

print(f"[OK] Loaded training set: {len(df_train)} rows")
print(f"[OK] Loaded expanded set: {len(df_exp)} rows")

# Identify SMILES columns
if "canonical_smiles" not in df_train.columns:
    raise ValueError("Training CSV must contain 'canonical_smiles' column.")

if "smiles" in df_exp.columns:
    exp_smi_col = "smiles"
elif "canonical_smiles" in df_exp.columns:
    exp_smi_col = "canonical_smiles"
else:
    raise ValueError("Expanded CSV must contain 'smiles' or 'canonical_smiles'.")

# ----------------------------------------------------
# Compute fingerprints for training set
# ----------------------------------------------------
train_smiles = df_train["canonical_smiles"].astype(str).tolist()
train_fps = []
valid_idx = []

for i, smi in enumerate(train_smiles):
    fp = ecfp4_fp(smi)
    if fp is not None:
        train_fps.append(fp)
        valid_idx.append(i)

if not train_fps:
    raise RuntimeError("No valid fingerprints could be computed for training set.")

print(f"[OK] Training fingerprints computed: {len(train_fps)}")

# ----------------------------------------------------
# For each expanded molecule, compute max similarity to train
# ----------------------------------------------------
exp_smiles = df_exp[exp_smi_col].astype(str).tolist()
r_novel_list = []

for idx, smi in enumerate(exp_smiles):
    fp = ecfp4_fp(smi)
    if fp is None:
        # No valid fingerprint → treat as zero novelty
        r_novel_list.append(0.0)
        continue

    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    max_sim = max(sims) if sims else 0.0
    r_novel = 1.0 - float(max_sim)
    r_novel_list.append(r_novel)

    if (idx + 1) % 100 == 0 or (idx + 1) == len(exp_smiles):
        print(f"[INFO] Processed {idx+1}/{len(exp_smiles)} expanded molecules", end="\r")

print()  # newline after progress

df_exp["R_novel"] = r_novel_list

df_exp.to_csv(OUT_CSV, index=False)
print(f"[OK] Saved expanded set with novelty to: {OUT_CSV}")
