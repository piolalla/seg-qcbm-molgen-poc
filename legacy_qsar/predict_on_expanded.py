# -*- coding: utf-8 -*-
# Load the trained QSAR classifier and predict P(active) for an expanded dataset.
#
# Usage:
#   python predict_on_expanded.py --expanded expanded_candidates_scored.csv \
#                                 --model qsar_ecfp4.joblib \
#                                 --out expanded_with_p.csv

import argparse
import numpy as np
import pandas as pd
from joblib import load
from rdkit import Chem
from rdkit.Chem import AllChem

def ecfp4(smiles, nBits=2048, radius=2):
    """Compute ECFP4 fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def main(args):
    df = pd.read_csv(args.expanded)

    # Identify SMILES column
    if "smiles" in df.columns:
        smi_col = "smiles"
    elif "canonical_smiles" in df.columns:
        smi_col = "canonical_smiles"
    else:
        raise ValueError("No SMILES column found (expected 'smiles' or 'canonical_smiles').")

    X = np.vstack([ecfp4(s) for s in df[smi_col].astype(str)])

    clf = load(args.model)
    p = clf.predict_proba(X)[:, 1]

    df["P_pred"] = p
    df.to_csv(args.out, index=False)
    print(f"[OK] Saved predictions to: {args.out}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="expanded_with_p.csv")
    args = ap.parse_args()
    main(args)
