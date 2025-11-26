#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classical preprocessing: cluster seeds into segments and assign segment_id.

Usage:
  python preprocess_segments.py \
    --seed-csv data/clean_kras_g12d.csv \
    --smiles-col canonical_smiles \
    --n-segments 3 \
    --out-csv data/clean_kras_g12d_segmented.csv
"""
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans


def smiles_to_morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048):
    """SMILES -> Morgan fingerprint (numpy array), return None if invalid."""
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.array(list(fp), dtype=np.int8)
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Cluster seed molecules into segments and assign segment_id."
    )
    parser.add_argument("--seed-csv", type=str, required=True,
                        help="Input CSV containing seed SMILES.")
    parser.add_argument("--smiles-col", type=str, default="canonical_smiles",
                        help="Column name of SMILES in seed CSV.")
    parser.add_argument("--n-segments", type=int, default=3,
                        help="Number of segments (clusters).")
    parser.add_argument("--out-csv", type=str, required=True,
                        help="Output CSV with added 'segment_id' column.")
    args = parser.parse_args()

    print(f"[INFO] Loading seeds from {args.seed_csv}")
    df = pd.read_csv(args.seed_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in {args.seed_csv}")

    smiles_list = df[args.smiles_col].tolist()
    fps = []
    valid_idx = []

    print("[INFO] Computing Morgan fingerprints for clustering...")
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_morgan_fp(smi)
        if fp is None:
            fps.append(None)
        else:
            fps.append(fp)
            valid_idx.append(i)

    if len(valid_idx) == 0:
        raise RuntimeError("No valid SMILES for fingerprinting; cannot cluster.")

    X = np.stack([fps[i] for i in valid_idx], axis=0)

    print(f"[INFO] Clustering {len(valid_idx)} valid seeds into {args.n_segments} segments...")
    kmeans = KMeans(n_clusters=args.n_segments, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Default: mark all as -1 (invalid / unclustered)
    df["segment_id"] = -1
    df.loc[df.index[valid_idx], "segment_id"] = labels

    print("[INFO] Segment distribution:")
    print(df["segment_id"].value_counts().sort_index())

    print(f"[INFO] Writing segmented seeds to {args.out_csv}")
    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
