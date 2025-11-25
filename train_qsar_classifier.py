# -*- coding: utf-8 -*-
# QSAR classifier consistent with common practice in drug discovery pipelines:
# - ECFP4 fingerprints (2048 bits, radius=2)
# - Scaffold split (Murcko scaffolds)
# - Logistic Regression with probability calibration (Platt scaling)
# - ROC-AUC, PR-AUC, EF@1%, EF@5% evaluation
#
# Usage:
#   python train_qsar_classifier.py --csv clean_kras_g12d.csv \
#                                   --model_out qsar_ecfp4.joblib \
#                                   --report_out qsar_report.json

import argparse
import json
import numpy as np
import pandas as pd
from joblib import dump
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# -------------------- Fingerprint utilities --------------------
def ecfp4(smiles, nBits=2048, radius=2):
    """Compute ECFP4 (Morgan) fingerprint for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# -------------------- Scaffold utilities --------------------
def get_scaffold(smiles):
    """Return Murcko scaffold SMILES for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core is None:
        return None
    return Chem.MolToSmiles(core)

def scaffold_split(df, test_frac=0.2, seed=42):
    """
    Simple scaffold split:
    - Group compounds by Murcko scaffold
    - Randomly select scaffold groups until reaching test fraction
    """
    rng = np.random.default_rng(seed)

    df = df.copy()
    df["_scaffold"] = df["canonical_smiles"].apply(get_scaffold).fillna("NONE")

    # buckets: scaffold -> list of indices
    buckets = {}
    for idx, scaf in df["_scaffold"].items():
        buckets.setdefault(scaf, []).append(idx)

    scaffolds = list(buckets.keys())
    rng.shuffle(scaffolds)

    test_idx = []
    target_n = int(len(df) * test_frac)

    for scaf in scaffolds:
        if len(test_idx) >= target_n:
            break
        test_idx.extend(buckets[scaf])

    mask = df.index.isin(test_idx)
    train_df = df.loc[~mask].reset_index(drop=True)
    test_df = df.loc[mask].reset_index(drop=True)

    return train_df, test_df

# -------------------- Training --------------------
def main(args):
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["canonical_smiles", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Scaffold split
    train_df, test_df = scaffold_split(df, test_frac=0.2, seed=42)

    # Featurization
    def make_XY(frame):
        X = np.vstack([ecfp4(s) for s in frame["canonical_smiles"]])
        y = frame["label"].values.astype(int)
        return X, y

    X_tr, y_tr = make_XY(train_df)
    X_te, y_te = make_XY(test_df)

    # Logistic Regression + probability calibration
    base = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    )
    clf = CalibratedClassifierCV(base, cv=5, method="sigmoid")
    clf.fit(X_tr, y_tr)

    # Evaluation
    p_te = clf.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_te, p_te)
    pr = average_precision_score(y_te, p_te)

    def enrichment_factor(y_true, scores, k=0.01):
        """
        EF@k:
        - Rank by score
        - Count hits in top k%
        - Compare to random expectation
        """
        n = len(y_true)
        top_k = max(1, int(n * k))
        order = np.argsort(scores)[::-1][:top_k]
        hits = y_true[order].sum()

        total_actives = y_true.sum()
        expected_hits = top_k * (total_actives / n + 1e-12)
        return float(hits / (expected_hits + 1e-12))

    ef1 = enrichment_factor(y_te, p_te, k=0.01)
    ef5 = enrichment_factor(y_te, p_te, k=0.05)

    print(f"[TEST] ROC-AUC = {roc:.3f}")
    print(f"[TEST] PR-AUC  = {pr:.3f}")
    print(f"[TEST] EF@1%   = {ef1:.2f}")
    print(f"[TEST] EF@5%   = {ef5:.2f}")
    print(f"[TEST] Positives in test = {y_te.sum()}/{len(y_te)}")

    # Save model
    dump(clf, args.model_out)

    # Save evaluation report
    report = {
        "test_roc_auc": float(roc),
        "test_pr_auc": float(pr),
        "ef@1%": float(ef1),
        "ef@5%": float(ef5),
        "n_train": len(train_df),
        "n_test": len(test_df)
    }
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Saved model to:  {args.model_out}")
    print(f"[OK] Saved report to: {args.report_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to clean_kras_g12d.csv")
    ap.add_argument("--model_out", default="qsar_ecfp4.joblib")
    ap.add_argument("--report_out", default="qsar_report.json")
    args = ap.parse_args()
    main(args)
