# -*- coding: utf-8 -*-
# Auto-generated QSAR pipeline script (English comments only)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os
import json
import numpy as np
import pandas as pd
from joblib import dump, load
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

DATA_DIR = "data"
CLEAN_CSV = os.path.join(DATA_DIR, "clean_kras_g12d.csv")
EXPANDED_CSV = os.path.join(DATA_DIR, "expanded_candidates_scored.csv")
MODEL_OUT = os.path.join(DATA_DIR, "qsar_ecfp4.joblib")
REPORT_OUT = os.path.join(DATA_DIR, "qsar_report.json")
EXPANDED_OUT = os.path.join(DATA_DIR, "expanded_with_p.csv")

def ecfp4(smiles, nBits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core) if core else None

def scaffold_split(df, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_scaf"] = df["canonical_smiles"].apply(get_scaffold).fillna("NONE")

    buckets = {}
    for idx, s in df["_scaf"].items():
        buckets.setdefault(s, []).append(idx)

    scaffolds = list(buckets.keys())
    rng.shuffle(scaffolds)

    test_idx, target = [], int(len(df) * test_frac)
    for sc in scaffolds:
        if len(test_idx) >= target:
            break
        test_idx.extend(buckets[sc])

    train = df.loc[~df.index.isin(test_idx)].reset_index(drop=True)
    test = df.loc[df.index.isin(test_idx)].reset_index(drop=True)
    return train, test

def enrichment_factor(y, score, k=0.01):
    n = len(y)
    top_k = max(1, int(k * n))
    idx = np.argsort(score)[::-1][:top_k]
    hits = y[idx].sum()
    expected = top_k * (y.sum() / n + 1e-12)
    return float(hits / (expected + 1e-12))

def train_qsar():
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(f"Cannot find {CLEAN_CSV}")
    df = pd.read_csv(CLEAN_CSV)
    df = df.dropna(subset=["canonical_smiles", "label"])
    df["label"] = df["label"].astype(int)

    train, test = scaffold_split(df)

    X_tr = np.vstack([ecfp4(s) for s in train["canonical_smiles"]])
    y_tr = train["label"].values
    X_te = np.vstack([ecfp4(s) for s in test["canonical_smiles"]])
    y_te = test["label"].values

    base = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf = CalibratedClassifierCV(base, cv=5, method="sigmoid")
    clf.fit(X_tr, y_tr)

    p = clf.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_te, p)
    pr = average_precision_score(y_te, p)
    ef1 = enrichment_factor(y_te, p, 0.01)
    ef5 = enrichment_factor(y_te, p, 0.05)

    print("[RESULT] ROC-AUC:", roc)
    print("[RESULT] PR-AUC:", pr)
    print("[RESULT] EF@1%:", ef1)
    print("[RESULT] EF@5%:", ef5)

    os.makedirs(DATA_DIR, exist_ok=True)
    dump(clf, MODEL_OUT)

    report = {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "ef1": float(ef1),
        "ef5": float(ef5),
        "n_train": int(len(train)),
        "n_test": int(len(test))
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)
    print("[OK] Saved model to:", MODEL_OUT)
    print("[OK] Saved report to:", REPORT_OUT)

def predict_on_expanded():
    if not os.path.exists(EXPANDED_CSV):
        raise FileNotFoundError(f"Cannot find {EXPANDED_CSV}")
    if not os.path.exists(MODEL_OUT):
        raise FileNotFoundError(f"Cannot find model file {MODEL_OUT}")

    df = pd.read_csv(EXPANDED_CSV)
    if "smiles" in df.columns:
        smi_col = "smiles"
    elif "canonical_smiles" in df.columns:
        smi_col = "canonical_smiles"
    else:
        raise ValueError("No SMILES column found (expected 'smiles' or 'canonical_smiles').")

    X = np.vstack([ecfp4(s) for s in df[smi_col].astype(str)])
    clf = load(MODEL_OUT)
    df["P_pred"] = clf.predict_proba(X)[:, 1]

    df.to_csv(EXPANDED_OUT, index=False)
    print("[OK] Saved predictions to:", EXPANDED_OUT)

if __name__ == "__main__":
    train_qsar()
    predict_on_expanded()
