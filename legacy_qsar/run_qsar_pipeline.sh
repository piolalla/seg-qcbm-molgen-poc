#!/usr/bin/env bash
# ============================================================
# Run QSAR + Reward pipeline:
#   1) Check and install required Python libraries
#   2) Generate and run QSAR pipeline (train + P_pred)
#   3) Compute novelty (R_novel)
#   4) Merge reward (P_pred + QED + SA + R_novel)
#   5) Select Top-K molecules for fine-tuning
#
# This script assumes:
#   - data/clean_kras_g12d.csv exists
#   - data/expanded_candidates_scored.csv exists
# ============================================================

set -e

echo "============================================"
echo " Step 0: Checking Python environment"
echo "============================================"

# packages to check
install_pkg() {
    pkg=$1
    pipname=$2
    echo "[INFO] Installing Python package: ${pipname}"
    python3 -m pip install --quiet "$pipname" || true
}

check_or_install() {
    pkg=$1
    pipname=$2

    python3 - <<EOF
import importlib
try:
    importlib.import_module("${pkg}")
    print("[OK] Python package '${pkg}' is installed.")
except ImportError:
    print("[MISS] Package '${pkg}' is missing.")
    raise SystemExit(1)
EOF

    if [[ $? != 0 ]]; then
        install_pkg "$pkg" "$pipname"
    fi
}

echo "============================================"
echo " Step 1: Ensuring required packages are installed"
echo "============================================"

check_or_install "pandas" "pandas"
check_or_install "numpy" "numpy"
check_or_install "sklearn" "scikit-learn"
check_or_install "joblib" "joblib"

# RDKit: try pip first, then conda
echo "[INFO] Checking RDKit..."
python3 - <<EOF
import importlib
try:
    importlib.import_module("rdkit")
    print("[OK] RDKit is installed.")
except ImportError:
    print("[MISS] RDKit is missing.")
    raise SystemExit(1)
EOF

if [[ $? != 0 ]]; then
    echo "[INFO] Attempting pip installation for rdkit-pypi"
    python3 -m pip install --quiet rdkit-pypi || true

    python3 - <<EOF
import importlib
try:
    importlib.import_module("rdkit")
    print("[OK] RDKit installed via pip.")
except ImportError:
    print("[FAIL] RDKit pip install failed. Trying conda...")
    raise SystemExit(1)
EOF

    if [[ $? != 0 ]]; then
        if command -v conda &> /dev/null; then
            echo "[INFO] Installing RDKit using conda-forge..."
            conda install -y -c conda-forge rdkit
        else:
            echo "[ERROR] RDKit installation failed. Install manually:"
            echo "        conda install -c conda-forge rdkit"
            exit 1
        fi
    fi
fi

echo "============================================"
echo " Step 2: Generating Python QSAR pipeline (using data/ folder)"
echo "============================================"

PIPELINE="run_qsar_pipeline.py"

cat > $PIPELINE << 'EOF'
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
EOF

echo "============================================"
echo " Step 3: Running Python QSAR pipeline"
echo "============================================"

python3 $PIPELINE

echo "============================================"
echo " QSAR pipeline finished."
echo " Generated in data/:"
echo "   - qsar_ecfp4.joblib"
echo "   - qsar_report.json"
echo "   - expanded_with_p.csv"
echo "============================================"


echo "============================================"
echo " Step 4: Compute novelty (R_novel)"
echo "============================================"

python3 compute_novelty.py


echo "============================================"
echo " Step 5: Merge reward (P_pred + QED + SA + R_novel)"
echo "============================================"

python3 merge_reward.py


echo "============================================"
echo " Step 6: Select Top-K for fine-tuning"
echo "============================================"

python3 select_top_k.py \
    --input data/reward.csv \
    --out_prefix data/finetune \
    --top_k 256

echo "============================================"
echo " All steps finished successfully."
echo " Generated in data/:"
echo "   - expanded_with_p.csv"
echo "   - expanded_with_novelty.csv"
echo "   - reward.csv"
echo "   - finetune_topk.csv"
echo "   - finetune_topk.selfies.txt"
echo "   - finetune_topk.jsonl"
echo "============================================"
