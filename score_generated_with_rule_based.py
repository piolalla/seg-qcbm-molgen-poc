#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Score generator outputs with the same rule-based KRAS G12D oracle.

Inputs:
  --train     clean_kras_g12d.csv   (must contain label + canonical_smiles)
  --generated generated_samples.csv (must contain 'smiles' column; may or may not have QED/SA)

Outputs:
  --out       generated_scored.csv  (default)

Columns added:
  QED, SA (if missing, will be computed)
  sim_g12d, novelty_raw, physchem_score, molwt
  f_qed, f_sa, f_sim, f_novelty, f_physchem
  reward
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem import AllChem, DataStructs


# ================== basic mol / fp helpers ==================

def to_mol(smi: str):
    if not isinstance(smi, str) or not smi:
        return None
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def ecfp4_fp(mol, nbits: int = 2048):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)


def tanimoto_max(fp, fp_list):
    if fp is None or not fp_list:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, fp_list)
    if not sims:
        return 0.0
    return float(max(sims))


# ================== SA + QED 计算（和预处理保持一致） ==================

def sa_score_simple(mol) -> float:
    """简化版 SA，~[1,10]，越低越易合成。"""
    if mol is None:
        return np.nan
    try:
        mw = Descriptors.MolWt(mol)
        ri = mol.GetRingInfo()
        nring = ri.NumRings()
        nrot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        n_hetero = sum(1 for a in mol.GetAtoms()
                       if a.GetAtomicNum() not in (1, 6))
        n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        natoms = mol.GetNumAtoms()
        arom_frac = (n_arom / natoms) if natoms > 0 else 0.0

        raw = (
            0.002 * mw +
            0.80 * nring +
            0.15 * nrot +
            0.25 * n_hetero +
            0.01 * tpsa +
            1.00 * arom_frac -
            0.50 * frac_csp3
        )
        raw = max(0.0, raw)
        score = 1.0 + 9.0 * (raw / (raw + 6.0))
        return float(score)
    except Exception:
        return np.nan


def ensure_qed_sa(df, smiles_col="smiles"):
    """若 df 中没有 QED/SA，就按 smiles 统一算一遍。"""
    if ("QED" in df.columns) and ("SA" in df.columns):
        return df

    qed_list, sa_list = [], []
    for smi in tqdm(df[smiles_col].astype(str).tolist(), desc="Compute QED/SA for generated"):
        m = to_mol(smi)
        if m is None:
            qed_list.append(np.nan)
            sa_list.append(np.nan)
            continue
        try:
            q = float(QED.qed(m))
        except Exception:
            q = np.nan
        s = sa_score_simple(m)
        qed_list.append(q)
        sa_list.append(s)
    if "QED" not in df.columns:
        df["QED"] = qed_list
    else:
        df["QED"] = df["QED"].fillna(pd.Series(qed_list))
    if "SA" not in df.columns:
        df["SA"] = sa_list
    else:
        df["SA"] = df["SA"].fillna(pd.Series(sa_list))
    return df


# ================== feature shaping (和 rule_based_reward 保持一致) ==================

def f_qed(q):
    if q is None or np.isnan(q):
        return 0.0
    q = float(q)
    if q <= 0.3:
        return 0.0
    elif q <= 0.4:
        return (q - 0.3) / 0.1 * 0.6
    elif q <= 0.8:
        return 0.6 + (q - 0.4) / 0.4 * 0.4
    elif q <= 0.95:
        return 1.0 - (q - 0.8) / 0.15 * 0.2
    else:
        return 0.7


def f_sa(sa):
    if sa is None or np.isnan(sa):
        return 0.0
    sa = float(sa)
    val = 1.0 - (sa - 2.0) / 8.0
    return float(max(0.0, min(1.0, val)))


def f_sim_donut(sim):
    if sim is None or np.isnan(sim):
        return 0.0
    s = float(sim)
    if s < 0.2:
        return 0.0
    elif s < 0.4:
        return (s - 0.2) / 0.2
    elif s < 0.8:
        return 1.0
    elif s < 0.9:
        return 1.0 - (s - 0.8) / 0.1 * 0.3
    else:
        return 0.5


def f_novelty_donut(novelty_raw):
    if novelty_raw is None or np.isnan(novelty_raw):
        return 0.0
    d = float(novelty_raw)
    if d <= 0.1:
        return 0.0
    elif d <= 0.3:
        return (d - 0.1) / 0.2
    elif d <= 0.6:
        return 1.0
    elif d <= 0.8:
        return 1.0 - (d - 0.6) / 0.2
    else:
        return 0.0


def physchem_window_score(mol, mw_min=300.0, mw_max=650.0):
    if mol is None:
        return 0.0
    mw = Descriptors.MolWt(mol)
    lo1 = mw_min - 50.0
    hi1 = mw_min
    lo2 = mw_max
    hi2 = mw_max + 50.0

    if mw <= lo1 or mw >= hi2:
        return 0.0
    if mw <= hi1:
        return (mw - lo1) / (hi1 - lo1) * 0.7
    if mw >= lo2:
        return (hi2 - mw) / (hi2 - lo2) * 0.7
    return 1.0


# ================== main scoring logic ==================

def score_generated(train_csv: str,
                    gen_csv: str,
                    out_csv: str,
                    w_qed=1.2,
                    w_sa=0.8,
                    w_sim=1.2,
                    w_nov=0.3,
                    w_phys=0.1):

    print(f"[INFO] Load train from {train_csv}")
    df_train = pd.read_csv(train_csv)

    print(f"[INFO] Load generated from {gen_csv}")
    df_gen = pd.read_csv(gen_csv)

    if "canonical_smiles" in df_train.columns:
        train_smiles = df_train["canonical_smiles"].astype(str).tolist()
    elif "smiles" in df_train.columns:
        train_smiles = df_train["smiles"].astype(str).tolist()
    else:
        raise ValueError("Train CSV must have 'canonical_smiles' or 'smiles' column.")

    if "smiles" not in df_gen.columns:
        raise ValueError("Generated CSV must have 'smiles' column.")

    # train fingerprints
    print("[INFO] Compute train fingerprints...")
    train_mols = [to_mol(s) for s in train_smiles]
    train_fps = [ecfp4_fp(m) for m in train_mols if m is not None]

    # active subset
    if "label" in df_train.columns:
        actives_mask = df_train["label"] == 1
        active_smiles = df_train.loc[actives_mask, "canonical_smiles" if "canonical_smiles" in df_train.columns else "smiles"].astype(str).tolist()
        active_mols = [to_mol(s) for s in active_smiles]
        active_fps = [ecfp4_fp(m) for m in active_mols if m is not None]
        if not active_fps:
            print("[WARN] No actives found, fallback to all train for sim_g12d.")
            active_fps = train_fps
    else:
        print("[WARN] Train has no 'label'; using all train for sim_g12d.")
        active_fps = train_fps

    # ensure generated QED/SA
    df_gen = ensure_qed_sa(df_gen, smiles_col="smiles")

    # score each generated molecule
    sim_list = []
    nov_raw_list = []
    phys_list = []
    mw_list = []

    print("[INFO] Scoring generated molecules...")
    for smi in tqdm(df_gen["smiles"].astype(str).tolist(), desc="Score generated"):
        m = to_mol(smi)
        fp = ecfp4_fp(m)

        sim_g12d = tanimoto_max(fp, active_fps)
        max_sim_train = tanimoto_max(fp, train_fps)
        novelty_raw = 1.0 - max_sim_train
        phys = physchem_window_score(m)
        mw = Descriptors.MolWt(m) if m is not None else np.nan

        sim_list.append(sim_g12d)
        nov_raw_list.append(novelty_raw)
        phys_list.append(phys)
        mw_list.append(mw)

    df_gen["sim_g12d"] = sim_list
    df_gen["novelty_raw"] = nov_raw_list
    df_gen["physchem_score"] = phys_list
    df_gen["molwt"] = mw_list

    # f_* features
    df_gen["f_qed"] = [f_qed(q) for q in df_gen["QED"]]
    df_gen["f_sa"] = [f_sa(s) for s in df_gen["SA"]]
    df_gen["f_sim"] = [f_sim_donut(s) for s in df_gen["sim_g12d"]]
    df_gen["f_novelty"] = [f_novelty_donut(d) for d in df_gen["novelty_raw"]]
    df_gen["f_physchem"] = df_gen["physchem_score"]

    reward = (
        w_qed * df_gen["f_qed"].values +
        w_sa * df_gen["f_sa"].values +
        w_sim * df_gen["f_sim"].values +
        w_nov * df_gen["f_novelty"].values +
        w_phys * df_gen["f_physchem"].values
    )
    df_gen["reward"] = reward

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df_gen.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved scored generated set to {out_csv}")
    print("Columns:", df_gen.columns.tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Score generator outputs with rule-based KRAS G12D oracle."
    )
    parser.add_argument("--train", type=str,
                        default="data/clean_kras_g12d.csv",
                        help="Path to clean_kras_g12d.csv")
    parser.add_argument("--generated", type=str,
                        default="data/generated_samples.csv",
                        help="Path to generator output CSV with 'smiles' column")
    parser.add_argument("--out", type=str,
                        default="data/generated_scored.csv",
                        help="Output CSV path")
    parser.add_argument("--w-qed", type=float, default=1.2)
    parser.add_argument("--w-sa", type=float, default=0.8)
    parser.add_argument("--w-sim", type=float, default=1.2)
    parser.add_argument("--w-nov", type=float, default=0.3)
    parser.add_argument("--w-phys", type=float, default=0.1)

    args = parser.parse_args()

    score_generated(
        train_csv=args.train,
        gen_csv=args.generated,
        out_csv=args.out,
        w_qed=args.w_qed,
        w_sa=args.w_sa,
        w_sim=args.w_sim,
        w_nov=args.w_nov,
        w_phys=args.w_phys,
    )


if __name__ == "__main__":
    main()
