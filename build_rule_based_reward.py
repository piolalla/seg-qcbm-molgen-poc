#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build rule-based reward for KRAS G12D project, using ONLY stable properties:

- QED (drug-likeness)           -> f_qed
- SA (synthetic accessibility)  -> f_sa
- sim_G12D (similarity to known actives) -> f_sim
- novelty (distance to train set)        -> f_novelty
- physchem_window (MW window, simple)    -> f_physchem

No pX / P_pred are used in the reward. They stay in clean_kras_g12d.csv for analysis only.

Inputs (defaults under data/):
  - clean_kras_g12d.csv
  - expanded_candidates_scored.csv

Output:
  - rule_based_reward.csv
"""
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem, DataStructs


# ============== Basic helpers ==============

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


# ============== Feature shaping ==============

def f_qed(q):
    """
    QED in [0,1] -> f_qed in [0,1].
    强化中段 (0.4~0.9)，太低/太高打折。
    """
    if q is None or np.isnan(q):
        return 0.0
    q = float(q)
    if q <= 0.3:
        return 0.0
    elif q <= 0.4:
        # 0.3~0.4: 从 0 线性涨到 0.6
        return (q - 0.3) / (0.1) * 0.6
    elif q <= 0.8:
        # 0.4~0.8: 给高分 0.6~1.0
        return 0.6 + (q - 0.4) / (0.4) * 0.4
    elif q <= 0.95:
        # 0.8~0.95: 轻微打折，降到 0.8
        return 1.0 - (q - 0.8) / (0.15) * 0.2
    else:
        # >0.95: 再稍微扣一点，避免极端
        return 0.7


def f_sa(sa):
    """
    SA ~[1,10], lower is easier -> map to [0,1]
    使用公式: f_sa = 1 - (SA - 2) / 8, 然后 clip。
    """
    if sa is None or np.isnan(sa):
        return 0.0
    sa = float(sa)
    val = 1.0 - (sa - 2.0) / 8.0
    return float(max(0.0, min(1.0, val)))


def f_sim_donut(sim):
    """
    sim_G12D 的“甜甜圈”形状映射:
    - sim < 0.2  -> 0
    - 0.2~0.4    -> 线性涨到 1
    - 0.4~0.8    -> 保持 1
    - 0.8~0.9    -> 线性降到 0.7
    - >=0.9      -> 0.5 (太像已知活性，略惩罚避免模式坍缩)
    """
    if sim is None or np.isnan(sim):
        return 0.0
    s = float(sim)
    if s < 0.2:
        return 0.0
    elif s < 0.4:
        return (s - 0.2) / 0.2  # 0~1
    elif s < 0.8:
        return 1.0
    elif s < 0.9:
        # 0.8~0.9: 1 -> 0.7
        return 1.0 - (s - 0.8) / 0.1 * 0.3
    else:
        return 0.5


def f_novelty_donut(novelty_raw):
    """
    novelty_raw = 1 - max_sim_train ∈ [0,1]
    希望中等距离最好:
    - dist <= 0.1: 0 (几乎抄训练集)
    - 0.1~0.3: 线性涨到 1
    - 0.3~0.6: 保持 1
    - 0.6~0.8: 线性降到 0
    - >0.8: 0 (太远，可能是乱飞/出分布)
    """
    if novelty_raw is None or np.isnan(novelty_raw):
        return 0.0
    d = float(novelty_raw)
    if d <= 0.1:
        return 0.0
    elif d <= 0.3:
        return (d - 0.1) / 0.2  # 0~1
    elif d <= 0.6:
        return 1.0
    elif d <= 0.8:
        # 0.6~0.8: 1 -> 0
        return 1.0 - (d - 0.6) / 0.2
    else:
        return 0.0


def physchem_window_score(mol, mw_min=300.0, mw_max=650.0):
    """
    简单 physchem_window: 目前只用 MW。
    - MW < mw_min - 50 或 > mw_max + 50: 0
    - [mw_min-50, mw_min] 和 [mw_max, mw_max+50]: 线性 0->0.7
    - [mw_min, mw_max]: 0.7~1.0 平滑升降
    （可以以后再加 logP / HBD / HBA / TPSA 约束）
    """
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
        # lo1~hi1: 0->0.7
        return (mw - lo1) / (hi1 - lo1) * 0.7
    if mw >= lo2:
        # lo2~hi2: 0.7->0
        return (hi2 - mw) / (hi2 - lo2) * 0.7
    # 中间 [mw_min, mw_max]: 0.7~1.0 的缓升/缓降，这里简单给 1.0
    return 1.0


# ============== Main ==============

def build_rule_based_reward(train_csv: str,
                            expanded_csv: str,
                            out_csv: str,
                            w_qed=1.0,
                            w_sa=1.0,
                            w_sim=1.0,
                            w_nov=1.0,
                            w_phys=0.5):
    print(f"[INFO] Loading train set from {train_csv}")
    df_train = pd.read_csv(train_csv)
    print(f"[INFO] Loading expanded set from {expanded_csv}")
    df_exp = pd.read_csv(expanded_csv)

    # 确保有 SMILES 列
    if "canonical_smiles" in df_train.columns:
        train_smiles = df_train["canonical_smiles"].astype(str).tolist()
    elif "smiles" in df_train.columns:
        train_smiles = df_train["smiles"].astype(str).tolist()
    else:
        raise ValueError("Train CSV must contain 'canonical_smiles' or 'smiles' column.")

    if "smiles" not in df_exp.columns:
        raise ValueError("Expanded CSV must contain 'smiles' column.")

    # 训练集: 分子 & 指纹
    print("[INFO] Computing train fingerprints...")
    train_mols = [to_mol(s) for s in train_smiles]
    train_fps = [ecfp4_fp(m) for m in train_mols if m is not None]

    # 活性子集: label == 1
    if "label" in df_train.columns:
        actives_mask = df_train["label"] == 1
        active_smiles = df_train.loc[actives_mask, "canonical_smiles" if "canonical_smiles" in df_train.columns else "smiles"].astype(str).tolist()
        active_mols = [to_mol(s) for s in active_smiles]
        active_fps = [ecfp4_fp(m) for m in active_mols if m is not None]
        if not active_fps:
            print("[WARN] No actives fingerprints found, fallback to all train FPS for sim_G12D.")
            active_fps = train_fps
    else:
        print("[WARN] No 'label' column in train set; use all train molecules as actives for similarity.")
        active_fps = train_fps

    # 扩增集: 对每个分子算 fp、sim、novelty、physchem
    sim_list = []
    nov_raw_list = []
    phys_list = []
    mw_list = []

    print("[INFO] Scoring expanded molecules...")
    for smi in tqdm(df_exp["smiles"].astype(str).tolist(), desc="Expanded scoring"):
        m = to_mol(smi)
        fp = ecfp4_fp(m)
        # sim_G12D: 对 active_fps 的 max Tanimoto
        sim_g12d = tanimoto_max(fp, active_fps)
        # novelty_raw: 1 - max_sim_train
        max_sim_train = tanimoto_max(fp, train_fps)
        novelty_raw = 1.0 - max_sim_train
        # physchem_window
        phys = physchem_window_score(m)
        mw = Descriptors.MolWt(m) if m is not None else np.nan

        sim_list.append(sim_g12d)
        nov_raw_list.append(novelty_raw)
        phys_list.append(phys)
        mw_list.append(mw)

    df_exp["sim_g12d"] = sim_list
    df_exp["novelty_raw"] = nov_raw_list
    df_exp["physchem_score"] = phys_list
    df_exp["molwt"] = mw_list

    # 如果扩增集已经有 QED/SA，用它；否则设 NaN
    if "QED" not in df_exp.columns:
        df_exp["QED"] = np.nan
    if "SA" not in df_exp.columns:
        df_exp["SA"] = np.nan

    # 映射到 f_* 特征
    f_qed_list = [f_qed(q) for q in df_exp["QED"]]
    f_sa_list = [f_sa(s) for s in df_exp["SA"]]
    f_sim_list = [f_sim_donut(s) for s in df_exp["sim_g12d"]]
    f_nov_list = [f_novelty_donut(d) for d in df_exp["novelty_raw"]]
    f_phys_list = df_exp["physchem_score"].tolist()

    df_exp["f_qed"] = f_qed_list
    df_exp["f_sa"] = f_sa_list
    df_exp["f_sim"] = f_sim_list
    df_exp["f_novelty"] = f_nov_list
    df_exp["f_physchem"] = f_phys_list

    # 最终 reward
    reward = (
        w_qed * df_exp["f_qed"].values +
        w_sa * df_exp["f_sa"].values +
        w_sim * df_exp["f_sim"].values +
        w_nov * df_exp["f_novelty"].values +
        w_phys * df_exp["f_physchem"].values
    )
    df_exp["reward"] = reward

    # 保存
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_exp.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved rule-based reward to {out_csv}")
    print("Sample columns:", df_exp.columns.tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Build rule-based reward for KRAS G12D (no pX/P_pred)."
    )
    parser.add_argument("--train",
                        type=str,
                        default="data/clean_kras_g12d.csv",
                        help="Path to clean_kras_g12d.csv")
    parser.add_argument("--expanded",
                        type=str,
                        default="data/expanded_candidates_scored.csv",
                        help="Path to expanded_candidates_scored.csv")
    parser.add_argument("--out",
                        type=str,
                        default="data/rule_based_reward.csv",
                        help="Output CSV path")
    parser.add_argument("--w-qed", type=float, default=1.0)
    parser.add_argument("--w-sa", type=float, default=1.0)
    parser.add_argument("--w-sim", type=float, default=1.0)
    parser.add_argument("--w-nov", type=float, default=1.0)
    parser.add_argument("--w-phys", type=float, default=0.5)

    args = parser.parse_args()

    build_rule_based_reward(
        train_csv=args.train,
        expanded_csv=args.expanded,
        out_csv=args.out,
        w_qed=args.w_qed,
        w_sa=args.w_sa,
        w_sim=args.w_sim,
        w_nov=args.w_nov,
        w_phys=args.w_phys
    )


if __name__ == "__main__":
    main()
