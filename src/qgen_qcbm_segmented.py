#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segmented quantum latent generator using per-segment QCBMs + SELFIES mutation.

- Train mode:
    * Load segmented seeds (must contain segment_id)
    * For each segment, compute seed-level rewards
    * Train a QCBM over seed indices restricted to that segment
    * Save per-segment QCBM parameters

- Sample mode:
    * Load segmented seeds + per-segment QCBM models
    * Repeatedly:
        - choose a segment (uniformly)
        - sample bitstrings from that segment's QCBM
        - map to seed indices, then mutate SELFIES
    * Output unique SMILES with segment_id and seed-related info

Usage examples:

  # Train segmented QCBMs
  python qgen_qcbm_segmented.py --mode train \
      --seed-csv data/clean_kras_g12d_segmented.csv \
      --seed-smiles-col canonical_smiles

  # Sample from segmented QCBMs
  python qgen_qcbm_segmented.py --mode sample \
      --seed-csv data/clean_kras_g12d_segmented.csv \
      --out-csv data/gen_qcbm_segmented_round1.csv \
      --n-samples 1000
"""

import warnings
from rdkit import RDLogger

# Turn off RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Turn off Python deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Optional: suppress any RDKit user warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import math
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import selfies as sf

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit import ParameterVector


# ===================== RDKit & SELFIES utilities =====================

def to_mol(smi: str):
    """Safe SMILES -> RDKit Mol (sanitized), or None."""
    if not isinstance(smi, str) or not smi:
        return None
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m


def canonical_smiles(smi: str):
    m = to_mol(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def smiles_to_selfies_safe(smi: str):
    m = to_mol(smi)
    if m is None:
        return None
    try:
        can = Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
        return sf.encoder(can)
    except Exception:
        return None


def selfies_to_smiles_safe(s: str):
    try:
        smi = sf.decoder(s)
    except Exception:
        return None
    return canonical_smiles(smi)


def mutate_selfies(s: str,
                   num_mutations: int = 1,
                   rng: np.random.Generator = None):
    """Simple SELFIES token-level mutation."""
    if s is None:
        return None
    if rng is None:
        rng = np.random.default_rng()
    try:
        tokens = list(sf.split_selfies(s))
        if not tokens:
            return None
        alphabet = list(sf.get_alphabet_from_selfies([s]))
        if not alphabet:
            return None
    except Exception:
        return None

    num_mutations = max(1, int(num_mutations))
    for _ in range(num_mutations):
        if not tokens:
            break
        r = rng.random()
        # 70% substitution, 20% insertion, 10% deletion
        if r < 0.7:
            idx = rng.integers(0, len(tokens))
            tokens[idx] = rng.choice(alphabet)
        elif r < 0.9:
            idx = rng.integers(0, len(tokens) + 1)
            tokens.insert(idx, rng.choice(alphabet))
        else:
            if len(tokens) > 1:
                idx = rng.integers(0, len(tokens))
                tokens.pop(idx)

    try:
        return "".join(tokens)
    except Exception:
        return None


# ===================== Seed reward (QED + MW window) =====================

def compute_seed_rewards(df_seeds: pd.DataFrame,
                         smiles_col: str):
    """
    Compute seed-level reward:
      reward_seed = 0.8 * QED + 0.2 * MW_window

    where MW_window is 1.0 if MW in [300, 550], decays outside.
    """
    qed_list = []
    mw_list = []
    for smi in df_seeds[smiles_col].tolist():
        m = to_mol(smi)
        if m is None:
            qed_list.append(0.0)
            mw_list.append(0.0)
            continue
        try:
            q = float(QED.qed(m))
        except Exception:
            q = 0.0
        mw = float(Descriptors.MolWt(m))
        qed_list.append(q)
        mw_list.append(mw)

    qed_arr = np.array(qed_list, dtype=float)
    mw_arr = np.array(mw_list, dtype=float)

    mw_center_low, mw_center_high = 300.0, 550.0
    mw_window = np.ones_like(mw_arr)
    low_mask = mw_arr < mw_center_low
    mw_window[low_mask] = np.clip(
        (mw_arr[low_mask] - 100.0) / (mw_center_low - 100.0), 0.0, 1.0
    )
    high_mask = mw_arr > mw_center_high
    mw_window[high_mask] = np.clip(
        (800.0 - mw_arr[high_mask]) / (800.0 - mw_center_high), 0.0, 1.0
    )

    reward_seed = 0.8 * qed_arr + 0.2 * mw_window
    return qed_arr, mw_arr, reward_seed


# ===================== QCBM construction & sampling =====================

def build_qcbm(n_qubits: int, n_layers: int):
    """
    Hardware-efficient QCBM: RX/RY layers + CZ ring, repeated n_layers times.
    Circuit includes measure_all for qasm sampling.
    """
    n_params = n_layers * n_qubits * 2
    params = ParameterVector("theta", n_params)

    qc = QuantumCircuit(n_qubits)
    p_idx = 0
    for _layer in range(n_layers):
        # single-qubit rotations
        for q in range(n_qubits):
            qc.rx(params[p_idx], q)
            p_idx += 1
            qc.ry(params[p_idx], q)
            p_idx += 1
        # entangling CZ ring
        for q in range(n_qubits):
            qc.cz(q, (q + 1) % n_qubits)

    qc.measure_all()
    return qc, params


def sample_bitstrings(qc_template,
                      params: np.ndarray,
                      n_shots: int = 512,
                      seed: int = 1234):
    """
    Sample bitstrings from given QCBM on Aer simulator.
    """
    backend = Aer.get_backend("aer_simulator")

    bind_dict = {p: float(v) for p, v in zip(qc_template.parameters, params)}
    qc_bound = qc_template.assign_parameters(bind_dict, inplace=False)
    qc_compiled = transpile(qc_bound, backend)

    job = backend.run(
        qc_compiled,
        shots=n_shots,
        seed_simulator=seed,
    )
    result = job.result()
    counts = result.get_counts()

    bitstrings = []
    for b, c in counts.items():
        bitstrings.extend([b] * c)
    return bitstrings


def bitstring_to_index(bitstr: str, K: int) -> int:
    """Map bitstring to [0, K-1] via modulo."""
    if K <= 0:
        return 0
    val = int(bitstr, 2)
    return val % K


def mean_seed_reward_for_params(qc_template,
                                params: np.ndarray,
                                seed_rewards: np.ndarray,
                                n_shots: int = 512,
                                seed: int = 1234):
    """
    Evaluate QCBM parameters by sampling bitstrings and mapping to seed_rewards.
    """
    K = len(seed_rewards)
    if K == 0:
        return 0.0

    bitstrings = sample_bitstrings(qc_template, params, n_shots=n_shots, seed=seed)
    vals = []
    for bs in bitstrings:
        idx = bitstring_to_index(bs, K)
        vals.append(seed_rewards[idx])
    if not vals:
        return 0.0
    return float(np.mean(vals))


# ===================== Training segmented QCBMs =====================

def train_segment_qcbm(seg_id: int,
                       df_seg: pd.DataFrame,
                       global_reward: np.ndarray,
                       args,
                       rng: np.random.Generator,
                       model_dir: str):
    """
    Train QCBM for a single segment.
    df_seg: seeds restricted to this segment (with original index preserved).
    global_reward: seed_reward array over all seeds; we index with df_seg.index.
    """
    K_seg = len(df_seg)
    if K_seg == 0:
        print(f"[SEG {seg_id}] No seeds, skipping.")
        return

    print("============================================")
    print(f" SEGMENT {seg_id} QCBM TRAINING")
    print("============================================")
    print(f"[SEG {seg_id}] Number of seeds = {K_seg}")

    # seed rewards for this segment (local array)
    seed_indices_global = df_seg.index.to_numpy()
    seed_rewards_seg = global_reward[seed_indices_global]

    print(f"[SEG {seg_id}] Example seed reward range: "
          f"min={seed_rewards_seg.min():.3f}, max={seed_rewards_seg.max():.3f}")

    # ---- dynamic qubit allocation based on segment size ----
    # K_seg seeds -> log2(K_seg) bits is enough to index them,
    # then clipped into [min_qubits, max_qubits]
    if K_seg <= 1:
        needed = args.min_qubits
    else:
        needed = int(math.ceil(math.log2(K_seg)))
    n_qubits = max(args.min_qubits, min(args.max_qubits, needed))
    n_layers = args.n_layers

    print(f"[SEG {seg_id}] Using n_qubits = {n_qubits} (K_seg={K_seg}, "
          f"min={args.min_qubits}, max={args.max_qubits})")

    qc_template, param_vec = build_qcbm(n_qubits, n_layers)
    n_params = len(param_vec)
    print(f"[SEG {seg_id}] QCBM params = {n_params}")

    if args.init == "zeros":
        params = np.zeros(n_params, dtype=float)
    else:
        params = rng.normal(loc=0.0, scale=0.2, size=n_params).astype(float)

    print(f"[SEG {seg_id}] Evaluating initial parameters...")
    best_reward = mean_seed_reward_for_params(
        qc_template, params, seed_rewards_seg,
        n_shots=args.n_shots, seed=args.seed
    )
    print(f"[SEG {seg_id} INIT] mean seed reward = {best_reward:.4f}")

    sigma = args.sigma
    for it in range(1, args.n_iters + 1):
        proposal = params + rng.normal(loc=0.0, scale=sigma, size=n_params)
        cand_reward = mean_seed_reward_for_params(
            qc_template, proposal, seed_rewards_seg,
            n_shots=args.n_shots, seed=args.seed + it
        )
        improved = cand_reward > best_reward
        if improved:
            params = proposal
            best_reward = cand_reward

        print(f"[SEG {seg_id} ITER {it:03d}] "
              f"best={best_reward:.4f} cand={cand_reward:.4f} "
              f"sigma={sigma:.3f} {'ACCEPT' if improved else 'reject'}")

        sigma = max(args.min_sigma, sigma * args.decay_sigma)

    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, f"seg_{seg_id}_qcbm.npz")
    np.savez(out_path,
             params=params,
             n_qubits=n_qubits,
             n_layers=n_layers,
             segment_id=seg_id,
             seed_indices_global=seed_indices_global)
    print(f"[SEG {seg_id}] Training finished. Saved to {out_path}")


def train_segmented_qcbm(args):
    df = pd.read_csv(args.seed_csv)
    if args.seed_smiles_col not in df.columns:
        raise ValueError(f"Column '{args.seed_smiles_col}' not found in {args.seed_csv}")
    if "segment_id" not in df.columns:
        raise ValueError("segment_id column not found; please run preprocess_segments.py first.")

    print("============================================")
    print(" SEGMENTED QCBM TRAINING")
    print("============================================")

    # compute global seed rewards once
    _, _, seed_rewards = compute_seed_rewards(df, args.seed_smiles_col)

    segments = sorted(df["segment_id"].dropna().unique())
    print(f"[INFO] Found segments: {segments}")

    rng = np.random.default_rng(args.seed)

    for seg_id in segments:
        seg_id_int = int(seg_id)
        df_seg = df[df["segment_id"] == seg_id_int]
        train_segment_qcbm(seg_id_int, df_seg, seed_rewards, args, rng, args.outdir)


# ===================== Sampling from segmented QCBMs =====================

def load_segment_models(model_dir: str):
    """
    Load all seg_*.npz models from model_dir.
    Returns dict: seg_id -> dict with params, n_qubits, n_layers, seed_indices_global.
    """
    models = {}
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} not found.")

    for fname in os.listdir(model_dir):
        if not fname.startswith("seg_") or not fname.endswith("_qcbm.npz"):
            continue
        path = os.path.join(model_dir, fname)
        data = np.load(path, allow_pickle=True)
        seg_id = int(data["segment_id"])
        models[seg_id] = {
            "params": data["params"],
            "n_qubits": int(data["n_qubits"]),
            "n_layers": int(data["n_layers"]),
            "seed_indices_global": data["seed_indices_global"].astype(int),
        }
    if not models:
        raise RuntimeError(f"No seg_*_qcbm.npz models found in {model_dir}")
    print(f"[INFO] Loaded segment models: {sorted(models.keys())}")
    return models


def sample_segmented_qcbm(args):
    df_seeds = pd.read_csv(args.seed_csv)
    if args.seed_smiles_col not in df_seeds.columns:
        raise ValueError(f"Column '{args.seed_smiles_col}' not found in {args.seed_csv}")
    if "segment_id" not in df_seeds.columns:
        raise ValueError("segment_id column not found; please run preprocess_segments.py first.")

    _, _, seed_rewards = compute_seed_rewards(df_seeds, args.seed_smiles_col)

    models = load_segment_models(args.model_dir)

    # Pre-build QCBM templates for each segment
    seg_circuits = {}
    for seg_id, m in models.items():
        qc_template, _ = build_qcbm(m["n_qubits"], m["n_layers"])
        seg_circuits[seg_id] = qc_template

    segment_ids = sorted(models.keys())
    rng = np.random.default_rng(args.seed)

    target_n = args.n_samples
    max_batches = args.max_batches
    batch_shots = args.n_shots

    seen = set()
    smiles_list = []
    segment_list = []
    seed_idx_list = []
    seed_reward_list = []

    print("============================================")
    print(" SEGMENTED QCBM SAMPLING")
    print("============================================")
    print(f"[INFO] Target unique molecules: {target_n}")
    print(f"[INFO] Each batch: {batch_shots} shots, up to {max_batches} batches.")

    for batch in range(1, max_batches + 1):
        print("--------------------------------------------")
        print(f"[BATCH {batch}] Sampling...")
        # 每个 batch 从一个随机 segment 采样（后面可以改为加权）
        seg_id = int(rng.choice(segment_ids))
        model = models[seg_id]
        qc_template = seg_circuits[seg_id]

        bitstrings = sample_bitstrings(
            qc_template, model["params"],
            n_shots=batch_shots,
            seed=args.seed + 1000 + batch
        )
        seed_indices_global = model["seed_indices_global"]
        K_seg = len(seed_indices_global)
        print(f"[BATCH {batch}] Segment {seg_id}, collected {len(bitstrings)} bitstrings.")

        for bs in bitstrings:
            if K_seg == 0:
                continue
            local_idx = bitstring_to_index(bs, K_seg)
            global_idx = int(seed_indices_global[local_idx])
            seed_smi = df_seeds.iloc[global_idx][args.seed_smiles_col]
            s_seed = smiles_to_selfies_safe(seed_smi)
            if s_seed is None:
                continue
            s_mut = mutate_selfies(
                s_seed,
                num_mutations=args.num_mutations,
                rng=rng
            )
            if s_mut is None:
                continue
            smi_new = selfies_to_smiles_safe(s_mut)
            if smi_new is None:
                continue
            if smi_new in seen:
                continue
            seen.add(smi_new)
            smiles_list.append(smi_new)
            segment_list.append(seg_id)
            seed_idx_list.append(global_idx)
            seed_reward_list.append(float(seed_rewards[global_idx]))

            if len(smiles_list) % 50 == 0:
                print(f"[PROGRESS] Collected {len(smiles_list)} unique molecules so far...")

            if len(smiles_list) >= target_n:
                break

        print(f"[BATCH {batch}] Total unique molecules so far: {len(smiles_list)}")
        if len(smiles_list) >= target_n:
            break

    print("============================================")
    print(f"[INFO] Sampling finished. Unique molecules collected: {len(smiles_list)}")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out = pd.DataFrame({
        "smiles": smiles_list,
        "segment_id": segment_list,
        "seed_idx": seed_idx_list,
        "seed_reward_qed_mw": seed_reward_list,
    })
    df_out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved generated molecules to: {args.out_csv}")
    print("============================================")


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(
        description="Segmented quantum latent generator using per-segment QCBMs + SELFIES."
    )
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True)

    # Common
    parser.add_argument("--seed-csv", type=str, required=True,
                        help="CSV with seeds; must contain segment_id for segmented mode.")
    parser.add_argument("--seed-smiles-col", type=str, default="canonical_smiles",
                        help="Column name for seed SMILES.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed.")

    # QCBM hyperparameters
    parser.add_argument("--min-qubits", type=int, default=3,
                        help="Minimum number of qubits per segment QCBM.")
    parser.add_argument("--max-qubits", type=int, default=8,
                        help="Maximum number of qubits per segment QCBM.")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of layers in each segment QCBM.")

    # Train-specific
    parser.add_argument("--n-shots", type=int, default=512,
                        help="Shots per evaluation when estimating mean seed reward.")
    parser.add_argument("--n-iters", type=int, default=30,
                        help="Number of hill-climbing iterations per segment.")
    parser.add_argument("--sigma", type=float, default=0.2,
                        help="Initial std-dev for parameter perturbations.")
    parser.add_argument("--min-sigma", type=float, default=0.02,
                        help="Minimum std-dev for parameter perturbations.")
    parser.add_argument("--decay-sigma", type=float, default=0.95,
                        help="Decay factor for sigma per iteration.")
    parser.add_argument("--init", type=str, choices=["zeros", "normal"], default="normal",
                        help="Initialization method for QCBM parameters.")
    parser.add_argument("--outdir", type=str, default="models/qcbm_segmented",
                        help="Directory to save per-segment QCBM parameters.")

    # Sample-specific
    parser.add_argument("--model-dir", type=str, default="models/qcbm_segmented",
                        help="Directory where seg_*_qcbm.npz files are stored.")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of unique molecules to generate.")
    parser.add_argument("--max-batches", type=int, default=50,
                        help="Maximum number of sampling batches.")
    parser.add_argument("--num-mutations", type=int, default=1,
                        help="Number of SELFIES mutations per generated molecule.")
    parser.add_argument("--out-csv", type=str, default="data/gen_qcbm_segmented_round1.csv",
                        help="Output CSV for generated molecules.")
    # For sample mode, n-shots = shots per batch (reused)

    args = parser.parse_args()

    if args.mode == "train":
        train_segmented_qcbm(args)
    elif args.mode == "sample":
        sample_segmented_qcbm(args)
    else:
        raise ValueError("Unknown mode.")


if __name__ == "__main__":
    main()
