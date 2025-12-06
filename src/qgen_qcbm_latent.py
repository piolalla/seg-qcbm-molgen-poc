#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantum latent generator (non-segmented QCBM + SELFIES mutation).

- Train mode:
    * Load seed SMILES from clean_kras_g12d.csv
    * Pre-compute a simple seed-level reward based on QED + MW window
    * Train a QCBM over seed indices using random hill-climbing
- Sample mode:
    * Sample seed indices from the trained QCBM
    * Around each seed, perform SELFIES mutation to generate new molecules
    * Output unique SMILES to a CSV (to be scored later by rule-based oracle)

Usage examples:

  # Train QCBM on KRAS seeds
  python qgen_qcbm_latent.py --mode train \
      --seed-csv data/clean_kras_g12d.csv \
      --seed-smiles-col canonical_smiles

  # Sample from trained QCBM and generate new molecules
  python qgen_qcbm_latent.py --mode sample \
      --seed-csv data/clean_kras_g12d.csv \
      --seed-smiles-col canonical_smiles \
      --out-csv data/gen_qcbm_round1.csv \
      --n-samples 1000
"""
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os
import argparse
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit import ParameterVector

from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import selfies as sf


# ===================== Utility: RDKit & SELFIES =====================

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
    Compute a simple seed-level reward for each SMILES:
      reward_seed = 0.8 * QED + 0.2 * MW_window

    where MW_window is 1.0 if MW in [300, 550], decays outside.
    """
    qed_list = []
    mw_list = []
    for i, smi in enumerate(df_seeds[smiles_col].tolist()):
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

    # MW_window: 1.0 in [300, 550], soft decay outside
    mw_center_low, mw_center_high = 300.0, 550.0
    mw_window = np.ones_like(mw_arr)
    # below low: linear decay to 0 at 100
    low_mask = mw_arr < mw_center_low
    mw_window[low_mask] = np.clip((mw_arr[low_mask] - 100.0) / (mw_center_low - 100.0), 0.0, 1.0)
    # above high: linear decay to 0 at 800
    high_mask = mw_arr > mw_center_high
    mw_window[high_mask] = np.clip((800.0 - mw_arr[high_mask]) / (800.0 - mw_center_high), 0.0, 1.0)

    # Combine
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
    for layer in range(n_layers):
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
    Sample bitstrings from the QCBM using qiskit 2.x + qiskit-aer backend.run API.
    """
    backend = Aer.get_backend("aer_simulator")

    # Bind parameters
    bind_dict = {p: float(v) for p, v in zip(qc_template.parameters, params)}
    qc_bound = qc_template.assign_parameters(bind_dict, inplace=False)

    # Transpile for backend
    qc_compiled = transpile(qc_bound, backend)

    # Run on simulator
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
    val = int(bitstr, 2)
    return val % K


def mean_seed_reward_for_params(qc_template,
                                params: np.ndarray,
                                seed_rewards: np.ndarray,
                                n_shots: int = 512,
                                seed: int = 1234):
    """
    Evaluate current QCBM parameters by sampling bitstrings and using
    precomputed seed_rewards.
    """
    bitstrings = sample_bitstrings(qc_template, params, n_shots=n_shots, seed=seed)
    K = len(seed_rewards)
    vals = []
    for bs in bitstrings:
        idx = bitstring_to_index(bs, K)
        vals.append(seed_rewards[idx])

    if not vals:
        return 0.0
    return float(np.mean(vals))


# ===================== Training (hill-climbing) =====================

def train_qcbm_latent(args):
    # Load seeds
    df_seeds = pd.read_csv(args.seed_csv)
    if args.seed_smiles_col not in df_seeds.columns:
        raise ValueError(f"Column '{args.seed_smiles_col}' not found in {args.seed_csv}")
    df_seeds = df_seeds.dropna(subset=[args.seed_smiles_col]).reset_index(drop=True)
    K = len(df_seeds)
    print("============================================")
    print(" QCBM latent training (non-segmented)")
    print("============================================")
    print(f"[INFO] Loaded {K} seed molecules from {args.seed_csv}")
    print(f"[INFO] Seed SMILES column: {args.seed_smiles_col}")

    # Compute seed-level reward once
    print("[INFO] Pre-computing seed-level QED + MW-based rewards...")
    qed_arr, mw_arr, seed_rewards = compute_seed_rewards(df_seeds, args.seed_smiles_col)
    print(f"[INFO] Seed QED: mean={qed_arr.mean():.3f}, std={qed_arr.std():.3f}")
    print(f"[INFO] Seed MW:  mean={mw_arr.mean():.1f}")
    print(f"[INFO] Example seed reward range: min={seed_rewards.min():.3f}, max={seed_rewards.max():.3f}")

    # Build QCBM
    n_qubits = args.n_qubits
    n_layers = args.n_layers
    qc_template, param_vec = build_qcbm(n_qubits, n_layers)
    n_params = len(param_vec)
    print(f"[INFO] QCBM: {n_qubits} qubits, {n_layers} layers, {n_params} parameters")

    rng = np.random.default_rng(args.seed)
    if args.init == "zeros":
        params = np.zeros(n_params, dtype=float)
    else:
        params = rng.normal(loc=0.0, scale=0.2, size=n_params).astype(float)

    # Initial evaluation
    print("[INFO] Evaluating initial parameters...")
    best_reward = mean_seed_reward_for_params(
        qc_template, params, seed_rewards,
        n_shots=args.n_shots, seed=args.seed
    )
    print(f"[INIT] mean seed reward = {best_reward:.4f}")

    sigma = args.sigma
    for it in range(1, args.n_iters + 1):
        proposal = params + rng.normal(loc=0.0, scale=sigma, size=n_params)
        cand_reward = mean_seed_reward_for_params(
            qc_template, proposal, seed_rewards,
            n_shots=args.n_shots, seed=args.seed + it
        )
        improved = cand_reward > best_reward
        if improved:
            params = proposal
            best_reward = cand_reward

        print(f"[ITER {it:03d}] "
              f"best={best_reward:.4f}  cand={cand_reward:.4f}  "
              f"sigma={sigma:.3f}  {'ACCEPT' if improved else 'reject'}")

        sigma = max(args.min_sigma, sigma * args.decay_sigma)

    # Save
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "qcbm_latent_params.npz")
    np.savez(out_path,
             params=params,
             n_qubits=n_qubits,
             n_layers=n_layers)
    print("============================================")
    print(f"[OK] Training finished. Saved parameters to: {out_path}")
    print("============================================")


# ===================== Sampling (generate molecules) =====================

def load_qcbm_latent_params(model_dir: str):
    path = os.path.join(model_dir, "qcbm_latent_params.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"QCBM latent params not found at {path}")
    data = np.load(path, allow_pickle=True)
    params = data["params"]
    n_qubits = int(data["n_qubits"])
    n_layers = int(data["n_layers"])
    return params, n_qubits, n_layers


def sample_qcbm_latent(args):
    # Load seeds
    df_seeds = pd.read_csv(args.seed_csv)
    if args.seed_smiles_col not in df_seeds.columns:
        raise ValueError(f"Column '{args.seed_smiles_col}' not found in {args.seed_csv}")
    df_seeds = df_seeds.dropna(subset=[args.seed_smiles_col]).reset_index(drop=True)
    K = len(df_seeds)
    print("============================================")
    print(" QCBM latent sampling (non-segmented)")
    print("============================================")
    print(f"[INFO] Loaded {K} seed molecules from {args.seed_csv}")
    print(f"[INFO] Seed SMILES column: {args.seed_smiles_col}")

    # Load QCBM params
    params, n_qubits, n_layers = load_qcbm_latent_params(args.model_dir)
    print(f"[INFO] Loaded QCBM params from {args.model_dir}")
    print(f"[INFO] QCBM: {n_qubits} qubits, {n_layers} layers, {len(params)} parameters")

    qc_template, _ = build_qcbm(n_qubits, n_layers)

    target_n = args.n_samples
    max_batches = args.max_batches
    batch_shots = args.n_shots
    rng = np.random.default_rng(args.seed)

    seen = set()
    smiles_list = []
    seed_idx_list = []
    seed_reward_list = []

    # Precompute seed-level rewards for logging (same as in training)
    _, _, seed_rewards = compute_seed_rewards(df_seeds, args.seed_smiles_col)

    print(f"[INFO] Target: {target_n} unique molecules")
    print(f"[INFO] Each batch: {batch_shots} shots, up to {max_batches} batches")

    for batch in range(1, max_batches + 1):
        print("--------------------------------------------")
        print(f"[BATCH {batch}] Sampling from QCBM...")
        bitstrings = sample_bitstrings(
            qc_template, params,
            n_shots=batch_shots,
            seed=args.seed + 1000 + batch
        )
        print(f"[BATCH {batch}] Collected {len(bitstrings)} bitstrings")

        for bs in bitstrings:
            idx = bitstring_to_index(bs, K)
            seed_smi = df_seeds.iloc[idx][args.seed_smiles_col]
            # encode -> mutate -> decode
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
            seed_idx_list.append(int(idx))
            seed_reward_list.append(float(seed_rewards[idx]))

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
        "seed_idx": seed_idx_list,
        "seed_reward_qed_mw": seed_reward_list
    })
    df_out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved generated molecules to: {args.out_csv}")
    print("============================================")


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(
        description="Non-segmented quantum latent generator using QCBM + SELFIES mutation."
    )
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True)

    # Common
    parser.add_argument("--seed-csv", type=str, default="data/clean_kras_g12d.csv",
                        help="CSV file containing seed SMILES.")
    parser.add_argument("--seed-smiles-col", type=str, default="canonical_smiles",
                        help="Column name for seed SMILES.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed.")

    # QCBM hyperparameters (both modes need these)
    parser.add_argument("--n-qubits", type=int, default=10,
                        help="Number of qubits in QCBM (used when building new circuits).")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of QCBM layers.")

    # Train-specific
    parser.add_argument("--n-shots", type=int, default=512,
                        help="Shots per evaluation when estimating mean seed reward.")
    parser.add_argument("--n-iters", type=int, default=30,
                        help="Number of hill-climbing iterations.")
    parser.add_argument("--sigma", type=float, default=0.2,
                        help="Initial std-dev for parameter perturbations.")
    parser.add_argument("--min-sigma", type=float, default=0.02,
                        help="Minimum std-dev for parameter perturbations.")
    parser.add_argument("--decay-sigma", type=float, default=0.95,
                        help="Decay factor for sigma per iteration.")
    parser.add_argument("--init", type=str, choices=["zeros", "normal"], default="normal",
                        help="Initialization method for QCBM parameters.")
    parser.add_argument("--outdir", type=str, default="models/qcbm_latent",
                        help="Directory to save trained QCBM latent parameters.")

    # Sample-specific
    parser.add_argument("--model-dir", type=str, default="models/qcbm_latent",
                        help="Directory where qcbm_latent_params.npz is stored.")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of unique molecules to generate.")
    parser.add_argument("--max-batches", type=int, default=50,
                        help="Maximum number of sampling batches.")
    parser.add_argument("--num-mutations", type=int, default=1,
                        help="Number of SELFIES mutations per generated molecule.")
    parser.add_argument("--out-csv", type=str, default="data/gen_qcbm_round1.csv",
                        help="Output CSV for generated molecules.")
    # For sampling, n-shots is also used (overload)
    # (we reuse the same argument name; for sample mode it will mean shots per batch)

    args = parser.parse_args()

    if args.mode == "train":
        train_qcbm_latent(args)
    elif args.mode == "sample":
        sample_qcbm_latent(args)
    else:
        raise ValueError("Unknown mode.")


if __name__ == "__main__":
    main()
