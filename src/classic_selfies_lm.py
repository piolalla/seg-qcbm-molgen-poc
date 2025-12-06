#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classic SELFIES-based LSTM generator for molecules.

Modes:
  - train  : Train an LSTM language model on SELFIES corpora.
  - sample : Sample molecules from a trained model and save as CSV with SMILES.

Dependencies (you probably已有):
  - selfies
  - rdkit
  - torch
  - pandas
"""

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os
import json
import argparse
import random
from typing import List, Dict

import selfies as sf
import pandas as pd
from tqdm import tqdm

from rdkit import Chem

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ========= Utils: SELFIES & SMILES =========

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>"]


def split_selfies(s: str) -> List[str]:
    """Split SELFIES string into tokens."""
    return list(sf.split_selfies(s))


def selfies_to_smiles_safe(s: str):
    try:
        smi = sf.decoder(s)
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
    except Exception:
        return None


# ========= Dataset =========

class SelfiesDataset(Dataset):
    def __init__(self, selfies_list: List[str], token_to_idx: Dict[str, int], max_len: int):
        self.selfies_list = selfies_list
        self.token_to_idx = token_to_idx
        self.max_len = max_len
        self.sos_idx = token_to_idx["<SOS>"]
        self.eos_idx = token_to_idx["<EOS>"]
        self.pad_idx = token_to_idx["<PAD>"]

    def __len__(self):
        return len(self.selfies_list)

    def __getitem__(self, idx):
        s = self.selfies_list[idx]
        tokens = split_selfies(s)
        # add SOS / EOS
        seq = ["<SOS>"] + tokens + ["<EOS>"]
        # truncate
        if len(seq) > self.max_len:
            seq = seq[: self.max_len]
            if seq[-1] != "<EOS>":
                seq[-1] = "<EOS>"
        # convert to indices
        ids = [self.token_to_idx.get(t, self.token_to_idx["<PAD>"]) for t in seq]
        # pad
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [self.pad_idx] * pad_len
        # input = seq[:-1], target = seq[1:]
        inp = ids[:-1]
        tgt = ids[1:]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# ========= Model =========

class SelfiesLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hid_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)              # (B, T, E)
        out, hidden = self.lstm(emb, hidden) # (B, T, H)
        logits = self.fc(out)               # (B, T, V)
        return logits, hidden


# ========= Training =========

def build_vocab(selfies_list: List[str]) -> Dict[str, int]:
    tokens = set()
    for s in selfies_list:
        for t in split_selfies(s):
            tokens.add(t)
    tokens = sorted(list(tokens))
    all_tokens = SPECIAL_TOKENS + tokens
    token_to_idx = {t: i for i, t in enumerate(all_tokens)}
    return token_to_idx


def train_model(args):
    # Collect SELFIES from one or more files
    selfies_all = []
    for path in args.selfies_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    selfies_all.append(s)
    print(f"[INFO] Loaded {len(selfies_all)} SELFIES sequences from {len(args.selfies_files)} files.")

    # Shuffle
    random.shuffle(selfies_all)

    # Build vocab
    token_to_idx = build_vocab(selfies_all)
    idx_to_token = {i: t for t, i in token_to_idx.items()}
    vocab_size = len(token_to_idx)
    print(f"[INFO] Vocab size = {vocab_size}")

    # Estimate max_len (with SOS/EOS)
    max_len_raw = max(len(split_selfies(s)) for s in selfies_all) + 2
    max_len = min(args.max_len, max_len_raw)
    print(f"[INFO] max_len_raw={max_len_raw}, using max_len={max_len}")

    dataset = SelfiesDataset(selfies_all, token_to_idx, max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")

    model = SelfiesLSTM(vocab_size=vocab_size,
                        emb_dim=args.emb_dim,
                        hid_dim=args.hid_dim,
                        num_layers=args.num_layers).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=token_to_idx["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_tokens = 0
        for inp, tgt in loader:
            inp = inp.to(device)   # (B, T-1)
            tgt = tgt.to(device)   # (B, T-1)
            optimizer.zero_grad()
            logits, _ = model(inp)           # (B, T-1, V)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), tgt.view(B * T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * B * T
            n_tokens += B * T

        avg_loss = total_loss / max(1, n_tokens)
        print(f"[Epoch {epoch}/{args.epochs}] avg token NLL = {avg_loss:.4f}")

        # simple checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.outdir, f"model_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

    # Save final model + vocab + config
    final_model_path = os.path.join(args.outdir, "model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    with open(os.path.join(args.outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"token_to_idx": token_to_idx, "idx_to_token": idx_to_token,
                   "max_len": max_len}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"[OK] Training finished. Saved model to {final_model_path}")


# ========= Sampling =========

def load_model(model_dir: str, device: str):
    with open(os.path.join(model_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    token_to_idx = {k: int(v) for k, v in vocab_data["token_to_idx"].items()}
    idx_to_token = {int(k): v for k, v in vocab_data["idx_to_token"].items()}
    max_len = int(vocab_data["max_len"])

    vocab_size = len(token_to_idx)
    model = SelfiesLSTM(vocab_size=vocab_size)
    state_path = os.path.join(model_dir, "model_final.pt")
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, token_to_idx, idx_to_token, max_len


def sample_sequence(model, token_to_idx, idx_to_token, max_len, temperature=1.0, device="cpu"):
    sos = token_to_idx["<SOS>"]
    eos = token_to_idx["<EOS>"]
    pad = token_to_idx["<PAD>"]

    x = torch.tensor([[sos]], dtype=torch.long, device=device)
    hidden = None
    tokens = []

    for _ in range(max_len - 1):
        logits, hidden = model(x, hidden)
        logits = logits[:, -1, :]  # (1, V)
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).view(-1)
        idx = next_token.item()
        if idx == eos:
            break
        if idx == pad:
            break
        tokens.append(idx_to_token[idx])
        x = torch.tensor([[idx]], dtype=torch.long, device=device)

    # join tokens into SELFIES
    if not tokens:
        return None
    try:
        s = "".join(tokens)
        return s
    except Exception:
        return None


def sample_mode(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    model, token_to_idx, idx_to_token, max_len = load_model(args.model_dir, device)
    print(f"[INFO] Loaded model from {args.model_dir}. max_len={max_len}")

    n = args.n_samples
    temperature = args.temperature
    max_trials = args.max_trials

    selfies_samples = []
    smiles_samples = []
    seen = set()

    trials = 0
    with torch.no_grad():
        while len(smiles_samples) < n and trials < max_trials:
            trials += 1
            s = sample_sequence(model, token_to_idx, idx_to_token, max_len,
                                temperature=temperature, device=device)
            if not s:
                continue
            smi = selfies_to_smiles_safe(s)
            if smi is None:
                continue
            if smi in seen:
                continue
            seen.add(smi)
            selfies_samples.append(s)
            smiles_samples.append(smi)

    print(f"[INFO] Generated {len(smiles_samples)} unique valid molecules (trials={trials}).")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df = pd.DataFrame({
        "smiles": smiles_samples,
        "selfies": selfies_samples,
    })
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved generated molecules to {args.out_csv}")


# ========= CLI =========

def main():
    parser = argparse.ArgumentParser(
        description="Classic SELFIES LSTM generator (train/sample)."
    )
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True)

    # shared
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cpu', or 'cuda'")

    # train args
    parser.add_argument("--selfies-files", nargs="+", default=[],
                        help="Paths to *.selfies.txt files for training.")
    parser.add_argument("--outdir", type=str, default="models/classic_lm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hid-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=120,
                        help="Max sequence length (including SOS/EOS).")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)

    # sample args
    parser.add_argument("--model-dir", type=str, default="models/classic_lm")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--out-csv", type=str, default="data/gen_classic_round1.csv")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-trials", type=int, default=10000,
                        help="Max sampling attempts to get n unique molecules.")

    args = parser.parse_args()

    if args.mode == "train":
        if not args.selfies_files:
            raise ValueError("--selfies-files is required in train mode.")
        train_model(args)
    elif args.mode == "sample":
        sample_mode(args)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
