#!/usr/bin/env python3
"""Two-stage fine-tune of AlphaGenome (PyTorch) on LentiMPRA data.

Stage 1: Head-only training (frozen backbone) with AdamW.
Stage 2: Full model fine-tuning (unfrozen backbone) with lower LR.

Usage:
  python training/agft_mpra_fulltwostep.py --name my_model_v1
  python training/agft_mpra_fulltwostep.py --name my_model_v1 --lr 1e-4 --stage2-lr 1e-5
  python training/agft_mpra_fulltwostep.py --name my_model_v1 --skip-stage2 --epochs 2
  python training/agft_mpra_fulltwostep.py --name my_model_v1  # resubmit: reloads saved args
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.training import create_lr_scheduler
from alphagenome_pytorch.extensions.finetuning.transfer import remove_all_heads
from alphagenome_pytorch.extensions.finetuning.utils import sequence_to_onehot

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_BASE = SCRIPT_DIR / "results"
MODEL_CLUB_DIR = SCRIPT_DIR / "pearson's_model_club"
CACHE_DIR = SCRIPT_DIR / "cache"
DATA_DIR = str(REPO_ROOT / "data" / "legnet_lentimpra")

ENCODER_DIM = 1536
ENCODER_RESOLUTION_BP = 128

DEFAULTS = {
    "cell_type": "K562",
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "center_bp": 256,
    "pooling_type": "flatten",
    "nl_size": 1024,
    "dropout": 0.1,
    "activation": "relu",
    "early_stopping": 5,
    "random_shift": True,
    "random_shift_likelihood": 0.5,
    "reverse_complement": True,
    "sequence_length": 256,
    "val_eval_frequency": 1,
    # Stage 2
    "stage2_lr": 1e-5,
    "stage2_epochs": 50,
    "stage2_patience": 5,
}


# ============================================================
# CLI & config
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AlphaGenome (PyTorch) on LentiMPRA")
    parser.add_argument("--name", required=True, help="Model name (used for results directory)")
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    parser.add_argument("--cache-embeddings", action="store_true",
                        help="Use cached encoder embeddings (generates if not found)")
    parser.add_argument("--weights", default=None,
                        help="Path to pretrained weights .pth (default: HuggingFace download)")
    parser.add_argument("--data-dir", default=None,
                        help="Path to LentiMPRA data directory (default: data/legnet_lentimpra)")
    parser.add_argument("--cell-type", default=None, choices=["HepG2", "K562", "WTC11"],
                        help="Cell type to train on (overrides config)")

    # Hyperparameters (CLI flags override config which overrides DEFAULTS)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)

    # Head architecture
    parser.add_argument("--nl-size", type=int, nargs="+", default=None,
                        help="Hidden layer size(s). E.g. --nl-size 1024 or --nl-size 1024 512")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["relu", "gelu"])
    parser.add_argument("--pooling-type", type=str, default=None,
                        choices=["flatten", "sum", "mean", "max", "center"])
    parser.add_argument("--center-bp", type=int, default=None)

    # Augmentation
    parser.add_argument("--no-reverse-complement", action="store_true")
    parser.add_argument("--no-random-shift", action="store_true")

    # Eval frequency
    parser.add_argument("--val-eval-frequency", type=int, default=None,
                        help="Evaluate validation every N epochs (default: 1)")

    # Stage 2
    parser.add_argument("--stage2-lr", type=float, default=None, help="Stage 2 learning rate")
    parser.add_argument("--stage2-epochs", type=int, default=None, help="Stage 2 max epochs")
    parser.add_argument("--stage2-patience", type=int, default=None, help="Stage 2 early stopping patience")
    parser.add_argument("--skip-stage2", action="store_true", help="Run stage 1 only (head-only)")

    return parser.parse_args()


def load_config(config_path, defaults):
    """Load JSON config and merge with defaults."""
    if config_path is None:
        return dict(defaults)
    with open(config_path) as f:
        cfg = json.load(f)
    hp = dict(defaults)
    if "cell_type" in cfg:
        hp["cell_type"] = cfg["cell_type"]
    data = cfg.get("data", {})
    hp["batch_size"] = data.get("batch_size", hp["batch_size"])
    hp["random_shift"] = data.get("random_shift", hp["random_shift"])
    hp["random_shift_likelihood"] = data.get("random_shift_likelihood", hp["random_shift_likelihood"])
    hp["reverse_complement"] = data.get("reverse_complement", hp["reverse_complement"])
    model = cfg.get("model", {})
    hp["center_bp"] = model.get("center_bp", hp["center_bp"])
    hp["pooling_type"] = model.get("pooling_type", hp["pooling_type"])
    nl = model.get("nl_size", hp["nl_size"])
    hp["nl_size"] = int(nl) if isinstance(nl, str) else nl
    hp["dropout"] = model.get("do", hp["dropout"])
    hp["activation"] = model.get("activation", hp["activation"])
    training = cfg.get("training", {})
    hp["num_epochs"] = training.get("num_epochs", hp["num_epochs"])
    hp["learning_rate"] = training.get("learning_rate", hp["learning_rate"])
    hp["weight_decay"] = training.get("weight_decay", hp["weight_decay"])
    hp["early_stopping"] = training.get("early_stopping_patience", hp["early_stopping"])
    hp["val_eval_frequency"] = training.get("val_eval_frequency", hp["val_eval_frequency"])
    hp["sequence_length"] = training.get("sequence_length", hp["sequence_length"])
    two_stage = cfg.get("two_stage", {})
    hp["stage2_lr"] = two_stage.get("second_stage_lr", hp["stage2_lr"])
    hp["stage2_epochs"] = two_stage.get("second_stage_epochs", hp["stage2_epochs"])
    hp["stage2_patience"] = two_stage.get("early_stopping_patience", hp["stage2_patience"])
    return hp


def apply_cli_overrides(hp, args):
    """CLI flags override config/defaults."""
    if args.cell_type is not None:
        hp["cell_type"] = args.cell_type
    if args.lr is not None:
        hp["learning_rate"] = args.lr
    if args.weight_decay is not None:
        hp["weight_decay"] = args.weight_decay
    if args.batch_size is not None:
        hp["batch_size"] = args.batch_size
    if args.epochs is not None:
        hp["num_epochs"] = args.epochs
    if args.patience is not None:
        hp["early_stopping"] = args.patience
    if args.sequence_length is not None:
        hp["sequence_length"] = args.sequence_length
    if args.nl_size is not None:
        hp["nl_size"] = args.nl_size if len(args.nl_size) > 1 else args.nl_size[0]
    if args.dropout is not None:
        hp["dropout"] = args.dropout
    if args.activation is not None:
        hp["activation"] = args.activation
    if args.pooling_type is not None:
        hp["pooling_type"] = args.pooling_type
    if args.center_bp is not None:
        hp["center_bp"] = args.center_bp
    if args.no_reverse_complement:
        hp["reverse_complement"] = False
    if args.no_random_shift:
        hp["random_shift"] = False
    if args.val_eval_frequency is not None:
        hp["val_eval_frequency"] = args.val_eval_frequency
    if args.stage2_lr is not None:
        hp["stage2_lr"] = args.stage2_lr
    if args.stage2_epochs is not None:
        hp["stage2_epochs"] = args.stage2_epochs
    if args.stage2_patience is not None:
        hp["stage2_patience"] = args.stage2_patience


# ============================================================
# Dataset
# ============================================================

class LentiMPRADataset(Dataset):
    """PyTorch Dataset for LentiMPRA data (Agarwal et al., 2025).

    Assembles the full construct (seq + promoter + barcode), one-hot encodes it,
    pads or trims to ``sequence_length``, and optionally applies augmentations.
    """

    PROMOTER_SEQ = "TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG"  # 36 bp
    RAND_BARCODE = "AGAGACTGAGGCCAC"                       # 15 bp

    FOLD_SPLITS: dict[str, list[int]] = {
        "train": [2, 3, 4, 5, 6, 7, 8, 9],
        "val":   [1],
        "test":  [10],
    }

    def __init__(
        self,
        data_dir: str,
        cell_type: str = "HepG2",
        split: str = "train",
        sequence_length: int = 256,
        reverse_complement: bool = False,
        rc_prob: float = 0.5,
        random_shift: bool = False,
        shift_prob: float = 0.5,
        max_shift: int = 15,
        subset_frac: float = 1.0,
        seed: int = 42,
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"

        self.sequence_length = sequence_length
        self.reverse_complement = reverse_complement
        self.rc_prob = rc_prob
        self.random_shift = random_shift
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self._rng = np.random.default_rng(seed)

        df = pd.read_csv(os.path.join(data_dir, f"{cell_type}.tsv"), sep="\t")
        df = df[df["rev"] == 0]
        df = df[df["fold"].isin(self.FOLD_SPLITS[split])].reset_index(drop=True)

        if subset_frac < 1.0:
            df = df.sample(frac=subset_frac, random_state=seed).reset_index(drop=True)

        self.sequences: list[str] = df["seq"].tolist()
        self.targets: np.ndarray = df["mean_value"].values.astype(np.float32)
        print(f"Loaded {len(self.sequences):,} {split} samples ({cell_type})")

    def _build_construct(self, seq: str) -> str:
        return seq + self.PROMOTER_SEQ + self.RAND_BARCODE

    def _pad_or_trim(self, onehot: np.ndarray) -> np.ndarray:
        L = onehot.shape[0]
        if L < self.sequence_length:
            pad = np.zeros((self.sequence_length - L, 4), dtype=np.float32)
            return np.concatenate([onehot, pad], axis=0)
        return onehot[: self.sequence_length]

    def _apply_reverse_complement(self, onehot: np.ndarray) -> np.ndarray:
        return onehot[::-1, :][:, [3, 2, 1, 0]].copy()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        construct = self._build_construct(self.sequences[idx])
        target = self.targets[idx]

        onehot = sequence_to_onehot(construct).astype(np.float32)

        if self.random_shift and self._rng.random() < self.shift_prob:
            shift = int(self._rng.integers(-self.max_shift, self.max_shift + 1))
            onehot = np.roll(onehot, shift, axis=0)

        onehot = self._pad_or_trim(onehot)

        if self.reverse_complement and self._rng.random() < self.rc_prob:
            onehot = self._apply_reverse_complement(onehot)

        return torch.from_numpy(onehot), torch.tensor(target)


# ============================================================
# MPRAHead (PyTorch port of EncoderMPRAHead with all pooling types)
# ============================================================

class MPRAHead(nn.Module):
    """MLP head for MPRA activity score regression from encoder-only features.

    Accepts raw CNN encoder output ``(B, n_positions, 1536)`` and predicts
    a scalar activity score per sequence.

    Supports pooling types: flatten, sum, mean, max, center.
    Supports multi-layer MLP via nl_size list.
    """

    def __init__(
        self,
        n_positions: int,
        nl_size: int | list[int] = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        pooling_type: str = "flatten",
        center_bp: int = 256,
    ) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        self.activation_name = activation
        self.center_bp = center_bp
        self.n_positions = n_positions
        self._center_window_positions = max(1, center_bp // ENCODER_RESOLUTION_BP)

        if isinstance(nl_size, int):
            hidden_sizes = [nl_size]
        else:
            hidden_sizes = list(nl_size)

        self.norm = nn.LayerNorm(ENCODER_DIM)

        # Compute input dim for first linear layer
        if pooling_type == "flatten":
            in_dim = n_positions * ENCODER_DIM
        else:
            in_dim = ENCODER_DIM

        # Build MLP layers
        layers: list[nn.Module] = []
        for hs in hidden_sizes:
            layers.append(nn.Linear(in_dim, hs))
            in_dim = hs
        self.hidden_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output = nn.Linear(in_dim, 1)

        # Choose activation
        if activation == "gelu":
            self.act_fn = F.gelu
        else:
            self.act_fn = F.relu

    def _pool(self, x: Tensor) -> Tensor:
        """Pool encoder output (B, n_pos, D) to (B, features)."""
        if self.pooling_type == "flatten":
            return x.flatten(1)  # (B, n_pos * D)
        elif self.pooling_type == "center":
            center_idx = x.shape[1] // 2
            return x[:, center_idx, :]  # (B, D)
        else:
            # Extract center window
            seq_len = x.shape[1]
            window_size = min(self._center_window_positions, seq_len)
            center_start = max(0, (seq_len - window_size) // 2)
            center = x[:, center_start : center_start + window_size, :]
            if self.pooling_type == "mean":
                return center.mean(dim=1)
            elif self.pooling_type == "max":
                return center.max(dim=1).values
            elif self.pooling_type == "sum":
                return center.sum(dim=1)
            else:
                raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

    def forward(self, encoder_output: Tensor) -> Tensor:
        x = self.norm(encoder_output)
        x = self._pool(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.act_fn(x)
            x = self.dropout(x)

        x = self.output(x)
        return x.squeeze(-1)


# ============================================================
# Cache generation
# ============================================================

def generate_cache(model: nn.Module, hp: dict, data_dir: str, device: torch.device) -> str:
    """Generate encoder embedding cache. Returns cache file path."""
    cache_file = CACHE_DIR / f"{hp['cell_type']}_embeddings.pkl"
    if cache_file.exists():
        print(f"Cache already exists: {cache_file}")
        return str(cache_file)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data_path = os.path.join(data_dir, f"{hp['cell_type']}.tsv")
    df = pd.read_csv(data_path, sep="\t")
    df = df[df["rev"] == 0].reset_index(drop=True)

    promoter_seq = LentiMPRADataset.PROMOTER_SEQ
    rand_barcode = LentiMPRADataset.RAND_BARCODE
    seq_len = hp["sequence_length"]

    cache = {}
    batch_size = hp["batch_size"]
    n = len(df)
    print(f"Generating cache for {n} sequences (one-time cost)...")

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_seqs = []
            batch_hashes = []

            for idx in range(start, end):
                seq_str = df.iloc[idx]["seq"] + promoter_seq + rand_barcode
                batch_hashes.append(hashlib.sha256(seq_str.encode()).hexdigest())
                onehot = sequence_to_onehot(seq_str).astype(np.float32)
                # Pad/trim
                L = onehot.shape[0]
                if L < seq_len:
                    onehot = np.concatenate(
                        [onehot, np.zeros((seq_len - L, 4), dtype=np.float32)], axis=0
                    )
                else:
                    onehot = onehot[:seq_len]
                batch_seqs.append(onehot)

            batch_tensor = torch.from_numpy(np.stack(batch_seqs)).to(device)
            org_idx = torch.zeros(batch_tensor.shape[0], dtype=torch.long, device=device)

            with amp_ctx:
                enc_out = model(
                    batch_tensor, org_idx, encoder_only=True
                )["encoder_output"].transpose(1, 2)

            enc_out_np = enc_out.float().cpu().numpy()
            for i, h in enumerate(batch_hashes):
                cache[h] = enc_out_np[i]

            done = min(end, n)
            if (start // batch_size) % 200 == 0:
                print(f"  {done}/{n} sequences cached...")

    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)
    print(f"Cache saved: {cache_file} ({len(cache)} sequences)")
    return str(cache_file)


class CachedEmbeddingDataset(Dataset):
    """Dataset that loads pre-computed encoder embeddings from cache."""

    PROMOTER_SEQ = LentiMPRADataset.PROMOTER_SEQ
    RAND_BARCODE = LentiMPRADataset.RAND_BARCODE

    FOLD_SPLITS = LentiMPRADataset.FOLD_SPLITS

    def __init__(
        self,
        data_dir: str,
        cache_file: str,
        cell_type: str = "HepG2",
        split: str = "train",
        subset_frac: float = 1.0,
        seed: int = 42,
    ) -> None:
        assert split in ("train", "val", "test")

        df = pd.read_csv(os.path.join(data_dir, f"{cell_type}.tsv"), sep="\t")
        df = df[df["rev"] == 0]
        df = df[df["fold"].isin(self.FOLD_SPLITS[split])].reset_index(drop=True)

        if subset_frac < 1.0:
            df = df.sample(frac=subset_frac, random_state=seed).reset_index(drop=True)

        with open(cache_file, "rb") as f:
            self._cache = pickle.load(f)

        self._hashes: list[str] = []
        self.targets: np.ndarray = df["mean_value"].values.astype(np.float32)
        for idx in range(len(df)):
            seq_str = df.iloc[idx]["seq"] + self.PROMOTER_SEQ + self.RAND_BARCODE
            self._hashes.append(hashlib.sha256(seq_str.encode()).hexdigest())

        print(f"Loaded {len(self._hashes):,} cached {split} samples ({cell_type})")

    def __len__(self) -> int:
        return len(self._hashes)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        emb = self._cache[self._hashes[idx]]
        return torch.from_numpy(emb.copy()), torch.tensor(self.targets[idx])


# ============================================================
# Training & evaluation loops
# ============================================================

def train_epoch_headonly(
    model: nn.Module,
    head: MPRAHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    use_amp: bool = True,
    use_cache: bool = False,
) -> tuple[float, list[float]]:
    """Train for one epoch with frozen backbone (or cached embeddings).

    Returns (avg_loss, list_of_batch_losses).
    """
    if not use_cache:
        model.eval()
    head.train()

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    batch_losses: list[float] = []
    pbar = tqdm(loader, desc=" train", leave=False)
    for sequences, targets in pbar:
        sequences = sequences.to(device)
        targets = targets.to(device).float()

        if use_cache:
            # sequences is already encoder output
            enc_out = sequences
        else:
            organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
            with torch.no_grad():
                with amp_ctx:
                    enc_out = model(
                        sequences, organism_idx, encoder_only=True
                    )["encoder_output"].transpose(1, 2)
            enc_out = enc_out.detach()

        with amp_ctx:
            preds = head(enc_out)
            loss = F.mse_loss(preds.float(), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_losses.append(loss.item())
        pbar.set_postfix({"mse": f"{loss.item():.4f}"})

    return float(np.mean(batch_losses)), batch_losses


def train_epoch_full(
    model: nn.Module,
    head: MPRAHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[float, list[float]]:
    """Train for one epoch with unfrozen backbone (stage 2)."""
    model.train()
    head.train()

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    batch_losses: list[float] = []
    pbar = tqdm(loader, desc=" train", leave=False)
    for sequences, targets in pbar:
        sequences = sequences.to(device)
        targets = targets.to(device).float()
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        with amp_ctx:
            enc_out = model(
                sequences, organism_idx, encoder_only=True
            )["encoder_output"].transpose(1, 2)
            preds = head(enc_out)
            loss = F.mse_loss(preds.float(), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_losses.append(loss.item())
        pbar.set_postfix({"mse": f"{loss.item():.4f}"})

    return float(np.mean(batch_losses)), batch_losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    head: MPRAHead,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    use_cache: bool = False,
) -> float:
    """Evaluate and return average MSE loss."""
    model.eval()
    head.eval()

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    losses: list[float] = []
    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device).float()

        if use_cache:
            enc_out = sequences
        else:
            organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
            with amp_ctx:
                enc_out = model(
                    sequences, organism_idx, encoder_only=True
                )["encoder_output"].transpose(1, 2)

        with amp_ctx:
            preds = head(enc_out)
            loss = F.mse_loss(preds.float(), targets)

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    head: MPRAHead,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    use_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect all predictions and targets."""
    model.eval()
    head.eval()

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    all_preds, all_targets = [], []
    for sequences, targets in loader:
        sequences = sequences.to(device)

        if use_cache:
            enc_out = sequences
        else:
            organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
            with amp_ctx:
                enc_out = model(
                    sequences, organism_idx, encoder_only=True
                )["encoder_output"].transpose(1, 2)

        with amp_ctx:
            preds = head(enc_out)

        all_preds.append(preds.float().cpu().numpy())
        all_targets.append(targets.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


# ============================================================
# Metrics & figures
# ============================================================

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    r, _ = pearsonr(preds, targets)
    rho, _ = spearmanr(preds, targets)
    mse = float(np.mean((preds - targets) ** 2))
    return {"pearson_r": float(r), "spearman_rho": float(rho), "mse": mse}


def make_summary_figure(
    epoch1_preds, epoch1_targets,
    best_preds, best_targets, metrics_best,
    train_loss_hist, valid_loss_hist,
    best_epoch, save_path, run_name,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    all_vals = [best_targets, best_preds]
    if epoch1_preds is not None:
        all_vals.append(epoch1_preds)
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)
    lims = [vmin, vmax]

    if epoch1_preds is not None:
        m1 = compute_metrics(epoch1_preds, epoch1_targets)
        axes[0].scatter(epoch1_targets, epoch1_preds, alpha=0.1, s=1, rasterized=True)
        axes[0].plot(lims, lims, "r--", linewidth=0.5)
        axes[0].set_title(f"Epoch 1: r={m1['pearson_r']:.3f}, rho={m1['spearman_rho']:.3f}")
    else:
        axes[0].set_title("Epoch 1: N/A")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)

    axes[1].scatter(best_targets, best_preds, alpha=0.1, s=1, rasterized=True)
    axes[1].plot(lims, lims, "r--", linewidth=0.5)
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"Best (ep {best_epoch}): r={metrics_best['pearson_r']:.3f}, rho={metrics_best['spearman_rho']:.3f}")
    axes[1].set_xlim(lims); axes[1].set_ylim(lims)

    epochs_range = range(1, len(train_loss_hist) + 1)
    axes[2].plot(epochs_range, train_loss_hist, label="Train")
    axes[2].plot(epochs_range, valid_loss_hist, label="Valid")
    axes[2].axvline(best_epoch, color="gray", linestyle=":", alpha=0.7, label=f"Best (epoch {best_epoch})")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss (MSE)")
    axes[2].set_title("Training Loss"); axes[2].legend()

    plt.suptitle(f"AlphaGenome-PyTorch FT -> LentiMPRA [{run_name}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary figure saved to {save_path}")


def make_combined_summary(
    epoch1_preds, epoch1_targets,
    final_preds, final_targets, final_metrics,
    s1_train_loss, s1_valid_loss, s1_best_epoch,
    s2_train_loss, s2_valid_loss, s2_best_epoch,
    save_path, run_name,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    all_vals = [final_targets, final_preds]
    if epoch1_preds is not None:
        all_vals.append(epoch1_preds)
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)
    lims = [vmin, vmax]

    if epoch1_preds is not None:
        m1 = compute_metrics(epoch1_preds, epoch1_targets)
        axes[0].scatter(epoch1_targets, epoch1_preds, alpha=0.1, s=1, rasterized=True)
        axes[0].plot(lims, lims, "r--", linewidth=0.5)
        axes[0].set_title(f"Epoch 1: r={m1['pearson_r']:.3f}, rho={m1['spearman_rho']:.3f}")
    else:
        axes[0].set_title("Epoch 1: N/A")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)

    axes[1].scatter(final_targets, final_preds, alpha=0.1, s=1, rasterized=True)
    axes[1].plot(lims, lims, "r--", linewidth=0.5)
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    best_label = f"S2 ep {s2_best_epoch}" if s2_train_loss else f"S1 ep {s1_best_epoch}"
    axes[1].set_title(f"Best ({best_label}): r={final_metrics['pearson_r']:.3f}, rho={final_metrics['spearman_rho']:.3f}")
    axes[1].set_xlim(lims); axes[1].set_ylim(lims)

    n_s1 = len(s1_train_loss)
    s1_epochs = list(range(1, n_s1 + 1))
    axes[2].plot(s1_epochs, s1_train_loss, color="tab:blue", label="S1 Train")
    axes[2].plot(s1_epochs, s1_valid_loss, color="tab:orange", label="S1 Valid")

    if s2_train_loss:
        s2_epochs = list(range(n_s1 + 1, n_s1 + len(s2_train_loss) + 1))
        axes[2].plot(s2_epochs, s2_train_loss, color="tab:blue", linestyle="--", label="S2 Train")
        axes[2].plot(s2_epochs, s2_valid_loss, color="tab:orange", linestyle="--", label="S2 Valid")
        unfreeze_x = n_s1 + 0.5
        axes[2].axvline(unfreeze_x, color="red", linestyle="-", alpha=0.7, label="Unfreeze")
        axes[2].axvline(n_s1 + s2_best_epoch, color="green", linestyle=":", alpha=0.7, label=f"S2 best (ep {s2_best_epoch})")

    axes[2].axvline(s1_best_epoch, color="gray", linestyle=":", alpha=0.7, label=f"S1 best (ep {s1_best_epoch})")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss (MSE)")
    axes[2].set_title("Training Loss"); axes[2].legend(fontsize=8)

    plt.suptitle(f"AlphaGenome-PyTorch FT -> LentiMPRA [{run_name}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined summary saved to {save_path}")


def update_model_club(name, metrics_best, preds_best, targets_best, hp):
    MODEL_CLUB_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = MODEL_CLUB_DIR / "best_models.csv"
    fields = ["name", "pearson_r", "spearman_rho", "mse", "cell_type", "timestamp"]
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    current_best = max((float(r["pearson_r"]) for r in rows), default=-1.0)
    new_r = metrics_best["pearson_r"]
    rows.append({
        "name": name, "pearson_r": f"{new_r:.6f}",
        "spearman_rho": f"{metrics_best['spearman_rho']:.6f}",
        "mse": f"{metrics_best['mse']:.6f}",
        "cell_type": hp["cell_type"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if new_r > current_best:
        print(f"New best model! Pearson r = {new_r:.4f} (previous best: {current_best:.4f})")
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(targets_best, preds_best, alpha=0.1, s=1, rasterized=True)
        lims = [min(targets_best.min(), preds_best.min()), max(targets_best.max(), preds_best.max())]
        ax.plot(lims, lims, "r--", linewidth=0.5)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"Best model: {name} (r={new_r:.4f})")
        ax.text(0.05, 0.95, f"r = {new_r:.4f}\nrho = {metrics_best['spearman_rho']:.4f}\nMSE = {metrics_best['mse']:.4f}",
                transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.tight_layout(); plt.savefig(MODEL_CLUB_DIR / "best_model_summary.png", dpi=150); plt.close()
    else:
        print(f"Model r = {new_r:.4f} did not beat current best ({current_best:.4f})")


# ============================================================
# Training state persistence (resume support)
# ============================================================

def save_training_state(
    path: Path,
    stage: int,
    epoch: int,
    best_valid_loss: float,
    train_loss_history: list,
    valid_loss_history: list,
    s1_completed: bool,
    s1_best_epoch: int,
    s2_train_loss_history: list | None = None,
    s2_valid_loss_history: list | None = None,
    s2_best_epoch: int = 0,
) -> None:
    state = {
        "stage": stage,
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
        "train_loss_history": [float(v) for v in train_loss_history],
        "valid_loss_history": [float(v) for v in valid_loss_history],
        "s1_completed": s1_completed,
        "s1_best_epoch": s1_best_epoch,
        "s2_train_loss_history": [float(v) for v in (s2_train_loss_history or [])],
        "s2_valid_loss_history": [float(v) for v in (s2_valid_loss_history or [])],
        "s2_best_epoch": s2_best_epoch,
    }
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_training_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================
# Model loading
# ============================================================

def load_pretrained_model(weights_path: str | None, device: torch.device) -> nn.Module:
    """Load AlphaGenome model with pretrained weights."""
    if weights_path is None:
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download("gtca/alphagenome_pytorch", "model_fold_0.safetensors")
        print(f"Downloaded weights to {weights_path}")

    print(f"Loading pretrained weights from {weights_path} ...")
    model = AlphaGenome.from_pretrained(weights_path, device=device)
    model = remove_all_heads(model)
    return model


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    results_dir = RESULTS_BASE / args.name
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Persist args so resubmits with just --name recover original flags
    args_file = results_dir / "args.json"
    has_cli_overrides = any([
        args.config, args.cache_embeddings, args.lr, args.weight_decay,
        args.batch_size, args.epochs, args.patience, args.nl_size,
        args.dropout, args.activation, args.pooling_type, args.center_bp,
        args.no_reverse_complement, args.no_random_shift,
        args.stage2_lr, args.stage2_epochs, args.stage2_patience, args.skip_stage2,
        args.weights, args.data_dir, args.sequence_length, args.cell_type,
    ])
    if has_cli_overrides or not args_file.exists():
        hp = load_config(args.config, DEFAULTS)
        apply_cli_overrides(hp, args)
        use_cache = args.cache_embeddings
        skip_stage2 = args.skip_stage2
        weights_path = args.weights
        data_dir = args.data_dir or DATA_DIR
        saved = {
            "hp": hp, "config": args.config,
            "cache_embeddings": use_cache, "skip_stage2": skip_stage2,
            "weights": weights_path, "data_dir": data_dir,
        }
        with open(args_file, "w") as f:
            json.dump(saved, f, indent=2)
    else:
        with open(args_file) as f:
            saved = json.load(f)
        hp = saved["hp"]
        use_cache = saved.get("cache_embeddings", False)
        skip_stage2 = saved.get("skip_stage2", False)
        weights_path = saved.get("weights")
        data_dir = saved.get("data_dir", DATA_DIR)
        print(f"Loaded saved args from {args_file}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"Device: {device}")
    print(f"Model name: {args.name}")
    print(f"Cell type: {hp['cell_type']}")
    print(f"Mode: {'cached embeddings' if use_cache else 'full model'}")
    print(f"Results dir: {results_dir}")
    print(f"Hyperparameters: {json.dumps(hp, indent=2)}")

    # ---- Model ----
    model = load_pretrained_model(weights_path, device)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    n_backbone = sum(p.numel() for p in model.parameters())
    print(f"Backbone loaded and frozen ({n_backbone:,} parameters)")

    # ---- Generate cache if needed ----
    cache_file = None
    if use_cache:
        cache_file = generate_cache(model, hp, data_dir, device)

    # ---- Head ----
    seq_len = hp["sequence_length"]
    n_positions = seq_len // ENCODER_RESOLUTION_BP
    nl_size = hp["nl_size"] if isinstance(hp["nl_size"], list) else hp["nl_size"]

    head = MPRAHead(
        n_positions=n_positions,
        nl_size=nl_size,
        dropout=hp["dropout"],
        activation=hp["activation"],
        pooling_type=hp["pooling_type"],
        center_bp=hp["center_bp"],
    ).to(device)

    n_head = sum(p.numel() for p in head.parameters())
    print(f"MPRAHead created: {n_head:,} trainable parameters")
    print(f"  pooling={hp['pooling_type']}, nl_size={nl_size}, "
          f"dropout={hp['dropout']}, activation={hp['activation']}")

    # ---- Data ----
    if use_cache:
        train_ds = CachedEmbeddingDataset(
            data_dir=data_dir, cache_file=cache_file, cell_type=hp["cell_type"], split="train",
        )
        val_ds = CachedEmbeddingDataset(
            data_dir=data_dir, cache_file=cache_file, cell_type=hp["cell_type"], split="val",
        )
        test_ds = CachedEmbeddingDataset(
            data_dir=data_dir, cache_file=cache_file, cell_type=hp["cell_type"], split="test",
        )
    else:
        ds_kwargs = dict(
            data_dir=data_dir, cell_type=hp["cell_type"], sequence_length=seq_len,
        )
        train_ds = LentiMPRADataset(
            **ds_kwargs, split="train",
            reverse_complement=hp["reverse_complement"], rc_prob=0.5,
            random_shift=hp["random_shift"], shift_prob=hp.get("random_shift_likelihood", 0.5),
            max_shift=15,
        )
        val_ds = LentiMPRADataset(**ds_kwargs, split="val")
        test_ds = LentiMPRADataset(**ds_kwargs, split="test")

    train_loader = DataLoader(
        train_ds, batch_size=hp["batch_size"], shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=hp["batch_size"], shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=hp["batch_size"], shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"Train: {len(train_ds):,} samples, {len(train_loader):,} batches")
    print(f"Val:   {len(val_ds):,} samples, {len(val_loader):,} batches")
    print(f"Test:  {len(test_ds):,} samples, {len(test_loader):,} batches")

    # ---- Check for resume ----
    state_file = results_dir / "training_state.json"
    resume_state = load_training_state(state_file)

    # ============================================================
    # Stage 1: Head-only training
    # ============================================================
    start_epoch = 1
    best_valid_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    train_loss_history: list[float] = []
    valid_loss_history: list[float] = []
    epoch1_preds = None
    epoch1_targets = None
    best_preds = None
    best_targets = None
    s1_completed = False

    if resume_state is not None and resume_state.get("s1_completed", False):
        # Stage 1 already done — reload state
        s1_completed = True
        train_loss_history = resume_state["train_loss_history"]
        valid_loss_history = resume_state["valid_loss_history"]
        best_epoch = resume_state["s1_best_epoch"]
        best_valid_loss = resume_state["best_valid_loss"]
        print(f"Stage 1 already completed (best epoch {best_epoch}). Skipping to stage 2.")

        # Load best head checkpoint
        best_ckpt_path = checkpoint_dir / "best_head.pt"
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
            head.load_state_dict(ckpt["head_state_dict"])
            print(f"Loaded best head from {best_ckpt_path}")

        # Load saved predictions
        epoch1_npz = results_dir / "epoch1_predictions.npz"
        if epoch1_npz.exists():
            d = np.load(epoch1_npz)
            epoch1_preds, epoch1_targets = d["preds"], d["targets"]
        best_npz = results_dir / "best_predictions.npz"
        if best_npz.exists():
            d = np.load(best_npz)
            best_preds, best_targets = d["preds"], d["targets"]

    elif resume_state is not None and resume_state["stage"] == 1:
        # Resume stage 1 mid-training
        start_epoch = resume_state["epoch"] + 1
        train_loss_history = resume_state["train_loss_history"]
        valid_loss_history = resume_state["valid_loss_history"]
        best_valid_loss = resume_state["best_valid_loss"]
        best_epoch = resume_state["s1_best_epoch"]

        best_ckpt_path = checkpoint_dir / "best_head.pt"
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
            head.load_state_dict(ckpt["head_state_dict"])

        # Load latest checkpoint for optimizer state
        latest_ckpt_path = checkpoint_dir / "latest_head.pt"
        if latest_ckpt_path.exists():
            ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=True)
            head.load_state_dict(ckpt["head_state_dict"])

        epoch1_npz = results_dir / "epoch1_predictions.npz"
        if epoch1_npz.exists():
            d = np.load(epoch1_npz)
            epoch1_preds, epoch1_targets = d["preds"], d["targets"]

        best_npz = results_dir / "best_predictions.npz"
        if best_npz.exists():
            d = np.load(best_npz)
            best_preds, best_targets = d["preds"], d["targets"]

        print(f"Resuming stage 1 from epoch {start_epoch}")

    if not s1_completed:
        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
        )
        steps_per_epoch = len(train_loader)
        total_steps = hp["num_epochs"] * steps_per_epoch
        warmup_steps = steps_per_epoch  # 1 epoch warmup
        scheduler = create_lr_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            schedule="cosine",
        )
        # Fast-forward scheduler if resuming
        if start_epoch > 1:
            for _ in range((start_epoch - 1) * steps_per_epoch):
                scheduler.step()

        patience = hp["early_stopping"]
        eval_freq = hp["val_eval_frequency"]
        epochs_no_improve = 0

        print(f"\n{'='*60}")
        print(f"Stage 1: Head-only training ({'cached embeddings' if use_cache else 'frozen encoder'})")
        print(f"  LR={hp['learning_rate']}, WD={hp['weight_decay']}, BS={hp['batch_size']}")
        print(f"  Epochs={hp['num_epochs']}, Patience={patience}, EvalFreq={eval_freq}")
        print(f"  Scheduler: cosine with {warmup_steps} warmup steps, {total_steps} total steps")
        print(f"{'='*60}")

        for epoch in range(start_epoch, hp["num_epochs"] + 1):
            train_loss, _ = train_epoch_headonly(
                model=model, head=head, loader=train_loader,
                optimizer=optimizer, scheduler=scheduler,
                device=device, use_amp=use_amp, use_cache=use_cache,
            )
            train_loss_history.append(train_loss)

            # Snapshot after epoch 1
            if epoch == 1:
                epoch1_preds, epoch1_targets = collect_predictions(
                    model, head, test_loader, device, use_amp=use_amp, use_cache=use_cache,
                )
                np.savez(results_dir / "epoch1_predictions.npz",
                         preds=epoch1_preds, targets=epoch1_targets)

            # Evaluate validation every eval_freq epochs
            if epoch % eval_freq == 0 or epoch == hp["num_epochs"]:
                valid_loss = evaluate(
                    model=model, head=head, loader=val_loader,
                    device=device, use_amp=use_amp, use_cache=use_cache,
                )
                valid_loss_history.append(valid_loss)

                is_best = valid_loss < best_valid_loss
                if is_best:
                    best_valid_loss = valid_loss
                    epochs_no_improve = 0
                    best_epoch = epoch
                    best_preds, best_targets = collect_predictions(
                        model, head, test_loader, device, use_amp=use_amp, use_cache=use_cache,
                    )
                    np.savez(results_dir / "best_predictions.npz",
                             preds=best_preds, targets=best_targets)
                    torch.save(
                        {"head_state_dict": head.state_dict(), "epoch": epoch,
                         "val_loss": valid_loss},
                        checkpoint_dir / "best_head.pt",
                    )
                    star = " * (saved)"
                else:
                    epochs_no_improve += 1
                    star = f"  (no improve {epochs_no_improve}/{patience})"

                print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                      f"| valid_loss={valid_loss:.4f}{star}")
            else:
                print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

            # Save latest for resume
            torch.save(
                {"head_state_dict": head.state_dict(), "epoch": epoch,
                 "optimizer_state_dict": optimizer.state_dict()},
                checkpoint_dir / "latest_head.pt",
            )
            save_training_state(
                state_file, stage=1, epoch=epoch,
                best_valid_loss=best_valid_loss,
                train_loss_history=train_loss_history,
                valid_loss_history=valid_loss_history,
                s1_completed=False, s1_best_epoch=best_epoch,
            )

            if epoch % eval_freq == 0 and patience > 0 and epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Mark stage 1 complete
        s1_completed = True
        save_training_state(
            state_file, stage=1, epoch=len(train_loss_history),
            best_valid_loss=best_valid_loss,
            train_loss_history=train_loss_history,
            valid_loss_history=valid_loss_history,
            s1_completed=True, s1_best_epoch=best_epoch,
        )

    # Ensure we have predictions even without improvement
    if best_preds is None:
        best_preds, best_targets = collect_predictions(
            model, head, test_loader, device, use_amp=use_amp, use_cache=use_cache,
        )
        best_epoch = len(train_loss_history)

    # ---- Stage 1 results ----
    s1_metrics = compute_metrics(best_preds, best_targets)
    print(f"\nStage 1 Test (best epoch {best_epoch}): "
          f"r={s1_metrics['pearson_r']:.4f}, "
          f"rho={s1_metrics['spearman_rho']:.4f}, "
          f"MSE={s1_metrics['mse']:.4f}")

    if epoch1_preds is not None:
        m1 = compute_metrics(epoch1_preds, epoch1_targets)
        print(f"Stage 1 Test (epoch 1): r={m1['pearson_r']:.4f}, "
              f"rho={m1['spearman_rho']:.4f}, MSE={m1['mse']:.4f}")

    make_summary_figure(
        epoch1_preds, epoch1_targets,
        best_preds, best_targets, s1_metrics,
        train_loss_history, valid_loss_history,
        best_epoch, results_dir / "summary_stage1.png", f"{args.name} (stage 1)",
    )

    # ============================================================
    # Stage 2: Full model fine-tuning (unfrozen backbone)
    # ============================================================
    s2_metrics = None
    s2_best_preds = None
    s2_best_targets = None
    s2_train_loss_history: list[float] = []
    s2_valid_loss_history: list[float] = []
    s2_best_epoch = 0

    if not skip_stage2 and not use_cache:
        # Check resume for stage 2
        s2_start_epoch = 1
        s2_best_valid_loss = float("inf")
        s2_epochs_no_improve = 0

        if resume_state is not None and resume_state.get("stage") == 2:
            # Possibly not used yet, but support it
            pass

        print(f"\n{'='*60}")
        print(f"Stage 2: Full model fine-tuning (unfrozen backbone)")
        print(f"  LR={hp['stage2_lr']}, WD={hp['weight_decay']}, BS={hp['batch_size']}")
        print(f"  Epochs={hp['stage2_epochs']}, Patience={hp['stage2_patience']}")
        print(f"{'='*60}")

        # Reload best stage 1 head
        best_ckpt_path = checkpoint_dir / "best_head.pt"
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
            head.load_state_dict(ckpt["head_state_dict"])
            print(f"Restored best stage 1 head (epoch {best_epoch})")

        # Unfreeze backbone
        for param in model.parameters():
            param.requires_grad = True
        print("Unfroze all backbone parameters")

        # New data loaders without augmentations disabled? No — keep same loaders.
        # Stage 2 uses same data.

        # New optimizer for all params
        all_params = list(model.parameters()) + list(head.parameters())
        s2_optimizer = torch.optim.AdamW(
            all_params,
            lr=hp["stage2_lr"],
            weight_decay=hp["weight_decay"],
        )
        s2_steps_per_epoch = len(train_loader)
        s2_total_steps = hp["stage2_epochs"] * s2_steps_per_epoch
        s2_warmup_steps = s2_steps_per_epoch  # 1 epoch warmup
        s2_scheduler = create_lr_scheduler(
            s2_optimizer,
            warmup_steps=s2_warmup_steps,
            total_steps=s2_total_steps,
            schedule="cosine",
        )

        s2_patience = hp["stage2_patience"]

        for epoch in range(s2_start_epoch, hp["stage2_epochs"] + 1):
            train_loss, _ = train_epoch_full(
                model=model, head=head, loader=train_loader,
                optimizer=s2_optimizer, scheduler=s2_scheduler,
                device=device, use_amp=use_amp,
            )

            s2_train_loss_history.append(train_loss)

            # Evaluate validation every eval_freq epochs
            if epoch % eval_freq == 0 or epoch == hp["stage2_epochs"]:
                valid_loss = evaluate(
                    model=model, head=head, loader=val_loader,
                    device=device, use_amp=use_amp, use_cache=False,
                )
                s2_valid_loss_history.append(valid_loss)

                is_best = valid_loss < s2_best_valid_loss
                if is_best:
                    s2_best_valid_loss = valid_loss
                    s2_epochs_no_improve = 0
                    s2_best_epoch = epoch
                    s2_best_preds, s2_best_targets = collect_predictions(
                        model, head, test_loader, device, use_amp=use_amp,
                    )
                    torch.save(
                        {"model_state_dict": model.state_dict(),
                         "head_state_dict": head.state_dict(),
                         "epoch": epoch, "val_loss": valid_loss},
                        checkpoint_dir / "best_stage2.pt",
                    )
                    star = " * (saved)"
                else:
                    s2_epochs_no_improve += 1
                    star = f"  (no improve {s2_epochs_no_improve}/{s2_patience})"

                print(f"S2 Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                      f"| valid_loss={valid_loss:.4f}{star}")
            else:
                print(f"S2 Epoch {epoch:03d} | train_loss={train_loss:.4f}")

            # Save state for resume
            save_training_state(
                state_file, stage=2, epoch=epoch,
                best_valid_loss=s2_best_valid_loss,
                train_loss_history=train_loss_history,
                valid_loss_history=valid_loss_history,
                s1_completed=True, s1_best_epoch=best_epoch,
                s2_train_loss_history=s2_train_loss_history,
                s2_valid_loss_history=s2_valid_loss_history,
                s2_best_epoch=s2_best_epoch,
            )

            if epoch % eval_freq == 0 and s2_patience > 0 and s2_epochs_no_improve >= s2_patience:
                print(f"\nStage 2 early stopping at epoch {epoch}")
                break

        if s2_best_preds is not None:
            s2_metrics = compute_metrics(s2_best_preds, s2_best_targets)
            print(f"\nStage 2 Test (best epoch {s2_best_epoch}): "
                  f"r={s2_metrics['pearson_r']:.4f}, "
                  f"rho={s2_metrics['spearman_rho']:.4f}, "
                  f"MSE={s2_metrics['mse']:.4f}")

            make_summary_figure(
                best_preds, best_targets,
                s2_best_preds, s2_best_targets, s2_metrics,
                s2_train_loss_history, s2_valid_loss_history,
                s2_best_epoch, results_dir / "summary_stage2.png", f"{args.name} (stage 2)",
            )

    elif not skip_stage2 and use_cache:
        print("\nNote: Stage 2 (full model fine-tuning) is not supported with "
              "cached embeddings. Skipping stage 2.")

    # ============================================================
    # Final metrics
    # ============================================================
    final_metrics = s2_metrics if s2_metrics is not None else s1_metrics
    final_preds = s2_best_preds if s2_best_preds is not None else best_preds
    final_targets = s2_best_targets if s2_best_targets is not None else best_targets

    metrics_out = {
        "name": args.name,
        "config_path": args.config,
        "cached_embeddings": use_cache,
        "hyperparameters": hp,
        "stage1_test": s1_metrics,
        "stage1_best_epoch": best_epoch,
        "stage1_epochs_trained": len(train_loss_history),
        "stage2_test": s2_metrics,
        "stage2_best_epoch": s2_best_epoch,
        "stage2_epochs_trained": len(s2_train_loss_history),
        "best_epoch_test": final_metrics,
        "history": {
            "stage1_train_loss": [float(v) for v in train_loss_history],
            "stage1_valid_loss": [float(v) for v in valid_loss_history],
            "stage2_train_loss": [float(v) for v in s2_train_loss_history],
            "stage2_valid_loss": [float(v) for v in s2_valid_loss_history],
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved to {results_dir / 'metrics.json'}")

    make_combined_summary(
        epoch1_preds, epoch1_targets,
        final_preds, final_targets, final_metrics,
        train_loss_history, valid_loss_history, best_epoch,
        s2_train_loss_history, s2_valid_loss_history, s2_best_epoch,
        results_dir / "summary_combined.png", args.name,
    )

    update_model_club(args.name, final_metrics, final_preds, final_targets, hp)
    print("\nDone!")


if __name__ == "__main__":
    main()
