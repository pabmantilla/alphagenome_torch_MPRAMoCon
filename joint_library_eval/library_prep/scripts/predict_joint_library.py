#!/usr/bin/env python3
"""Predict joint library activity with a PyTorch AlphaGenome finetuned model.

Usage:
  python predict_joint_library.py --model_type k562 --model_name K562_twostep_v2
  python predict_joint_library.py --model_type hepg2 --model_name HepG2_twostep_v2
  python predict_joint_library.py --model_type wtc11 --model_name WTC11_twostep_v2

Each run loads one finetuned model, predicts on all ~57k joint library sequences,
and saves a per-model CSV + npy to joint_results/<model_type>_pred.{csv,npy}.

A merge step at the end combines all 3 per-model outputs into a single CSV
(only runs when all 3 per-model files exist).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent                  # .../scripts
LIBRARY_PREP_DIR = SCRIPT_DIR.parent                          # .../library_prep
JOINT_EVAL_DIR = LIBRARY_PREP_DIR.parent                      # .../joint_library_eval
TORCH_REPO = JOINT_EVAL_DIR.parent                            # .../alphagenome_torch_MPRAMoCon
LENTIMPRA_MCS = TORCH_REPO.parent                             # .../LentiMPRA_mcs

TRAINING_DIR = TORCH_REPO / "training"
RESULTS_BASE = TRAINING_DIR / "results"
WEIGHTS_DIR = TORCH_REPO / "weights"

# Alphagenome pytorch library
sys.path.insert(0, str(TORCH_REPO / "alphagenome-pytorch" / "src"))

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.transfer import remove_all_heads
from alphagenome_pytorch.extensions.finetuning.utils import sequence_to_onehot

# ============================================================
# Config
# ============================================================
ENCODER_DIM = 1536
ENCODER_RESOLUTION_BP = 128

PROMOTER_SEQ = "TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG"
RAND_BARCODE = "AGAGACTGAGGCCAC"
BATCH_SIZE = 64

# --- EDIT THIS to point at your joint library CSV ---
INPUT_CSV = (LENTIMPRA_MCS / "Cell_line_MoCon" / "Cross-line_analysis"
             / "pred_first" / "joint_data" / "joint_library_combined.csv")
OUTPUT_DIR = JOINT_EVAL_DIR / "joint_results"

# --- EDIT model_name values below to switch model versions ---
MODEL_CONFIGS = {
    "k562": {
        "pred_col": "k562_pred",
        "model_name": "K562_twostep_v2",      # <-- replace to change version
    },
    "hepg2": {
        "pred_col": "hepg2_pred",
        "model_name": "HepG2_twostep_v2",     # <-- replace to change version
    },
    "wtc11": {
        "pred_col": "wtc11_pred",
        "model_name": "WTC11_twostep_v2",     # <-- replace to change version
    },
}


# ============================================================
# MPRAHead (copied from training script for standalone use)
# ============================================================

class MPRAHead(nn.Module):
    """MLP head for MPRA activity score regression from encoder-only features."""

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

        if pooling_type == "flatten":
            in_dim = n_positions * ENCODER_DIM
        else:
            in_dim = ENCODER_DIM

        layers: list[nn.Module] = []
        for hs in hidden_sizes:
            layers.append(nn.Linear(in_dim, hs))
            in_dim = hs
        self.hidden_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output = nn.Linear(in_dim, 1)

        if activation == "gelu":
            self.act_fn = F.gelu
        else:
            self.act_fn = F.relu

    def _pool(self, x: Tensor) -> Tensor:
        if self.pooling_type == "flatten":
            return x.flatten(1)
        elif self.pooling_type == "center":
            center_idx = x.shape[1] // 2
            return x[:, center_idx, :]
        else:
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
            x = self.dropout(x)
            x = self.act_fn(x)
        x = self.output(x)
        return x.squeeze(-1)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict joint library with one PyTorch AlphaGenome model"
    )
    parser.add_argument(
        "--model_type", required=True, choices=["k562", "hepg2", "wtc11"],
        help="Which cell-line model to use for prediction",
    )
    parser.add_argument(
        "--model_name", default=None,
        help="Training results directory name (e.g. HepG2_twostep_v2). "
             "Overrides the default in MODEL_CONFIGS.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"Batch size for prediction (default: {BATCH_SIZE})",
    )
    return parser.parse_args()


# ============================================================
# Model loading
# ============================================================

def load_model(model_name: str, device: torch.device):
    """Load a finetuned PyTorch AlphaGenome model + MPRAHead from checkpoint.

    Supports both:
      - Stage 2 checkpoints (best_stage2.pt): contains model_state_dict + head_state_dict
      - Head-only checkpoints (best_head.pt): contains head_state_dict only
    """
    results_dir = RESULTS_BASE / model_name
    checkpoint_dir = results_dir / "checkpoints"
    args_path = results_dir / "args.json"

    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found at {args_path}")

    with open(args_path) as f:
        saved = json.load(f)
    hp = saved["hp"]

    # Determine weights path for base model
    weights_path = saved.get("weights")
    if weights_path is None:
        weights_path = str(WEIGHTS_DIR / "model_fold_0.safetensors")

    # Load base AlphaGenome model
    print(f"Loading base model from {weights_path} ...")
    model = AlphaGenome.from_pretrained(weights_path, device=device)
    model = remove_all_heads(model)

    # Build MPRAHead with same hyperparameters used during training
    sequence_length = hp.get("sequence_length", 384)
    n_positions = sequence_length // ENCODER_RESOLUTION_BP
    nl_size = hp.get("nl_size", 1024)
    if isinstance(nl_size, int):
        nl_size_list = [nl_size]
    else:
        nl_size_list = nl_size

    head = MPRAHead(
        n_positions=n_positions,
        nl_size=nl_size_list,
        dropout=hp.get("dropout", 0.1),
        activation=hp.get("activation", "relu"),
        pooling_type=hp.get("pooling_type", "flatten"),
        center_bp=hp.get("center_bp", 256),
    ).to(device)

    # Load checkpoint — prefer stage2, fall back to head-only
    stage2_path = checkpoint_dir / "best_stage2.pt"
    head_path = checkpoint_dir / "best_head.pt"

    if stage2_path.exists():
        print(f"Loading stage 2 checkpoint: {stage2_path}")
        ckpt = torch.load(stage2_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        head.load_state_dict(ckpt["head_state_dict"])
    elif head_path.exists():
        print(f"Loading head-only checkpoint: {head_path}")
        ckpt = torch.load(head_path, map_location=device, weights_only=True)
        head.load_state_dict(ckpt["head_state_dict"])
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}. "
            f"Expected best_stage2.pt or best_head.pt"
        )

    model.eval()
    head.eval()
    return model, head, hp


# ============================================================
# Prediction
# ============================================================

def pad_or_trim(onehot: np.ndarray, seq_len: int) -> np.ndarray:
    L = onehot.shape[0]
    if L < seq_len:
        pad = np.zeros((seq_len - L, 4), dtype=np.float32)
        return np.concatenate([onehot, pad], axis=0)
    return onehot[:seq_len]


@torch.no_grad()
def predict_all(
    model: nn.Module,
    head: MPRAHead,
    sequences: list[str],
    device: torch.device,
    sequence_length: int = 384,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Run predictions on all sequences.

    Args:
        model: AlphaGenome backbone (encoder).
        head: Finetuned MPRAHead.
        sequences: List of 230nt DNA sequence strings.
        device: torch device.
        sequence_length: Pad/trim length (must match training).
        batch_size: Prediction batch size.

    Returns:
        numpy array of predictions (n_sequences,)
    """
    use_amp = device.type == "cuda"
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    )

    all_preds = []
    n = len(sequences)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        batch_onehots = []
        for i in range(start, end):
            full_seq = sequences[i] + PROMOTER_SEQ + RAND_BARCODE
            oh = sequence_to_onehot(full_seq).astype(np.float32)
            oh = pad_or_trim(oh, sequence_length)
            batch_onehots.append(oh)

        batch_tensor = torch.from_numpy(np.stack(batch_onehots)).to(device)
        org_idx = torch.zeros(batch_tensor.shape[0], dtype=torch.long, device=device)

        with amp_ctx:
            enc_out = model(
                batch_tensor, org_idx, encoder_only=True
            )["encoder_output"].transpose(1, 2)
            preds = head(enc_out)

        all_preds.append(preds.float().cpu().numpy())

        if (start // batch_size) % 100 == 0:
            print(f"  {min(end, n)}/{n} sequences predicted...")

    return np.concatenate(all_preds)


# ============================================================
# Merge
# ============================================================

def merge_predictions():
    """Merge per-model prediction files into a single CSV.

    Only runs when all 3 per-model npy files exist.
    """
    all_exist = all(
        (OUTPUT_DIR / f"{cfg['pred_col']}_predictions.npy").exists()
        for cfg in MODEL_CONFIGS.values()
    )
    if not all_exist:
        print("Not all per-model predictions exist yet -- skipping merge.")
        return

    print("\nAll 3 per-model predictions found -- merging...")
    df = pd.read_csv(INPUT_CSV)

    for cfg in MODEL_CONFIGS.values():
        pred_col = cfg["pred_col"]
        preds = np.load(OUTPUT_DIR / f"{pred_col}_predictions.npy")
        df[pred_col] = preds

    output_csv = OUTPUT_DIR / "joint_library_predictions.csv"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=OUTPUT_DIR, suffix=".csv", prefix=".merge_tmp_")
    os.close(tmp_fd)
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, output_csv)

    print(f"Merged predictions saved to {output_csv}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    model_type = args.model_type
    cfg = MODEL_CONFIGS[model_type]
    pred_col = cfg["pred_col"]

    # CLI --model_name overrides the default in MODEL_CONFIGS
    model_name = args.model_name or cfg["model_name"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"Model type:  {model_type}")
    print(f"Model name:  {model_name}")
    print(f"Pred column: {pred_col}")
    print(f"Results dir: {RESULTS_BASE / model_name}")
    print(f"Device:      {device}")
    print(f"{'='*60}")

    print(f"\nLoading joint library from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df)} sequences loaded")

    valid_mask = df["sequence"].notna()
    n_missing = (~valid_mask).sum()
    if n_missing > 0:
        print(f"  Skipping {n_missing} rows with missing sequences")
    sequences = df.loc[valid_mask, "sequence"].tolist()

    model, head, hp = load_model(model_name, device)
    sequence_length = hp.get("sequence_length", 384)

    valid_preds = predict_all(
        model, head, sequences, device,
        sequence_length=sequence_length,
        batch_size=args.batch_size,
    )

    # Re-insert NaN for missing rows
    preds = np.full(len(df), np.nan, dtype=np.float32)
    preds[valid_mask.values] = valid_preds

    # Save predictions
    npy_path = OUTPUT_DIR / f"{pred_col}_predictions.npy"
    np.save(npy_path, preds)
    print(f"Saved: {npy_path}")

    df[pred_col] = preds
    csv_path = OUTPUT_DIR / f"{pred_col}_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    merge_predictions()
    print("\nDone!")


if __name__ == "__main__":
    main()
