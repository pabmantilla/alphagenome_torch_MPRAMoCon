"""Predictions + DeepLIFT/SHAP attributions for the joint-900 SEAM library using
the standardized AlphaGenome encoder ckpts (EncoderMPRAModel.from_checkpoint).

Each cell line's model attributes ONLY its own sequences: --cell-type filters the
joint-900 df to that cell type's 300 seqs (HepG2 model -> HepG2 seqs, etc.).

Gradient (hypothetical) correction `attr -= attr.mean(channel)` is applied
immediately after deep_lift_shap and BEFORE the H5 is written, so the clustered
maps downstream are the corrected ones.

Usage:
    python SEAM_attr_standardtorch.py --cell-type K562 [--start S] [--end E]
                                       [--n-shuffles 20]
"""

import argparse
import gc
import sys
import time
import pickle
import numpy as np
import h5py
import torch
import torch.nn as nn
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EI_DIR = (REPO_ROOT / "Hippo_axis/Hippo_dependency_mpra/eigen-interactions")
sys.path.insert(0, str(EI_DIR))

from ag_deeplift_patches import patch_alphagenome, AGCustomGELU
from tangermeme.deep_lift_shap import deep_lift_shap, _nonlinear
from tangermeme.ersatz import dinucleotide_shuffle

from alphagenome_encoder_ft import EncoderMPRAModel

patch_alphagenome()

ENHANCER_LEN = 230

CKPT_PATHS = {
    'K562':  '/grid/koo/home/shared/models/alphagenome_encoder/torch/mpra_K562/finetuned_encoder.pt',
    'HepG2': '/grid/koo/home/shared/models/alphagenome_encoder/torch/mpra_HepG2/finetuned_encoder.pt',
    'WTC11': '/grid/koo/home/shared/models/alphagenome_encoder/torch/mpra_WTC11/finetuned_encoder.pt',
}

SEAM_ROOT = Path(__file__).resolve().parent.parent
TARGET_LIB = SEAM_ROOT / "libraries/jointlib900_library.pkl"
MUT_LIB_DIR = SEAM_ROOT / "results/mutagenesis_lib"
OUT_DIR = SEAM_ROOT / "results/attributions"


class TransposeWrapper(nn.Module):
    """Tangermeme passes (B, 4, L); EncoderMPRAModel expects (B, L, 4).
    Returns (B, 1) so deep_lift_shap target=0 indexes the scalar output."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x.transpose(-1, -2))
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out


ALPHA_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def str_to_onehot_cf(seq_str):
    ohe = np.zeros((4, len(seq_str)), dtype=np.float32)
    for j, base in enumerate(seq_str):
        if base in ALPHA_MAP:
            ohe[ALPHA_MAP[base], j] = 1.0
    return ohe


def get_construct_seqs(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cc = ck['construct_config']
    return cc['promoter_seq'], cc['barcode_seq']


def load_model(cell_type, device='cuda'):
    ckpt_path = CKPT_PATHS[cell_type]
    base = EncoderMPRAModel.from_checkpoint(ckpt_path, device=device).eval()
    return TransposeWrapper(base).to(device).eval()


def pad_to_281(x_enhancer_cf, promoter_seq, barcode_seq):
    """(N, 4, 230) -> (N, 4, 281) by appending promoter+barcode."""
    construct_suffix = promoter_seq + barcode_seq
    suffix_ohe = str_to_onehot_cf(construct_suffix)
    suffix_tiled = np.tile(suffix_ohe, (len(x_enhancer_cf), 1, 1))
    return np.concatenate([x_enhancer_cf, suffix_tiled], axis=2)


def compute_predictions(model, x_cf, batch_size=512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_cf), batch_size):
            batch = torch.from_numpy(x_cf[i:i+batch_size]).float().cuda()
            preds.append(model(batch).squeeze(-1).cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def process_sequence(model, seq_idx, condition, mut_path, out_path,
                     n_shuffles, cell_type, promoter_seq, barcode_seq):
    if out_path.exists():
        with h5py.File(out_path, 'r') as f:
            if 'predictions' in f and 'attributions' in f:
                return False

    with h5py.File(mut_path, 'r') as f:
        all_nlc = f['sequences'][:]

    all_cf = all_nlc.transpose(0, 2, 1)
    del all_nlc

    all_281 = pad_to_281(all_cf, promoter_seq, barcode_seq)
    del all_cf

    print(f"    predictions...")
    predictions = compute_predictions(model, all_281)
    print(f"    -> wt_pred={predictions[0]:.4f}")

    var_start, var_end = 15, 215  # 200bp variable region inside the 230bp insert (15bp adapters each side)

    X = torch.from_numpy(all_281).float()
    t0 = time.time()
    wt_tensor = X[0:1]  # (1, 4, 281)
    print(f"    computing {n_shuffles} dinucleotide shuffles of WT variable region...")
    var_shuf = dinucleotide_shuffle(wt_tensor[:, :, var_start:var_end].cpu(), n=n_shuffles, random_state=42)
    wt_refs = wt_tensor.unsqueeze(1).expand(1, n_shuffles, 4, 281).contiguous()
    wt_refs[:, :, :, var_start:var_end] = var_shuf
    refs = wt_refs.expand(len(X), -1, -1, -1).to('cuda')
    print(f"    shuffles done in {(time.time()-t0)/60:.1f}min")

    print(f"    deep_lift_shap...")
    attr = deep_lift_shap(
        model, X, target=0, references=refs,
        hypothetical=True, batch_size=512,
        device='cuda',
        additional_nonlinear_ops={AGCustomGELU: _nonlinear},
        warning_threshold=0.01, verbose=False,
    ).cpu().numpy()

    # Gradient (hypothetical) correction: mean-center across the 4 nucleotide
    # channels. Done HERE, before the H5 is written, so the attributions the
    # clusterer reads downstream are already corrected.
    attr = attr - attr.mean(axis=1, keepdims=True)
    attr_var = attr[:, :, var_start:var_end]
    attr_nlc = attr_var.transpose(0, 2, 1)

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('predictions', data=predictions)
        f.create_dataset('attributions', data=attr_nlc,
                         compression='gzip', compression_opts=4)
        f.attrs['seq_idx'] = int(seq_idx)
        f.attrs['condition'] = condition
        f.attrs['cell_type'] = cell_type
        f.attrs['n_shuffles'] = n_shuffles
        f.attrs['alphabet'] = 'ACGT'
        f.attrs['format'] = 'NLC'

    elapsed = (time.time() - t0) / 60
    print(f"    -> {attr_nlc.shape} in {elapsed:.1f}min")

    del X, refs, attr, attr_var, attr_nlc, all_281
    gc.collect()
    torch.cuda.empty_cache()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--n-shuffles', type=int, default=20)
    parser.add_argument('--cell-type', type=str, required=True,
                        choices=['K562', 'HepG2', 'WTC11'])
    args = parser.parse_args()

    ct = args.cell_type
    out_dir = OUT_DIR / ct
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(TARGET_LIB, 'rb') as f:
        lib = pickle.load(f)
    # each cell line's model attributes ONLY its own seqs
    df = lib['df']
    df = df[df['cell_type'] == ct].reset_index(drop=True)

    end = args.end if args.end is not None else len(df)
    df = df.iloc[args.start:end]
    print(f"Processing {len(df)} {ct} sequences [{args.start}:{end}]")

    ckpt_path = CKPT_PATHS[ct]
    promoter_seq, barcode_seq = get_construct_seqs(ckpt_path)
    assert len(promoter_seq) == 36, f"promoter_seq len {len(promoter_seq)} != 36"
    assert len(barcode_seq) == 15, f"barcode_seq len {len(barcode_seq)} != 15"
    print(f"  promoter ({len(promoter_seq)}bp): {promoter_seq}")
    print(f"  barcode  ({len(barcode_seq)}bp): {barcode_seq}")

    print(f"Loading {ct} model from {ckpt_path}...")
    model = load_model(ct)

    for i, (_, row) in enumerate(df.iterrows()):
        seq_idx = row['seq_idx']
        condition = row['condition']
        mut_path = MUT_LIB_DIR / f"{condition}_{seq_idx}.h5"
        out_path = out_dir / f"{condition}_{seq_idx}.h5"

        if not mut_path.exists():
            print(f"  [{i+1}/{len(df)}] {condition}/{seq_idx} - mutagenesis lib not found, skipping")
            continue

        print(f"  [{i+1}/{len(df)}] {condition}/{seq_idx}")
        try:
            if not process_sequence(model, seq_idx, condition, mut_path, out_path,
                                    args.n_shuffles, ct, promoter_seq, barcode_seq):
                print(f"    complete, skipping")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
