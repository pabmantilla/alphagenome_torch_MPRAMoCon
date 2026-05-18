"""SEAM clustering + MetaExplainer background separation for the joint-900 lib.

Loads the pre-computed predictions + gradient-corrected attributions (from
SEAM_attr_standardtorch.py), runs SEAM K-means clustering (30 clusters) and
MetaExplainer to extract scaled foreground/background. --cell-type restricts to
that cell type's own 300 seqs (matching the per-cell-line attribution).

The attributions loaded here are already hypothetical/gradient-corrected (the
mean-center across channels was applied before they were written), so the
clustered maps are the corrected ones.

Usage:
    python SEAM_explainer.py --cell-type K562 [--start S] [--end E]
"""

import argparse
import pickle
import gc
import numpy as np
import h5py
from pathlib import Path

from seam import Compiler, Clusterer, MetaExplainer

# Config
N_CLUSTERS = 30
MUT_RATE = 0.10
ALPHABET = ['A', 'C', 'G', 'T']

SEAM_ROOT = Path(__file__).resolve().parent.parent
TARGET_LIB = SEAM_ROOT / "libraries/jointlib900_library.pkl"
MUT_LIB_DIR = SEAM_ROOT / "results/mutagenesis_lib"
ATTR_DIR = SEAM_ROOT / "results/attributions"
OUT_DIR = SEAM_ROOT / "results/foregrounds"


def process_sequence(seq_idx, condition, cell_type):
    prefix = f"{condition}_{seq_idx}"
    mut_path = MUT_LIB_DIR / f"{prefix}.h5"
    attr_path = ATTR_DIR / cell_type / f"{prefix}.h5"
    seq_dir = OUT_DIR / cell_type / str(seq_idx)
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Load mutagenesis library (WT at index 0, 25000 total)
    with h5py.File(mut_path, 'r') as f:
        x_mut = f['sequences'][:]  # (25000, 230, 4) NLC — WT is index 0
        wt_seq = f['wt_sequence'][:]  # (230, 4)

    # Load predictions + attributions (NLC ACGT; already gradient-corrected)
    with h5py.File(attr_path, 'r') as f:
        predictions = f['predictions'][:]     # (25000,)
        attributions = f['attributions'][:]   # (25000, 200, 4)

    # K-means clustering on flattened attribution maps
    clusterer = Clusterer(attributions, gpu=False)
    cluster_labels = clusterer.cluster(
        embedding=clusterer.maps,
        method='kmeans',
        n_clusters=N_CLUSTERS
    )

    # Compile MAVE dataframe
    compiler = Compiler(
        x=x_mut,
        y=predictions,
        x_ref=wt_seq[np.newaxis],
        y_bg=None,
        alphabet=ALPHABET,
        gpu=False
    )
    mave_df = compiler.compile()

    # Fresh clusterer with labels for MetaExplainer
    clusterer = Clusterer(attributions, gpu=False)
    clusterer.cluster_labels = cluster_labels

    # MetaExplainer: sort clusters, MSM, background separation
    meta = MetaExplainer(
        clusterer=clusterer,
        mave_df=mave_df,
        attributions=attributions,
        sort_method='median',
        ref_idx=0,
        mut_rate=MUT_RATE
    )
    msm = meta.generate_msm(gpu=False)
    meta.compute_background(
        mut_rate=MUT_RATE,
        entropy_multiplier=0.5,
        adaptive_background_scaling=True,
        process_logos=False
    )

    # Get WT reference cluster and compute foreground
    if meta.cluster_order is not None:
        mapping = {old: new for new, old in enumerate(meta.cluster_order)}
        meta.membership_df['Cluster_Sorted'] = meta.membership_df['Cluster'].map(mapping)
        ref_cluster = meta.membership_df.loc[0, 'Cluster_Sorted']
    else:
        ref_cluster = meta.membership_df.loc[0, 'Cluster']

    ref_cluster_avg = np.mean(meta.get_cluster_maps(ref_cluster), axis=0)
    bg_scale = meta.background_scaling[ref_cluster] if meta.background_scaling is not None else 1.0
    foreground_scaled = ref_cluster_avg - bg_scale * meta.background

    # Per-cluster average maps
    cluster_maps = np.stack([
        np.mean(meta.get_cluster_maps(k), axis=0) for k in range(N_CLUSTERS)
    ])  # (N_CLUSTERS, 200, 4)

    # Save outputs (write to tmp, rename to avoid corrupt partial writes)
    for name, arr in [
        ('foreground_scaled', foreground_scaled),
        ('average_background', meta.background),
        ('average_background_scaled', bg_scale * meta.background),
        ('wt_attribution', attributions[0]),
        ('ref_cluster_avg', ref_cluster_avg),
        ('cluster_labels', cluster_labels),
        ('cluster_maps', cluster_maps),
        ('cluster_backgrounds', meta.cluster_backgrounds),
        ('ref_cluster_idx', np.array(ref_cluster)),
    ]:
        tmp = seq_dir / f'.{name}_tmp'
        np.save(tmp, arr)  # writes .{name}_tmp.npy
        (seq_dir / f'.{name}_tmp.npy').rename(seq_dir / f'{name}.npy')

    print(f"    bg_scale={bg_scale:.4f}, ref_cluster={ref_cluster}")

    del x_mut, wt_seq, predictions, attributions, clusterer, compiler, mave_df, meta
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--cell-type', type=str, required=True,
                        choices=['K562', 'HepG2', 'WTC11'])
    args = parser.parse_args()

    ct = args.cell_type
    (OUT_DIR / ct).mkdir(parents=True, exist_ok=True)

    with open(TARGET_LIB, 'rb') as f:
        lib = pickle.load(f)
    df = lib['df']
    df = df[df['cell_type'] == ct].reset_index(drop=True)

    end = args.end if args.end is not None else len(df)
    df = df.iloc[args.start:end]
    print(f"Processing {len(df)} {ct} sequences [{args.start}:{end}]")

    for i, (_, row) in enumerate(df.iterrows()):
        seq_idx = row['seq_idx']
        condition = row['condition']
        seq_dir = OUT_DIR / ct / str(seq_idx)

        check_path = seq_dir / 'cluster_backgrounds.npy'
        if check_path.exists():
            try:
                np.load(check_path)
                continue
            except Exception:
                pass  # corrupt, recompute

        attr_path = ATTR_DIR / ct / f"{condition}_{seq_idx}.h5"
        if not attr_path.exists():
            print(f"  [{i+1}/{len(df)}] {seq_idx} — attributions not found, skipping")
            continue

        print(f"  [{i+1}/{len(df)}] {condition}/{seq_idx}")
        try:
            process_sequence(seq_idx, condition, ct)
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
