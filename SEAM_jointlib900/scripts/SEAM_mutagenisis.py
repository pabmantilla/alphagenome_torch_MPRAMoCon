"""Build mutagenesis libraries for SEAM analysis of the joint-900 library.

25K random mutants at 10% mutation rate for each of the 900 sequences
(100 per bin x 3 bins x 3 cell types). Saves gzipped HDF5, WT at index 0.

Usage:
    python SEAM_mutagenisis.py [--start START] [--end END]
"""

import argparse
import pickle
import numpy as np
import h5py
from pathlib import Path
import squid

# Config
LIB_SIZE = 25000
MUT_RATE = 0.10
SEQ_LENGTH = 230
SEED = 42

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_LIB = REPO_ROOT / "libraries/jointlib900_library.pkl"
OUT_DIR = REPO_ROOT / "results/mutagenesis_lib"

ALPHA_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def str_to_onehot(seq_str):
    ohe = np.zeros((len(seq_str), 4), dtype=np.float32)
    for j, base in enumerate(seq_str):
        if base in ALPHA_MAP:
            ohe[j, ALPHA_MAP[base]] = 1.0
    return ohe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start index (inclusive)')
    parser.add_argument('--end', type=int, default=None, help='End index (exclusive)')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TARGET_LIB, 'rb') as f:
        lib = pickle.load(f)
    df = lib['df'].reset_index(drop=True)

    end = args.end if args.end is not None else len(df)
    df = df.iloc[args.start:end]
    print(f"Processing {len(df)} sequences [{args.start}:{end}]")

    mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=MUT_RATE, seed=SEED)

    for i, (_, row) in enumerate(df.iterrows()):
        seq_idx = row["seq_idx"]
        condition = row["condition"]
        out_file = OUT_DIR / f"{condition}_{seq_idx}.h5"

        if out_file.exists():
            continue

        wt_onehot = str_to_onehot(row["sequence"])
        x_mut = mut_generator(wt_onehot, num_sim=LIB_SIZE - 1)
        # WT at index 0, 24999 mutants after -> 25000 total
        x_all = np.concatenate([wt_onehot[np.newaxis], x_mut], axis=0)

        with h5py.File(out_file, 'w') as f:
            f.create_dataset('sequences', data=x_all, dtype='float32',
                             compression='gzip', compression_opts=4)
            f.create_dataset('wt_sequence', data=wt_onehot, dtype='float32')
            f.attrs['seq_idx'] = int(seq_idx)
            f.attrs['name'] = str(row['name'])
            f.attrs['condition'] = condition
            f.attrs['cell_type'] = row['cell_type']
            f.attrs['actbin'] = row['actbin']
            f.attrs['pred_koo'] = float(row['pred_koo'])
            f.attrs['n_mutants'] = LIB_SIZE
            f.attrs['mut_rate'] = MUT_RATE
            f.attrs['seq_length'] = SEQ_LENGTH
            f.attrs['alphabet'] = 'ACGT'

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(df)}] {condition}/{seq_idx} saved")

    total = len(list(OUT_DIR.glob("*.h5")))
    print(f"\nDone! {total} H5 files in {OUT_DIR}")


if __name__ == '__main__':
    main()
