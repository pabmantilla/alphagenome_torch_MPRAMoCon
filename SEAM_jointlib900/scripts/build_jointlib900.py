"""Build the joint-900 SEAM target library from the per-cell-type pred-stratified
libraries (100 per bin x 3 bins x 3 cell types = 900).

Each row's `cell_type` pins which koo model will attribute it later (HepG2 model
-> HepG2 seqs only, etc.). Mirrors the `lib['df']` shape the Hippo SEAM scripts
expect: columns seq_idx, condition, sequence (+ provenance cols).

Output: SEAM_jointlib900/libraries/jointlib900_library.pkl
"""

import pickle
from pathlib import Path
import pandas as pd

STRAT_DIR = Path(
    "/grid/koo/home/pmantill/projects/Virtual_Experiments/MoConSwap_mpra/"
    "data/stratified_libs"
)
OUT = Path(__file__).resolve().parent.parent / "libraries/jointlib900_library.pkl"
CELL_TYPES = ["HepG2", "K562", "WTC11"]

rows = []
for ct in CELL_TYPES:
    d = pd.read_csv(STRAT_DIR / f"{ct}_pred_stratified_lib.csv")
    assert len(d) == 300, (ct, len(d))
    for _, r in d.iterrows():
        rows.append({
            "cell_type": ct,
            "actbin": r[f"{ct}_actbin"],
            "name": r["name"],          # original library name (kept for provenance only)
            "condition": f"{ct}_{r[f'{ct}_actbin']}",
            "sequence": r["sequence"],
            "pred_koo": r[f"{ct}_pred_koo"],
        })

df = pd.DataFrame(rows).reset_index(drop=True)
# seq_idx = stable global row index (0..899). ALL file matching keys on this,
# never on `name` (some names are huge/pathological strings).
df.insert(0, "seq_idx", df.index.astype(int))
assert len(df) == 900, len(df)
assert df["seq_idx"].is_unique
assert (df["sequence"].str.len() == 230).all()
print(df.groupby(["cell_type", "actbin"]).size())
print("total", len(df))

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "wb") as f:
    pickle.dump({"df": df}, f)
print("wrote", OUT)
