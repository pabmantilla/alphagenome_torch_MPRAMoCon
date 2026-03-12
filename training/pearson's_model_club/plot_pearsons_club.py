"""Pearson's Model Club: bar chart comparing best model r vs baselines."""
import matplotlib.pyplot as plt
import numpy as np
import csv, os

# --- Baselines (provided by user) ---
baselines = {"K562": 0.885, "HepG2": 0.877, "WTC11": 0.822}

# --- Read completed models from best_models.csv ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "best_models.csv")

models = {}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        models[row["cell_type"]] = float(row["pearson_r"])

# --- Build plot data ---
cell_types = ["HepG2", "K562", "WTC11"]
baseline_vals = [baselines[c] for c in cell_types]
model_vals = [models.get(c, None) for c in cell_types]

fig, ax = plt.subplots(figsize=(7, 5))

x = np.arange(len(cell_types))
bar_w = 0.32

# Baseline bars
bars_base = ax.bar(x - bar_w / 2, baseline_vals, bar_w,
                   label="Baseline", color="#7fb3d8", edgecolor="black", linewidth=0.6)

# Model bars (skip if not finished)
model_colors = []
model_x = []
model_v = []
labels_added = set()
for i, (ct, v) in enumerate(zip(cell_types, model_vals)):
    if v is not None:
        model_x.append(x[i] + bar_w / 2)
        model_v.append(v)
        model_colors.append("#f4845f")

if model_v:
    bars_model = ax.bar(model_x, model_v, bar_w,
                        label="AlphaGenome-FT (ours)", color="#f4845f",
                        edgecolor="black", linewidth=0.6)

# Annotate bars with values
for bar in bars_base:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
if model_v:
    for bar in bars_model:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

# Mark K562 as running if missing
for i, ct in enumerate(cell_types):
    if model_vals[i] is None:
        ax.text(x[i] + bar_w / 2, baselines[ct] * 0.5, "running...",
                ha="center", va="center", fontsize=9, fontstyle="italic", color="gray")

ax.set_xticks(x)
ax.set_xticklabels(cell_types, fontsize=12)
ax.set_ylabel("Pearson r (test)", fontsize=12)
ax.set_title("Pearson's Model Club", fontsize=14, fontweight="bold")
ax.set_ylim(0.70, 0.96)
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = os.path.join(script_dir, "pearsons_model_club.png")
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.close()
