import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
subtask = 1



LOLO_FILE = f"../04_results/excel/subtask_{subtask}/s{subtask}_lolo.xlsx"
MULTI_FILE = f"../04_results/excel/subtask_{subtask}/s{subtask}_multi_lolo.xlsx"

models = ["mBERT", "XLM-R", "mDeBE"]

def read_table(path):
    df = pd.read_excel(path, header=None)
    cols = df.iloc[0].tolist()
    langs = [c for c in cols if str(c).lower() != "avg"]

    vals = df.iloc[1:1+len(models)].copy()
    vals.columns = cols
    vals.index = models
    return langs, vals[langs].astype(float)

langs, lolo = read_table(LOLO_FILE)
_, multi = read_table(MULTI_FILE)

x = np.arange(len(langs))
offsets = [-0.22, 0.0, 0.22]  # one per model
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plt.figure(figsize=(12, 5))

for i, model in enumerate(models):
    xi = x + offsets[i]

    y_lolo = lolo.loc[model]
    y_multi = multi.loc[model]
    delta = y_multi - y_lolo

    # vertical connecting line
    plt.vlines(
        xi,
        y_lolo,
        y_multi,
        color=colors[i],
        linewidth=2,
        alpha=0.8
    )

    # dots
    plt.scatter(xi, y_lolo, color=colors[i], s=40)
    plt.scatter(xi, y_multi, color=colors[i], s=40)

    # ---- ΔF1 annotations (NEW) ----
    for xj, y_top, d in zip(xi, y_multi, delta):
        plt.text(
            xj,
            y_top + 0.6,
            f"+{d:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors[i]
        )

plt.xticks(x, langs)
plt.ylabel("Macro F1")
plt.xlabel("Target language (held out)")
plt.title(f"Subtask {subtask}: Leave-One-Language-Out vs Multilingual")

# Legend (models only)
handles = [
    plt.Line2D([0], [0], color=colors[i], lw=2, marker="o", label=models[i])
    for i in range(len(models))
]
plt.legend(handles=handles, title="Model")

plt.tight_layout()
plt.savefig(f"../04_results/plots/s{subtask}_lolo_vs_multi_vertical_dumbbell_annotated.png", dpi=300, bbox_inches="tight")
plt.show()
