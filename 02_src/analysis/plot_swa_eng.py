import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
subtask = 1

ENG_FILE = f"../04_results/excel/subtask_{subtask}/s{subtask}_eng_zero.xlsx"
SWA_FILE = f"../04_results/excel/subtask_{subtask}/s{subtask}_swa_zero.xlsx"

eng = pd.read_excel(ENG_FILE, header=None)
swa = pd.read_excel(SWA_FILE, header=None)

cols = eng.iloc[0].tolist()
langs = [c for c in cols if str(c).lower() != "avg"]

model_names = ["mBERT", "XLM-R", "mDeBE"]

rows = []
for i, model in enumerate(model_names, start=1):
    eng_row = eng.iloc[i].tolist()
    swa_row = swa.iloc[i].tolist()

    eng_map = dict(zip(cols, eng_row))
    swa_map = dict(zip(cols, swa_row))

    for lg in langs:
        rows.append({
            "model": model,
            "lang": lg,
            "zs_en": float(eng_map[lg]),
            "zs_sw": float(swa_map[lg]),
        })

df = pd.DataFrame(rows)

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)

x = np.arange(len(langs))
w = 0.38

for ax, model in zip(axes, model_names):
    sub = df[df["model"] == model].set_index("lang").reindex(langs)

    ax.bar(x - w/2, sub["zs_en"].to_numpy(), width=w, label="Zero-shot English")
    ax.bar(x + w/2, sub["zs_sw"].to_numpy(), width=w, label="Zero-shot Swahili")

    ax.set_title(model)
    ax.set_ylabel("Macro F1")
    ax.legend()

    ax.tick_params(axis="x", labelbottom=True)

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=45, ha="right")

axes[-1].set_xlabel("Target language")

plt.tight_layout()
plt.savefig(f"../04_results/plots/s{subtask}_zero_shot_en_vs_swa_lang.png", dpi=300, bbox_inches="tight")
plt.show()