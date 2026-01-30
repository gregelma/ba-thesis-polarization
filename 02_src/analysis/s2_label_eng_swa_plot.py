import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



subtask = 2

CSV_PATH = f"../04_results/test_results/subtask{subtask}_details.csv"
METRIC = "f1"                             # or "precision" / "recall"

# Keep only the models you care about (adjust if you have more)
RELEVANT_MODELS = {
    # Zero-shot EN
    "zero_shot_eng_bert-base-multilingual-cased_1",
    "zero_shot_eng_bert-base-multilingual-cased_2",
    "zero_shot_eng_bert-base-multilingual-cased_3",
    "zero_shot_eng_xlm-roberta-base_1",
    "zero_shot_eng_xlm-roberta-base_2",
    "zero_shot_eng_xlm-roberta-base_3",
    "zero_shot_eng_microsoft_mdeberta-v3-base_1",
    "zero_shot_eng_microsoft_mdeberta-v3-base_2",
    "zero_shot_eng_microsoft_mdeberta-v3-base_3",

    # Zero-shot SWA (optional — include only if present)
    "zero_shot_swa_bert-base-multilingual-cased_1",
    "zero_shot_swa_bert-base-multilingual-cased_2",
    "zero_shot_swa_bert-base-multilingual-cased_3",
    "zero_shot_swa_xlm-roberta-base_1",
    "zero_shot_swa_xlm-roberta-base_2",
    "zero_shot_swa_xlm-roberta-base_3",
    "zero_shot_swa_microsoft_mdeberta-v3-base_1",
    "zero_shot_swa_microsoft_mdeberta-v3-base_2",
    "zero_shot_swa_microsoft_mdeberta-v3-base_3",
}

# Display names in final table
MODEL_DISPLAY = {
    "mBERT": "mBERT",
    "XLM-R": "XLM-R",
    "mDeBERTa": "mDeBERTa",
}

REGIME_DISPLAY = {
    "zs_eng": "Zero-shot EN",
    "zs_swa": "Zero-shot SWA",
}

def parse_model(m: str):
    """
    Returns (regime, family, seed)
      regime: "zs_eng" or "zs_swa"
      family: {"mBERT","XLM-R","mDeBERTa"}
      seed: int
    """
    # regime prefix
    if m.startswith("zero_shot_eng_"):
        regime = "zs_eng"
        base = m[len("zero_shot_eng_"):]
    elif m.startswith("zero_shot_swa_"):
        regime = "zs_swa"
        base = m[len("zero_shot_swa_"):]
    else:
        raise ValueError(f"Unknown regime prefix in model: {m}")

    # seed
    match = re.search(r"_(\d+)$", base)
    seed = int(match.group(1)) if match else None
    base_no_seed = re.sub(r"_(\d+)$", "", base)

    # family mapping
    if base_no_seed == "bert-base-multilingual-cased":
        family = "mBERT"
    elif base_no_seed == "xlm-roberta-base":
        family = "XLM-R"
    elif base_no_seed == "microsoft_mdeberta-v3-base":
        family = "mDeBERTa"
    else:
        raise ValueError(f"Unknown model base: {base_no_seed}")

    return (regime, family, seed)

def format_de(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")

# -----------------------------
# Load + filter
# -----------------------------
df = pd.read_csv(CSV_PATH)

needed_cols = {"model", "language", "label", METRIC}
missing = needed_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV missing columns: {missing}")

df = df[df["model"].isin(RELEVANT_MODELS)].copy()
if df.empty:
    raise ValueError("No rows left after filtering. Check RELEVANT_MODELS and model names in the CSV.")

parsed = df["model"].apply(parse_model)
df["regime"] = parsed.apply(lambda x: x[0])   # zs_eng / zs_swa
df["family"] = parsed.apply(lambda x: x[1])   # mBERT / XLM-R / mDeBERTa
df["seed"]   = parsed.apply(lambda x: x[2])   # 1/2/3

# -----------------------------
# 1) Average over seeds (within each regime, family, language, label)
# -----------------------------
seed_avg = (
    df.groupby(["regime", "family", "language", "label"], as_index=False)[METRIC]
      .mean()
)

seed_avg = seed_avg[
    ~(
        (seed_avg["regime"] == "zs_eng") & (seed_avg["language"] == "eng")
    ) &
    ~(
        (seed_avg["regime"] == "zs_swa") & (seed_avg["language"] == "swa")
    )
]


# -----------------------------
# 2) Average over languages -> label-wise difficulty table base
# -----------------------------
label_avg = (
    seed_avg.groupby(["regime", "family", "label"], as_index=False)[METRIC]
            .mean()
)

# -----------------------------
# 3) Pivot wide: columns = (family, regime), index = label
# -----------------------------
wide_num = label_avg.pivot_table(
    index="label",
    columns=["family", "regime"],
    values=METRIC
)

# Order columns nicely
col_order = []
for reg in ["zs_eng", "zs_swa"]:
    for fam in ["mBERT", "XLM-R", "mDeBERTa"]:
        if (fam, reg) in wide_num.columns:
            col_order.append((fam, reg))
wide_num = wide_num[col_order]

# -----------------------------
# 4) Add Avg column + Avg row
# -----------------------------
avg_row = wide_num.mean(axis=0).to_frame().T         # per column
avg_row.index = ["Avg"]
wide_num = pd.concat([wide_num, avg_row], axis=0)

# -----------------------------
# 5) Scale to 0..100 and format
# -----------------------------
wide_num = wide_num * 100

# Flatten column names (except "Avg")
flat_cols = []
for c in wide_num.columns:
    if c == "Avg":
        flat_cols.append("Avg")
    else:
        fam, reg = c
        flat_cols.append(f"{MODEL_DISPLAY.get(fam, fam)} ({REGIME_DISPLAY.get(reg, reg)})")
wide_num.columns = flat_cols

wide_fmt = wide_num.applymap(format_de).reset_index().rename(columns={"index": "Label"})

print("\n=== Label-wise table (seeds averaged, then languages averaged), + Avg row/col ===")
print(wide_fmt.to_string(index=False))


plot_df = label_avg.copy()
plot_df[METRIC] = plot_df[METRIC] * 100  # scale to 0–100

LABEL_ORDER = [
    "political",
    "racial/ethnic",
    "religious",
    "gender/sexual",
    "other",
]

labels = [l for l in LABEL_ORDER if l in plot_df["label"].unique()]
models = ["mBERT", "XLM-R", "mDeBERTa"]
regimes = ["zs_eng", "zs_swa"]

# -------------------------------------------------
# Plot: 3 models in one row, EN vs SWA per label
# -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

bar_width = 0.35
x = np.arange(len(labels))

for ax, model in zip(axes, models):
    sub = plot_df[plot_df["family"] == model]

    en_vals = (
        sub[sub["regime"] == "zs_eng"]
        .set_index("label")
        .reindex(labels)[METRIC]
        .values
    )

    swa_vals = (
        sub[sub["regime"] == "zs_swa"]
        .set_index("label")
        .reindex(labels)[METRIC]
        .values
    )

    ax.bar(x - bar_width / 2, en_vals, bar_width, label="Zero-shot English")
    ax.bar(x + bar_width / 2, swa_vals, bar_width, label="Zero-shot Swahili")

    ax.set_title(model)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 100)

axes[0].set_ylabel("Macro F1")
axes[0].legend(loc="upper left")

plt.tight_layout()
plt.savefig(f"../04_results/plots/s{subtask}_zero_shot_en_vs_swa_label.png", dpi=300)
plt.show()
