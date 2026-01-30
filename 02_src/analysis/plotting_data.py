import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

from utils.config import SUBTASK_LABELS, SUBTASK1_LABELS, SUBTASK2_LABELS, SUBTASK3_LABELS


def load_subtask_data(base_path):
    paths = glob.glob(os.path.join(base_path, "*.csv"))
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def plot_subtask1():
    data_s1 = load_subtask_data("../01_data/dev_phase/subtask1/train")

    counts = data_s1[SUBTASK1_LABELS[0]].value_counts().sort_index()

    plt.figure()
    plt.bar(["0", "1"], counts.values)
    plt.title("Subtask 1: Polarization Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("../04_results/plots/subtask1_label_distribution.png", dpi=200)
    plt.show()


def plot_subtask2():
    data_s2 = load_subtask_data("../01_data/dev_phase/subtask2/train")

    label_counts = data_s2[SUBTASK2_LABELS].sum()

    plt.figure()
    plt.bar(label_counts.index, label_counts.values)
    plt.title("Subtask 2: Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Positive Samples")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("../04_results/plots/subtask2_label_distribution.png", dpi=200)
    plt.show()


def plot_subtask3():
    data_s3 = load_subtask_data("../01_data/dev_phase/subtask3/train")

    label_counts = data_s3[SUBTASK3_LABELS].sum()

    plt.figure()
    plt.bar(label_counts.index, label_counts.values)
    plt.title("Subtask 3: Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Positive Samples")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("../04_results/plots/subtask3_label_distribution.png", dpi=200)
    plt.show()


def plot_language_distribution(subtask):
    language_counts = {}

    for path in glob.glob(os.path.join(f"../01_data/dev_phase/subtask{subtask}/train", "*.csv")):
        lang = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        language_counts[lang] = len(df)

    langs = list(language_counts.keys())
    counts = list(language_counts.values())

    plt.figure()
    plt.bar(langs, counts)
    plt.title(f"Subtask {subtask}: Language Distribution (Train Set)")
    plt.xlabel("Language")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"../04_results/plots/subtask{subtask}_language_distribution.png", dpi=200)
    plt.show()


def plot_lang_label_heatmap(subtask, annotate, log_scale):
    rows = []

    for path in glob.glob(os.path.join(f"../01_data/dev_phase/subtask{subtask}/train", "*.csv")):
        lang = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)

        counts = df[SUBTASK_LABELS[subtask]].sum()
        counts.name = lang
        rows.append(counts)

    mat = pd.DataFrame(rows).sort_index()  # languages x labels
    values = mat.values

    plt.figure()

    if log_scale:
        # avoid vmin=0 for LogNorm; use smallest positive value if it exists
        positive_vals = values[values > 0]
        vmin = positive_vals.min() if positive_vals.size else 1
        im = plt.imshow(values, aspect="auto", norm=LogNorm(vmin=vmin, vmax=values.max()))
    else:
        im = plt.imshow(values, aspect="auto")

    plt.colorbar(im, label="Positive samples (sum of 1s)")

    plt.title(f"Subtask {subtask}: Language x Label Distribution (Train)")
    plt.xlabel("Label")
    plt.ylabel("Language")

    plt.xticks(range(len(mat.columns)), mat.columns, rotation=30, ha="right")
    plt.yticks(range(len(mat.index)), mat.index)

    # ---- annotate cells with numbers ----
    if annotate:
        # choose a readable font size based on how big the matrix is
        fs = 9
        if mat.shape[0] > 25 or mat.shape[1] > 10:
            fs = 7
        if mat.shape[0] > 40 or mat.shape[1] > 15:
            fs = 6

        for i in range(mat.shape[0]):          # rows (languages)
            for j in range(mat.shape[1]):      # cols (labels)
                plt.text(
                    j, i, f"{int(values[i, j])}",
                    ha="center", va="center",
                    fontsize=fs,
                    color="black"
                )

    plt.tight_layout()
    plt.savefig(f"../04_results/plots/subtask{subtask}_lang_x_label_heatmap_{'log' if log_scale else 'linear'}_{'annotated' if annotate else 'not_annotated'}.png", dpi=200)
    plt.show()


def plot_subtask1_language_distribution():
    languages = []
    polarized = []
    non_polarized = []

    for path in glob.glob(os.path.join("../01_data/dev_phase/subtask1/train", "*.csv")):
        lang = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)

        pos = df["polarization"].sum()
        total = len(df)
        neg = total - pos

        languages.append(lang)
        polarized.append(pos)
        non_polarized.append(neg)

    plt.figure()
    plt.bar(languages, non_polarized, label="non-polarized")
    plt.bar(languages, polarized, bottom=non_polarized, label="polarized")

    plt.title("Subtask 1: Polarization Distribution by Language")
    plt.xlabel("Language")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../04_results/plots/subtask1_polarization_by_language.png", dpi=200)
    plt.show()

# if __name__ == "__main__":
#     plot_subtask1()
#     plot_subtask2()
#     plot_subtask3()
#     plot_language_distribution(1)
#     plot_language_distribution(2)
#     plot_language_distribution(3)
#     plot_subtask1_language_distribution()
#     plot_lang_label_heatmap(2, annotate=False, log_scale=False)
#     plot_lang_label_heatmap(2, annotate=False, log_scale=True)
#     plot_lang_label_heatmap(3, annotate=False, log_scale=False)
#     plot_lang_label_heatmap(3, annotate=False, log_scale=True)
#     plot_lang_label_heatmap(2, annotate=True, log_scale=False)
#     plot_lang_label_heatmap(2, annotate=True, log_scale=True)
#     plot_lang_label_heatmap(3, annotate=True, log_scale=False)
#     plot_lang_label_heatmap(3, annotate=True, log_scale=True)