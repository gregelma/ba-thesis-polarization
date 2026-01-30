import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

from utils.config import SUBTASK_LANGUAGES


def save_best_params(language, vectorizer_type, analyzer, param_grid, cv_score, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = {
        "language": language,
        "vectorizer": vectorizer_type,
        "analyzer": analyzer,
        "best_params": param_grid,
        "cv_score": cv_score,
    }

    file_path = Path(output_dir) / f"{language}_config.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved params to {file_path}")


def load_best_params(language, input_dir):
    file_path = Path(input_dir) / f"{language}.json"

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(model_name, scores_by_lang, metric, subtask):
    csv_path = f"../04_results/test_results/{metric}_subtask{subtask}.csv"
    languages = SUBTASK_LANGUAGES[subtask]

    row_scores = {
        lang: float(scores_by_lang[lang]) if lang in scores_by_lang else np.nan
        for lang in languages
    }
    avg = np.nanmean(list(row_scores.values()))

    row = {
        "model": model_name,
        **row_scores,
        "average": avg
    }

    df_new = pd.DataFrame([row], columns=["model", *languages, "average"])

    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        df_new.to_csv(csv_path, index=False)


def save_details(model_name, details, language, subtask):
    csv_path = f"../04_results/test_results/subtask{subtask}_details.csv"

    rows = [
        {
            "model": model_name,
            "language": language,
            "label": label,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        for label, metrics in details.items()
    ]

    df_new = pd.DataFrame(rows, columns=["model", "language", "label", "f1", "precision", "recall"])

    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        df_new.to_csv(csv_path, index=False)


def save_exceptions(model_name, lang, dropped_cols, subtask):
    if not dropped_cols:
        return  # nothing to save
    csv_path = f"../04_results/test_results/subtask{subtask}_dropped_labels.csv"

    rows = [
        {
            "model": model_name,
            "language": lang,
            "label": label,
        }
        for label in dropped_cols
    ]

    df_new = pd.DataFrame(rows, columns=["model", "language", "label"])

    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        df_new.to_csv(csv_path, index=False)


def save_training_history(model_name, trainer, subtask, thresholds=None):
    rows = []

    for log in trainer.state.log_history:
        if "eval_f1_macro" in log:
            rows.append({
                "model": model_name,
                "epoch": log["epoch"],
                "val_loss": log.get("eval_loss"),
                "val_f1_macro": log["eval_f1_macro"],
                # thresholds als String speichern (CSV-freundlich)
                "thresholds": (
                    ",".join(map(str, thresholds)) if thresholds is not None else None
                )
            })
    df = pd.DataFrame(rows)

    out_path = f"../04_results/training_history/history_subtask{subtask}.csv"

    if os.path.exists(out_path):
        df.to_csv(out_path, mode="a", index=False, header=False)
    else:
        df.to_csv(out_path, index=False)