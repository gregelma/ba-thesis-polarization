import os
import json
from pathlib import Path
import pandas as pd

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
    csv_path = f"../04_results/{metric}_subtask{subtask}.csv"
    languages = SUBTASK_LANGUAGES[subtask]

    avg = sum(float(scores_by_lang[lang]) for lang in languages) / len(languages)

    row = {"model": model_name, **{lang: float(scores_by_lang[lang]) for lang in languages}, "average": avg}
    df_new = pd.DataFrame([row], columns=["model", *languages, "average"])

    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        df_new.to_csv(csv_path, index=False)


def save_details(model_name, details, language, subtask):
    csv_path = f"../04_results/subtask{subtask}_details.csv"

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
    csv_path = f"../04_results/subtask{subtask}_dropped_labels.csv"

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