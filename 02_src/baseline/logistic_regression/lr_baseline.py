from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from utils.config import SUBTASK_LANGUAGES
from baseline.logistic_regression.run_gridsearch import run_logreg_gridsearch, run_logreg_gridsearch_multilabel
from utils.io import save_best_params, save_results, save_details, save_exceptions
from utils.data import load_and_split_language
from utils.metrics import compute_general_metrics, compute_detailed_metrics

# -------------------------
# Parameter Grids
# -------------------------

def get_param_grid(analyzer, isMultilabel):
    clf_prefix = "classifier__estimator__" if isMultilabel else "classifier__"

    if analyzer == "char":
        param_grid = {
            "vectorizer__ngram_range": [(3,5)],
            "vectorizer__min_df": [2, 5],

            f"{clf_prefix}C": [0.1, 1, 10],
            f"{clf_prefix}class_weight": [None, "balanced"]
        }
    elif analyzer == "word":
        param_grid = {
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "vectorizer__min_df": [1, 2, 5],

            f"{clf_prefix}C": [0.1, 1, 10],
            f"{clf_prefix}class_weight": [None, "balanced"],
        }
    else:
        raise ValueError("analyzer must be 'char' or 'word'")

    return param_grid


# -------------------------
# Vectorizer
# -------------------------

def get_vectorizer(vectorizer_type, analyzer):
    if vectorizer_type == "count":
        return CountVectorizer(analyzer=analyzer)
    elif vectorizer_type == "tfidf":
        return TfidfVectorizer(analyzer=analyzer)
    else:
        raise ValueError("vectorizer_type must be 'count' or 'tfidf'")


# -------------------------
# Determine best baseline
# -------------------------

def determine_best_baseline(X_train, y_train, subtask):
    isMultilabel = subtask in (2,3)
    run_gridsearch = run_logreg_gridsearch_multilabel if isMultilabel else run_logreg_gridsearch

    best_model = None
    best_score = -float("inf")
    best_vectorizer_type = None
    best_analyzer = None
    best_params = None

    for vectorizer_type in ["count", "tfidf"]:
        for analyzer in ["word", "char"]:

            vectorizer = get_vectorizer(vectorizer_type, analyzer)
            param_grid = get_param_grid(analyzer, isMultilabel)

            model, score, params = run_gridsearch(
                X_train=X_train,
                y_train=y_train,
                vectorizer=vectorizer,
                param_grid=param_grid,
            )

            if score > best_score:
                best_model = model
                best_score = score
                best_vectorizer_type = vectorizer_type
                best_analyzer = analyzer
                best_params = params

    return best_model, best_score, best_vectorizer_type, best_analyzer, best_params


# -------------------------
# Run all languages
# -------------------------

def run_all_languages():
    for subtask, languages in SUBTASK_LANGUAGES.items():
        accuracy_by_lang = {}
        f1_macro_by_lang = {}

        for lang in languages:
            split = load_and_split_language(lang, subtask=subtask)
            X_train, y_train = split["train"]
            X_test, y_test = split["test"]

            if subtask in (2,3):
                valid_cols = [c for c in y_train.columns if y_train[c].nunique() > 1]
                dropped_cols = [c for c in y_train.columns if c not in valid_cols]

                y_train = y_train[valid_cols]
                y_test = y_test[valid_cols]
                save_exceptions("Logistic regression", lang, dropped_cols, subtask)

            model, score, vectorizer_type, analyzer, best_params = determine_best_baseline(
                X_train, y_train, subtask=subtask
            )

            save_best_params(
                lang,
                vectorizer_type,
                analyzer,
                best_params,
                score,
                f"../04_results/logistic_regression_config/subtask{subtask}",
            )

            y_pred = model.predict(X_test)

            general_metrics = compute_general_metrics(y_test, y_pred)
            accuracy_by_lang[lang] = general_metrics["accuracy"]
            f1_macro_by_lang[lang] = general_metrics["f1_macro"]

            details = compute_detailed_metrics(y_test, y_pred, subtask=subtask)
            save_details("Logistic regression", details, lang, subtask=subtask)

        save_results("Logistic regression", accuracy_by_lang, "accuracy", subtask=subtask)
        save_results("Logistic regression", f1_macro_by_lang, "f1_macro", subtask=subtask)


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    run_all_languages()