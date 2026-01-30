from utils.config import SUBTASK_LANGUAGES
from utils.data import load_and_split_all_languages
from baseline.logistic_regression.lr_baseline import determine_best_baseline
from utils.io import save_best_params, save_results, save_details
from utils.metrics import evaluate_per_language, compute_detailed_metrics, evaluate_details_per_language


# -------------------------
# Run multilingual logistic regression
# -------------------------

def run_multilingual_lr():
    for subtask in [3]:
        split = load_and_split_all_languages(subtask=subtask, languages=SUBTASK_LANGUAGES[subtask])
        X_train, y_train = split["train"]
        X_test, y_test = split["test"]

        model, score, vectorizer_type, analyzer, best_params = determine_best_baseline(
            X_train["text"], y_train, subtask=subtask
        )
        
        save_best_params(
            'multilingual',
            vectorizer_type,
            analyzer,
            best_params,
            score,
            f"../04_results/logistic_regression_config/subtask{subtask}",
        )

        y_pred = model.predict(X_test["text"])

        accuracy_by_lang, f1_macro_by_lang = evaluate_per_language(X_test, y_test, y_pred)

        details_by_lang = evaluate_details_per_language(X_test, y_test, y_pred, subtask=subtask)

        for lang, details in details_by_lang.items():
            save_details("Multilingual Logistic regression", details, lang, subtask=subtask)

        save_results("Multilingual Logistic regression", accuracy_by_lang, "accuracy", subtask=subtask)
        save_results("Multilingual Logistic regression", f1_macro_by_lang, "f1_macro", subtask=subtask)


if __name__ == "__main__":
    run_multilingual_lr()