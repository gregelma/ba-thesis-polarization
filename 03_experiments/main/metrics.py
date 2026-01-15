import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }

def compute_language_macro_metrics(eval_pred, languages):
    """
    eval_pred: logits, labels
    languages: Liste der Sprachen in der gleichen Reihenfolge wie die Beispiele
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    results = {}
    macro_f1_list = []

    unique_langs = np.unique(languages)
    for lang in unique_langs:
        mask = np.array(languages) == lang
        if mask.sum() == 0:
            continue
        lang_labels = labels[mask]
        lang_preds = preds[mask]
        f1 = f1_score(lang_labels, lang_preds, average="macro")
        results[f"macro_f1_{lang}"] = f1
        macro_f1_list.append(f1)

    # Durchschnitt über alle Sprachen
    results["macro_f1_lang_balanced"] = np.mean(macro_f1_list)
    results["accuracy"] = accuracy_score(labels, preds)

    return results