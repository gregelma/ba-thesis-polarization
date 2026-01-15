from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_general_metrics(y_test, y_pred):
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def compute_detailed_metrics(y_test, y_pred, subtask):
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec = recall_score(y_test, y_pred, average=None, zero_division=0)

    lables = ["non_polarized", "polarized"] if subtask == 1 else y_test.columns

    return {
        label: {"f1": float(f1[i]), "precision": float(prec[i]), "recall": float(rec[i])}
        for i, label in enumerate(lables)
    }