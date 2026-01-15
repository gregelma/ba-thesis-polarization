import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import RANDOM_STATE, TEST_SIZE, VAL_SIZE, SUBTASK1_LABELS, SUBTASK2_LABELS, SUBTASK3_LABELS


# -------------------------
# Loading and Splitting Data (Subtask_1 & Subtask_2 & Subtask_3)
# -------------------------

def load_and_split_language(lang, subtask):
    # Subtask 1: binary labels (Series) -> stratified split
    # Subtask 2/3: multilabel (DataFrame) -> non-stratified split
    
    if subtask == 1:
        path = f"../01_data/dev_phase/subtask1/train/{lang}.csv"
        label_cols = SUBTASK1_LABELS[0]
    elif subtask == 2:
        path = f"../01_data/dev_phase/subtask2/train/{lang}.csv"
        label_cols = SUBTASK2_LABELS
    elif subtask == 3:
        path = f"../01_data/dev_phase/subtask3/train/{lang}.csv"
        label_cols = SUBTASK3_LABELS
    else:
        raise ValueError("subtask must be 1 or 2 or 3")

    df = pd.read_csv(path)

    X = df["text"]
    y = df[label_cols]

    stratify_key = y if subtask == 1 else None

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_key
    )

    val_relative_size = VAL_SIZE / (1 - TEST_SIZE)

    stratify_key_2 = y_train_val if subtask == 1 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=RANDOM_STATE,
        stratify=stratify_key_2
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }