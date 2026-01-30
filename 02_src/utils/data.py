import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

from utils.config import RANDOM_STATE, TEST_SIZE, VAL_SIZE, SUBTASK1_LABELS, SUBTASK2_LABELS, SUBTASK3_LABELS, MAX_LENGTH, SUBTASK_LABELS


# -----------------------------------------------------------------------------------
# Loading and splitting data for a given language (Subtask_1 & Subtask_2 & Subtask_3)
# -----------------------------------------------------------------------------------

def load_and_split_language(lang, subtask):
    # Subtask 1: binary labels (Series) -> stratified split
    # Subtask 2/3: multilabel (DataFrame) -> non-stratified split
    
    label_cols, path = _get_label_columns_and_path(subtask, lang)

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


# ---------------------------------------------------------------------------------
# Loading and splitting data for every language (Subtask_1 & Subtask_2 & Subtask_3)
# ---------------------------------------------------------------------------------

def load_and_split_all_languages(subtask, languages):
    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []
    X_test_all, y_test_all = [], []

    for lang in languages:
        split = load_and_split_language(lang, subtask)

        X_train, y_train = split["train"]
        X_val, y_val     = split["val"]
        X_test, y_test   = split["test"]

        # Convert Series -> DataFrame with columns ["text", "lang"]
        X_train = pd.DataFrame({"text": X_train, "lang": lang})
        X_val   = pd.DataFrame({"text": X_val,   "lang": lang})
        X_test  = pd.DataFrame({"text": X_test,  "lang": lang})

        X_train_all.append(X_train)
        y_train_all.append(y_train)

        X_val_all.append(X_val)
        y_val_all.append(y_val)

        X_test_all.append(X_test)
        y_test_all.append(y_test)

    return {
        "train": (pd.concat(X_train_all, ignore_index=True),
                  pd.concat(y_train_all, ignore_index=True)),
        "val":   (pd.concat(X_val_all, ignore_index=True),
                  pd.concat(y_val_all, ignore_index=True)),
        "test":  (pd.concat(X_test_all, ignore_index=True),
                  pd.concat(y_test_all, ignore_index=True)),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Loading and splitting data for a given language for the competition (no test set) (Subtask_1 & Subtask_2 & Subtask_3)
# ---------------------------------------------------------------------------------------------------------------------

def load_and_split_language_for_comp(lang, subtask):
    label_cols, path = _get_label_columns_and_path(subtask, lang)

    df = pd.read_csv(path)

    X = df["text"]
    y = df[label_cols]

    stratify_key = y if subtask == 1 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_key
    )
    
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
    }


# -------------------------------------------------------------------------------------------------------------------
# Loading and splitting data for every language for the competition (no test set) (Subtask_1 & Subtask_2 & Subtask_3)
# -------------------------------------------------------------------------------------------------------------------

def load_and_split_all_languages_for_comp(subtask, languages):
    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []

    for lang in languages:
        split = load_and_split_language_for_comp(lang, subtask)

        X_train, y_train = split["train"]
        X_val, y_val     = split["val"]

        # Convert Series -> DataFrame with columns ["text", "lang"]
        X_train = pd.DataFrame({"text": X_train, "lang": lang})
        X_val   = pd.DataFrame({"text": X_val,   "lang": lang})

        X_train_all.append(X_train)
        y_train_all.append(y_train)

        X_val_all.append(X_val)
        y_val_all.append(y_val)

    return {
        "train": (pd.concat(X_train_all, ignore_index=True),
                  pd.concat(y_train_all, ignore_index=True)),
        "val":   (pd.concat(X_val_all, ignore_index=True),
                  pd.concat(y_val_all, ignore_index=True)),
    }


# ------------------------------------------------------------------------------------------------------
# Return the label columns and path for a given subtask and language (Subtask_1 & Subtask_2 & Subtask_3)
# ------------------------------------------------------------------------------------------------------

def _get_label_columns_and_path(subtask, lang):
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
    
    return label_cols, path


# -------------------------------------------------------------------------------------------------------
# Format data for train, validation, and test sets to be compatible with Hugging Face (Subtask_1 & 2 & 3)
# -------------------------------------------------------------------------------------------------------

def format_data_train_val_test(split, tokenizer, subtask):
    X_train, y_train = split["train"]
    X_test, y_test   = split["test"]
    X_val, y_val     = split["val"]

    label_cols = SUBTASK_LABELS[subtask]

    X_train = X_train["text"] if isinstance(X_train, pd.DataFrame) else X_train
    X_val   = X_val["text"]   if isinstance(X_val, pd.DataFrame)   else X_val
    X_test  = X_test["text"]  if isinstance(X_test, pd.DataFrame)  else X_test

    if subtask in (2, 3):
        y_train = y_train[label_cols].astype(np.float32).values
        y_val = y_val[label_cols].astype(np.float32).values
        y_test = y_test[label_cols].astype(np.float32).values

    train_ds = Dataset.from_dict({"text": X_train.tolist(), "labels": y_train.tolist()})
    val_ds   = Dataset.from_dict({"text": X_val.tolist(),   "labels": y_val.tolist()})
    test_ds  = Dataset.from_dict({"text": X_test.tolist(),  "labels": y_test.tolist()})

    def tokenize(batch):
       return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds   = val_ds.map(tokenize, batched=True)
    test_ds  = test_ds.map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format("torch", columns=cols)
    val_ds.set_format("torch", columns=cols)
    test_ds.set_format("torch", columns=cols)

    return train_ds, val_ds, test_ds


# -------------------------------------------------------------------------------------------------
# Format data for train, and validation sets to be compatible with Hugging Face (Subtask_1 & 2 & 3)
# -------------------------------------------------------------------------------------------------

def format_data_train_val(split, tokenizer, subtask):
    X_train, y_train = split["train"]
    X_val, y_val     = split["val"]

    label_cols = SUBTASK_LABELS[subtask]

    X_train = X_train["text"] if isinstance(X_train, pd.DataFrame) else X_train
    X_val   = X_val["text"]   if isinstance(X_val, pd.DataFrame)   else X_val

    if subtask in (2, 3):
        y_train = y_train[label_cols].astype(np.float32).values
        y_val = y_val[label_cols].astype(np.float32).values

    train_ds = Dataset.from_dict({"text": X_train.tolist(), "labels": y_train.tolist()})
    val_ds   = Dataset.from_dict({"text": X_val.tolist(),   "labels": y_val.tolist()})

    def tokenize(batch):
       return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds   = val_ds.map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format("torch", columns=cols)
    val_ds.set_format("torch", columns=cols)

    return train_ds, val_ds


# ------------------------------------------------
# Load dev data for prediction (Subtask_1 & 2 & 3)
# ------------------------------------------------

def load_dev_data_for_prediction(lang, subtask):
    df = pd.read_csv(f"../01_data/dev_phase/subtask{subtask}/dev/{lang}.csv")
    df["lang"] = lang
    return df


# --------------------------------------------------
# Format dev data for prediction (Subtask_1 & 2 & 3)
# --------------------------------------------------

def format_dev_data_for_prediction(df, tokenizer):
    ds = Dataset.from_pandas(df[["id", "text"]], preserve_index=False)

    def tokenize(batch):
       return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds = ds.map(tokenize, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask"])

    return ds


# -------------------------------------------------------------------------
# Load split all languages for competition, no stratify (Subtask_1 & 2 & 3)
# -------------------------------------------------------------------------

def load_split_all_languages_for_comp_no_stratify(subtask, languages):
    label_cols = SUBTASK_LABELS[subtask]

    # 1) alles erst einlesen
    df_all = []
    for lang in languages:
        _, path = _get_label_columns_and_path(subtask, lang)
        df = pd.read_csv(path)

        # text + labels + lang
        df = df[["text", *label_cols]].copy()
        df["lang"] = lang
        df_all.append(df)

    df_all = pd.concat(df_all, ignore_index=True)

    # 2) global splitten
    X = df_all[["text", "lang"]]
    y = df_all[label_cols[0]] if subtask == 1 else df_all[label_cols]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )

    return {
        "train": (X_train.reset_index(drop=True), y_train.reset_index(drop=True)),
        "val":   (X_val.reset_index(drop=True),   y_val.reset_index(drop=True)),
    }