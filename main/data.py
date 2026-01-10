import pandas as pd
import os
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH, DATA_PATH, RANDOM_SEED

def load_dataset(): 
    df = load_multilingual_data(DATA_PATH)

    df.rename(columns={"polarization": "labels"}, inplace=True) 
    df["labels"] = df["labels"].astype(int) 

    dataset = Dataset.from_pandas(df)

    dataset = dataset.train_test_split( test_size=0.2, seed=RANDOM_SEED )      
    tmp = dataset["test"].train_test_split( test_size=0.5, seed=RANDOM_SEED ) 
    
    dataset["validation"] = tmp["train"] 
    dataset["test"] = tmp["test"] 
    
    return dataset

# def load_dataset():
#     df = load_multilingual_data(DATA_PATH)
#     df.rename(columns={"polarization": "labels"}, inplace=True)
#     df["labels"] = df["labels"].astype(int)

    
#     df["stratify_col"] = df["language"] + "_" + df["labels"].astype(str)

#     # Train / Test+Validation Split (80/20)
#     train_df, test_val_df = train_test_split(
#         df,
#         test_size=0.2,
#         random_state=RANDOM_SEED,
#         stratify=df["stratify_col"]  # stratify nach Sprache+Klasse
#     )

#     # Test / Validation Split (50/50 vom 20%)
#     val_df, test_df = train_test_split(
#         test_val_df,
#         test_size=0.5,
#         random_state=RANDOM_SEED,
#         stratify=test_val_df["stratify_col"]
#     )

#     # Die Hilfsspalte können wir danach wieder entfernen
#     train_df = train_df.drop(columns=["stratify_col"])
#     val_df = val_df.drop(columns=["stratify_col"])
#     test_df = test_df.drop(columns=["stratify_col"])

#     dataset = DatasetDict({
#         "train": Dataset.from_pandas(train_df),
#         "validation": Dataset.from_pandas(val_df),
#         "test": Dataset.from_pandas(test_df)
#     })

#     return dataset


def tokenize_dataset(dataset):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    dataset = dataset.map(tokenize, batched=True)
    #dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")

    return dataset, tokenizer

def load_multilingual_data(
    data_dir: str,
    text_col: str = "text",
    label_col: str = "polarization"
) -> pd.DataFrame:

    dataframes = []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".csv"):
            continue

        lang = os.path.splitext(filename)[0]
        filepath = os.path.join(data_dir, filename)

        df = pd.read_csv(filepath)

        # Basic sanity checks
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(
                f"{filename} must contain columns "
                f"'{text_col}' and '{label_col}'"
            )

        df = df[[text_col, label_col]].copy()
        df["language"] = lang

        dataframes.append(df)

    if not dataframes:
        raise RuntimeError(f"No CSV files found in {data_dir}")

    multilingual_df = pd.concat(dataframes, ignore_index=True)

    return multilingual_df