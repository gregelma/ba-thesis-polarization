import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)

for l in ['amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin', 'ita', 'khm', 'mya', 'nep', 'ori', 'pan', 'pol', 'rus', 'spa', 'swa', 'tel', 'tur', 'urd', 'zho']:
    # =====================
    # CONFIG
    # =====================
    MODEL_DIR = "results/bert"   # oder dein OUTPUT_DIR
    INPUT_CSV = f"dev_phase/subtask1/dev/{l}.csv"
    OUTPUT_CSV = f"submissions/bert/pred_{l}.csv"
    MAX_LENGTH = 512
    BATCH_SIZE = 16

    # =====================
    # LOAD MODEL & TOKENIZER
    # =====================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # =====================
    # LOAD CSV
    # =====================
    df = pd.read_csv(INPUT_CSV)

    # Sicherheit
    assert "id" in df.columns
    assert "text" in df.columns
    # =====================
    # TO HF DATASET
    # =====================
    dataset = Dataset.from_pandas(df)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["text", "polarization"])
    dataset.set_format("torch")

    # =====================
    # PREDICT
    # =====================
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    predictions = trainer.predict(dataset)

    # Logits → Klassen
    y_pred = np.argmax(predictions.predictions, axis=1)

    # =====================
    # SAVE CSV
    # =====================
    out_df = pd.DataFrame({
        "id": df["id"],
        "polarization": y_pred
    })
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Predictions saved to {OUTPUT_CSV}")