from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        DataCollatorWithPadding,
    )
import pandas as pd
import numpy as np
import os
import torch

from utils.config import SUBTASK_LANGUAGES, SUBTASK_LABELS
from utils.data import load_dev_data_for_prediction, format_dev_data_for_prediction


def predict_and_save_dev(model_dir, subtask, out_dir, thresholds=None):
    isMultilabel = subtask in (2, 3)

    # --- load model/tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    os.makedirs(out_dir, exist_ok=True)

    for lang in SUBTASK_LANGUAGES[subtask]:
        output_csv = os.path.join(out_dir, f"pred_{lang}.csv")

        df = load_dev_data_for_prediction(lang, subtask)
        ds = format_dev_data_for_prediction(df, tokenizer)

        preds = trainer.predict(ds).predictions  # logits

        if not isMultilabel:
            # subtask 1:
            y_pred = np.argmax(preds, axis=1).astype(int)

            out_df = pd.DataFrame({
                "id": df["id"].values,
                "polarization": y_pred,
            })
        else:
            # subtask 2/3: multilabel
            probs = torch.sigmoid(torch.from_numpy(preds))

            if thresholds is None:
                y_pred = (probs >= 0.5).int().numpy()
            else:
                y_pred = (probs >= torch.from_numpy(np.asarray(thresholds))).int().numpy()

            out_df = pd.DataFrame(y_pred, columns=SUBTASK_LABELS[subtask])
            out_df.insert(0, "id", df["id"].values)

        out_df.to_csv(output_csv, index=False)