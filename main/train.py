import os
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback
)
from data import load_dataset, tokenize_dataset
from metrics import compute_language_macro_metrics
from config import (
    MODEL_NAME,
    OUTPUT_DIR,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    CSV_LOG_FILE
)

# class WeightedTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.get("labels")
#         outputs = model(**inputs)
#         logits = outputs.get('logits')

#         # Beispiel: class weights
#         class_weights = torch.tensor([1.0671, 0.9409]).to(logits.device)
#         loss_fct = nn.CrossEntropyLoss(weight=class_weights)

#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

class LogMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else None
        if metrics is not None and epoch is not None:
            # Die Werte direkt in CSV schreiben
            data = { 
                "epoch": [epoch],
                "accuracy": [metrics.get("accuracy")],
                "macro_f1_lang_balanced": [metrics.get("macro_f1_lang_balanced")],
                "loss": [metrics.get("eval_loss")]
            }
            for key in metrics:
                    if key.startswith("macro_f1_") and key != "macro_f1_lang_balanced":
                        data[key] = [metrics.get(key)]
            df = pd.DataFrame(data)

            # Wenn Datei existiert, anhängen, sonst neu erstellen
            if os.path.exists(CSV_LOG_FILE):
                df.to_csv(CSV_LOG_FILE, mode='a', header=False, index=False)
            else:
                df.to_csv(CSV_LOG_FILE, mode='w', header=True, index=False)
            

def main():
    dataset = load_dataset()
    dataset, tokenizer = tokenize_dataset(dataset)

    # labels = np.array(dataset["train"]["labels"])
    # class_weights = compute_class_weight(
    #     class_weight="balanced",
    #     classes=np.unique(labels),
    #     y=labels
    # )
    # class_weights = torch.tensor(class_weights, dtype=torch.float)

    num_labels = 2 
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label={0: "NOT_POLARIZED", 1: "POLARIZED"},
        label2id={"NOT_POLARIZED": 0, "POLARIZED": 1},
         ######################################### problem_type="single_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1_lang_balanced",
        greater_is_better=True,
        # logging_dir=f"{OUTPUT_DIR}/logs",
        report_to="wandb",             ###### none
        fp16=True,
        dataloader_num_workers=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda eval_pred: compute_language_macro_metrics(eval_pred, dataset["validation"]["language"]),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), LogMetricsCallback]
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
