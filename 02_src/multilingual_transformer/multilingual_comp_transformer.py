from sklearn.metrics import f1_score
from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        set_seed,
    )
import numpy as np
import torch

from utils.data import load_split_all_languages_for_comp_no_stratify, format_data_train_val
from utils.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, SUBTASK_LABELS, SUBTASK_LANGUAGES
from monolingual_transformer.monolingual_transformer import find_best_thresholds_per_label
from utils.predict import predict_and_save_dev


def run_multilingual_comp_transformer(model_name, subtask, seed):
    set_seed(seed)

    isMultilabel = subtask in (2, 3)
    label_cols = SUBTASK_LABELS[subtask]

    split = load_split_all_languages_for_comp_no_stratify(subtask=subtask, languages=SUBTASK_LANGUAGES[subtask])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds, val_ds = format_data_train_val(split, tokenizer, subtask=subtask)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label_cols) if isMultilabel else 2, 
        problem_type="multi_label_classification" if isMultilabel else "single_label_classification"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        if isMultilabel:
            probs = torch.sigmoid(torch.from_numpy(logits))
            preds = (probs > 0.5).int().numpy()
            return {'f1_macro': f1_score(labels, preds, average='macro')}
        else:
            preds = np.argmax(logits, axis=1)
            return {'f1_macro': f1_score(labels, preds, average='macro')}
    
    args = TrainingArguments(
        output_dir=f"../05_models/comp_subtask{subtask}_{model_name.replace('/', '_')}_{seed}_no_s_512",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=128,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=True,
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    val_metrics  = trainer.evaluate()

    print(f"Validation metrics: {val_metrics}")
    
    best_t = None
    _, y_val = split["val"]
    if isMultilabel:
        _, y_val = split["val"]
        best_t = find_best_thresholds_per_label(trainer, val_ds, y_val, label_cols)
        print("Best thresholds:", best_t)
    
    predict_and_save_dev(args.output_dir, subtask, out_dir=f"../06_submissions/comp_subtask{subtask}_{model_name.replace('/', '_')}_{seed}_no_s_512", thresholds=best_t)   


# if __name__ == "__main__":
#     seeds = [1,2,3]

#     for seed in seeds:
#         run_multilingual_comp_transformer(
#             model_name="microsoft/mdeberta-v3-base",
#             subtask=1,
#             seed=seed
#         )