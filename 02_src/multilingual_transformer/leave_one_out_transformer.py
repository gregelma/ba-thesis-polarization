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
import os

from utils.data import load_and_split_all_languages, load_and_split_language, format_data_train_val_test
from utils.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, SUBTASK_LABELS, SUBTASK_LANGUAGES
from utils.metrics import compute_detailed_metrics, compute_general_metrics
from utils.io import save_details, save_results, save_training_history
from monolingual_transformer.monolingual_transformer import find_best_thresholds_per_label


def run_leave_one_out_transformer(model_name, lang, subtask, seed):
    set_seed(seed)

    languages = [l for l in SUBTASK_LANGUAGES[subtask] if l != lang]

    isMultilabel = subtask in (2, 3)
    label_cols = SUBTASK_LABELS[subtask]

    train_val_split = load_and_split_all_languages(subtask=subtask, languages=languages)
    test_split = load_and_split_language(lang, subtask=subtask)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds, val_ds, _ = format_data_train_val_test(train_val_split, tokenizer, subtask=subtask)
    _, _, test_ds = format_data_train_val_test(test_split, tokenizer, subtask=subtask)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label_cols) if isMultilabel else 2, 
        problem_type="multi_label_classification" if isMultilabel else "single_label_classification"
    )

    model_name = model_name.replace('/', '_')

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
        output_dir=f"../05_models/subtask{subtask}_leave_{lang}_out_{model_name}_{seed}",
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
    test_metrics = trainer.evaluate(test_ds)

    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")

    logits = trainer.predict(test_ds).predictions
    _, y_test = test_split["test"]
    
    best_t = None
    if isMultilabel:
        _, y_val = train_val_split["val"]
        best_t = find_best_thresholds_per_label(trainer, val_ds, y_val, label_cols)
        print("Best thresholds:", best_t)
        
        probs = torch.sigmoid(torch.from_numpy(logits))
        y_pred = (probs >= torch.from_numpy(best_t)).int().numpy()
    else:
        y_pred = np.argmax(logits, axis=1)

    details = compute_detailed_metrics(y_test, y_pred, subtask=subtask)
    general_metrics = compute_general_metrics(y_test, y_pred)
    accuracy_by_lang = {lang: general_metrics["accuracy"]}
    f1_macro_by_lang = {lang: general_metrics["f1_macro"]}

    print(f"Detailed test metrics: {details}")
    print(f"General test metrics: {general_metrics}")

    save_details(f"leave_{lang}_out_{model_name}_{seed}", details, lang, subtask=subtask)

    save_results(f"leave_{lang}_out_{model_name}_{seed}", accuracy_by_lang, "accuracy", subtask=subtask)
    save_results(f"leave_{lang}_out_{model_name}_{seed}", f1_macro_by_lang, "f1_macro", subtask=subtask)

    save_training_history(f"leave_{lang}_out_{model_name}_{seed}", trainer, subtask=subtask, thresholds=best_t)


if __name__ == "__main__":
    seeds = [1,2,3]
    subtasks = [1,2,3]

    for model_name in ["bert-base-multilingual-cased", "xlm-roberta-base", "microsoft/mdeberta-v3-base"]:
        for subtask in subtasks:
            for lang in ['amh', 'arb', 'deu', 'eng', 'swa', 'zho']:
                for seed in seeds:
                    print(f"Running leave-one-out for lang={lang}, subtask={subtask}, seed={seed}")
                    run_leave_one_out_transformer(
                        model_name=model_name,
                        lang=lang,
                        subtask=subtask,
                        seed=seed
                    )