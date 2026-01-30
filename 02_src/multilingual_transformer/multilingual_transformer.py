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

from utils.data import load_and_split_all_languages, format_data_train_val_test
from utils.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, SUBTASK_LABELS, SUBTASK_LANGUAGES
from utils.io import save_results, save_training_history, save_details
from monolingual_transformer.monolingual_transformer import find_best_thresholds_per_label
from utils.metrics import compute_detailed_metrics, compute_general_metrics


def run_multilingual_transformer(model_name, subtask, seed, languages):
    set_seed(seed)

    isMultilabel = subtask in (2, 3)
    label_cols = SUBTASK_LABELS[subtask]

    split = load_and_split_all_languages(subtask=subtask, languages=languages)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds, val_ds, test_ds = format_data_train_val_test(split, tokenizer, subtask=subtask)

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
        output_dir=f"../05_models/subtask{subtask}_{model_name}_{seed}",
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

    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)

    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")

    logits = trainer.predict(test_ds).predictions
    X_test, y_test = split["test"]
    test_langs = X_test["lang"].to_numpy()
    
    best_t = None
    if isMultilabel:
        _, y_val = split["val"]
        best_t = find_best_thresholds_per_label(trainer, val_ds, y_val, label_cols)
        print("Best thresholds:", best_t)
        
        probs = torch.sigmoid(torch.from_numpy(logits))
        y_pred = (probs >= torch.from_numpy(best_t)).int().numpy()
    else:
        y_pred = np.argmax(logits, axis=1)
    

    accuracy_by_lang = {}
    f1_macro_by_lang = {}

    for lang in languages:
        idx = (test_langs == lang)

        y_test_lang = y_test.loc[idx]
        y_pred_lang = y_pred[idx]

        details = compute_detailed_metrics(y_test_lang, y_pred_lang, subtask=subtask)
        save_details(f"{model_name}_{seed}", details, lang, subtask=subtask)

        general_metrics = compute_general_metrics(y_test_lang, y_pred_lang)
        accuracy_by_lang[lang] = general_metrics["accuracy"]
        f1_macro_by_lang[lang] = general_metrics["f1_macro"]

    print("Accuracy by lang:", accuracy_by_lang)
    print("F1 macro by lang:", f1_macro_by_lang)

    save_results(f"{model_name}_{seed}", accuracy_by_lang, "accuracy", subtask=subtask)
    save_results(f"{model_name}_{seed}", f1_macro_by_lang, "f1_macro", subtask=subtask)

    save_training_history(f"{model_name}_{seed}", trainer, subtask=subtask, thresholds=best_t)


if __name__ == "__main__":
    seeds = [1, 2, 3]
    subtasks = [1, 2, 3]
    
    for subtask in subtasks:
        for model_name in ["bert-base-multilingual-cased", "xlm-roberta-base", "microsoft/mdeberta-v3-base"]:
            for seed in seeds:
                run_multilingual_transformer(
                    model_name=model_name,
                    subtask=subtask,
                    seed=seed,
                    languages=SUBTASK_LANGUAGES[subtask]
                )