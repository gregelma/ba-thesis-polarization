import shutil
import os
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
    
    X_val_all, y_val_all = split["val"]
    scores = {}
    for lang in languages:
        mask = X_val_all["lang"] == lang
        val_lang = (
            X_val_all.loc[mask].reset_index(drop=True),
            y_val_all.loc[mask].reset_index(drop=True),
        )

        val_lang = format_data_train_val_test({"train": split["train"], "val": val_lang, "test": split["test"]},
                                              tokenizer, subtask=subtask)[1]  # take val_ds only

        pred = trainer.predict(val_lang)
        scores[lang] = float(compute_metrics((pred.predictions, pred.label_ids))["f1_macro"])

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    return scores


if __name__ == "__main__":
    import json
    import statistics

    seeds = [1, 2, 3]
    subtask = 1
    min_delta = 0.002
    model = "microsoft/mdeberta-v3-base"
    safe_model = model.replace("/", "_")

    # baseline scores (single-language training)
    with open(f"../04_results/val_results/{safe_model}.json", "r") as f:
        base = json.load(f)

    langs = list(base.keys())

    # init output structure
    out = {}
    for lang in langs:
        out[lang] = {
            "baseline": base[lang],
            "helpful_languages": [],
            "pair_scores": {},  # cand -> avg_target_score
        }

    # only run each unordered pair once
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            a = langs[i]
            b = langs[j]

            # collect per-seed scores for BOTH languages from ONE run
            a_scores = []
            b_scores = []

            for seed in seeds:
                scores = run_multilingual_transformer(model, subtask, seed, [a, b])
                a_scores.append(scores[a])
                b_scores.append(scores[b])

            avg_a = sum(a_scores) / len(a_scores)
            std_a = statistics.stdev(a_scores)
            avg_b = sum(b_scores) / len(b_scores)
            std_b = statistics.stdev(b_scores)

            # store pair scores for both directions
            out[a]["pair_scores"][b] = avg_a
            out[b]["pair_scores"][a] = avg_b

            # decide helpfulness for a (does b help a?)
            if avg_a > base[a] + max(min_delta, 0.5 * std_a):
                out[a]["helpful_languages"].append(b)
                print(f"Adding {b} as helpful for {a}: {avg_a:.4f} > {base[a]:.4f} + {max(min_delta, 0.5 * std_a):.4f}")
            else:
                print(f"Not adding {b} for {a}: {avg_a:.4f} <= {base[a]:.4f} + {max(min_delta, 0.5 * std_a):.4f}")
            # decide helpfulness for b (does a help b?)
            if avg_b > base[b] + max(min_delta, 0.5 * std_b):
                out[b]["helpful_languages"].append(a)
                print(f"Adding {a} as helpful for {b}: {avg_b:.4f} > {base[b]:.4f} + {max(min_delta, 0.5 * std_b):.4f}")
            else:
                print(f"Not adding {a} for {b}: {avg_b:.4f} <= {base[b]:.4f} + {max(min_delta, 0.5 * std_b):.4f}")
            
            print("helpful_languages so far:", {lang: out[lang]["helpful_languages"] for lang in langs})

    with open("../04_results/val_results/helpful_languages.json", "w") as f:
        json.dump(out, f, indent=2)

    print("saved -> ../04_results/val_results/helpful_languages.json")