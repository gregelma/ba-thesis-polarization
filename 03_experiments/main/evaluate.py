from transformers import Trainer
from data import load_dataset, tokenize_dataset
from metrics import compute_metrics
from config import OUTPUT_DIR


def main():
    dataset = load_dataset()
    dataset, _ = tokenize_dataset(dataset)

    trainer = Trainer(
        model=None,
        compute_metrics=compute_metrics
    )

    metrics = trainer.evaluate(
        eval_dataset=dataset["test"],
        metric_key_prefix="test"
    )

    print(metrics)


if __name__ == "__main__":
    main()
