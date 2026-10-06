"""train.py — Fine-tune FinBERT on the tokenized Financial PhraseBank.

Loads the tokenized DatasetDict, fine-tunes ProsusAI/finbert for 3-class
sentiment classification, logs params + metrics to MLflow, and saves the
trained model to models/finbert-finetuned/.
"""

import numpy as np
import mlflow
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------- Config ----------
MODEL_NAME = "ProsusAI/finbert"
DATA_DIR = "data/processed/tokenized"
OUTPUT_DIR = "models/finbert-finetuned"
NUM_LABELS = 3            # negative / neutral / positive
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

LABELS = {0: "negative", 1: "neutral", 2: "positive"}


# ---------- Metrics ----------
def compute_metrics(eval_pred):
    """Called by the Trainer after each epoch to score predictions."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    # 1. Load the tokenized data
    print("Loading tokenized dataset...")
    ds = load_from_disk(DATA_DIR)

    # 2. Load FinBERT with a fresh 3-class classification head
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    # 3. Training configuration
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",   # we log to MLflow manually below
    )

    # 4. The Trainer bundles model + data + args + metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
    )

    # 5. Train, with MLflow logging
    mlflow.set_experiment("finbert-sentiment")
    with mlflow.start_run():
        # log our settings
        mlflow.log_params({
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": 128,
        })

        print("\nTraining...")
        trainer.train()

        # evaluate on validation and log the final metrics
        print("\nEvaluating on validation set...")
        metrics = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        print(metrics)

        # save the fine-tuned model + tokenizer
        trainer.save_model(OUTPUT_DIR)
        print(f"\nModel saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()