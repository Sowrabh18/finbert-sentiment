"""tokenize.py — Tokenize the train/val/test splits for FinBERT.

Reads:  data/processed/{train,val,test}.csv
Writes: data/processed/tokenized/  (a Hugging Face DatasetDict on disk)

Converts each sentence into input_ids + attention_mask (max_length=128,
derived from EDA: 99th percentile token length was 68).
"""

from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

PROC_DIR = Path("data/processed")
OUT_DIR = PROC_DIR / "tokenized"

MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 128  # from EDA: covers 99.98% of sentences with no truncation


def main():
    # Load the FinBERT tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the three CSV splits into a DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_pandas(pd.read_csv(PROC_DIR / "train.csv")),
        "validation": Dataset.from_pandas(pd.read_csv(PROC_DIR / "val.csv")),
        "test": Dataset.from_pandas(pd.read_csv(PROC_DIR / "test.csv")),
    })
    print("\nLoaded splits:")
    print(dataset)

    # The function applied to every row to tokenize its sentence
    def tokenize_batch(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    # Apply tokenization to all splits at once
    print("\nTokenizing...")
    tokenized = dataset.map(tokenize_batch, batched=True)

    # Keep only the columns the model needs; drop the raw text
    tokenized = tokenized.remove_columns(["sentence"])

    # Save to disk
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(OUT_DIR))
    print(f"\nSaved tokenized dataset to: {OUT_DIR.resolve()}")
    print("\nFinal structure:")
    print(tokenized)


if __name__ == "__main__":
    main()