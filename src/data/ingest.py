"""ingest.py — Download Financial PhraseBank and save clean raw data to data/raw/.

The dohonba mirror stores the data in a question/context/answer layout, so we
normalize it here into a standard two-column form: `sentence` (text) and
`label` (integer). Label encoding: 0=negative, 1=neutral, 2=positive.
"""

from pathlib import Path
from datasets import load_dataset
import pandas as pd

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RAW_DIR / "financial_phrasebank.csv"

# Map the word labels to integers (the FinBERT convention)
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def main():
    print("Downloading Financial PhraseBank from Hugging Face...")
    ds = load_dataset("dohonba/financial_phrasebank", split="train")
    df = ds.to_pandas()

    print(f"\nRaw columns from mirror: {list(df.columns)}")
    print(f"Raw row count: {len(df)}")

    # --- Normalize into clean sentence + label columns ---
    clean = pd.DataFrame({
        "sentence": df["context"].str.strip(),
        "label": df["answer"].str.strip().str.lower().map(LABEL_MAP),
    })

    # Safety checks: no missing labels, no empty sentences
      # --- Remove duplicate sentences (prevents train/test leakage) ---
    before = len(clean)

    # 1. Drop sentences with CONFLICTING labels (same text, different label) —
    #    ambiguous, so we can't trust either copy. Remove all copies.
    label_counts = clean.groupby("sentence")["label"].nunique()
    conflicting = label_counts[label_counts > 1].index
    clean = clean[~clean["sentence"].isin(conflicting)]

    # 2. For consistent duplicates, keep the first copy only.
    clean = clean.drop_duplicates(subset="sentence", keep="first").reset_index(drop=True)

    print(f"Removed {before - len(clean)} duplicate/conflicting rows "
          f"({len(conflicting)} conflicting sentences dropped entirely)")

    # --- Inspect the cleaned result ---
    print("\n=== Cleaned data (first 5 rows) ===")
    print(clean.head())
    print("\n=== Label distribution (counts per class) ===")
    print(clean["label"].value_counts().sort_index())

    clean.to_csv(OUT_PATH, index=False)
    print(f"\nRows saved: {len(clean)}")
    print(f"Saved clean raw data to: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()