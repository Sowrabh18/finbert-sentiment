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
    missing = clean["label"].isna().sum()
    if missing > 0:
        raise ValueError(f"{missing} rows had a label we couldn't map — check the 'answer' values.")
    clean = clean[clean["sentence"].str.len() > 0].reset_index(drop=True)
    clean["label"] = clean["label"].astype(int)

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