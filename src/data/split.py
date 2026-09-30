"""split.py — Split the raw data into stratified train / validation / test sets.

Reads:  data/raw/financial_phrasebank.csv
Writes: data/processed/train.csv, val.csv, test.csv
Split:  80% train, 10% validation, 10% test — stratified by label to preserve
        the class balance (0=negative, 1=neutral, 2=positive) in every split.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/financial_phrasebank.csv")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42  # fixed randomness so the split is identical every run (reproducibility)


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows from {RAW_PATH}")

    # Step 1: split off the TRAIN set (80%), leaving 20% as a temporary pool
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=SEED,
    )

    # Step 2: split the 20% pool evenly into VALIDATION (10%) and TEST (10%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED,
    )

    # Save all three
    train_df.to_csv(PROC_DIR / "train.csv", index=False)
    val_df.to_csv(PROC_DIR / "val.csv", index=False)
    test_df.to_csv(PROC_DIR / "test.csv", index=False)

    # Report sizes and confirm the class balance held in each split
    print(f"\nTrain: {len(train_df)}   Val: {len(val_df)}   Test: {len(test_df)}")
    for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = part["label"].value_counts(normalize=True).sort_index().round(3)
        print(f"\n{name} label proportions:\n{dist}")

    print(f"\nSaved splits to: {PROC_DIR.resolve()}")


if __name__ == "__main__":
    main()