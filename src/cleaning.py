from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils import split_and_normalize_categories, to_pipe_separated

RAW_DATA_PATH = Path("data/raw/netflix_titles.csv")
PROCESSED_DATA_PATH = Path("data/processed/netflix_titles_cleaned.csv")


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Netflix dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    return pd.read_csv(path)


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with explicit defaults and drop empty titles."""
    cleaned = df.copy()

    default_unknown_cols = [
        "director",
        "cast",
        "country",
        "date_added",
        "rating",
        "duration",
        "listed_in",
        "description",
    ]

    for col in default_unknown_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].replace(r"^\s*$", np.nan, regex=True)
            cleaned[col] = cleaned[col].fillna("Unknown")

    if "title" in cleaned.columns:
        cleaned["title"] = cleaned["title"].replace(r"^\s*$", np.nan, regex=True)
        cleaned = cleaned.dropna(subset=["title"])

    return cleaned


def normalize_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize country, rating, and listed_in columns."""
    cleaned = df.copy()

    if "country" in cleaned.columns:
        cleaned["country"] = cleaned["country"].apply(
            lambda x: to_pipe_separated(
                split_and_normalize_categories(x, title_case=True),
                title_case=True,
            )
        )

    if "listed_in" in cleaned.columns:
        cleaned["listed_in"] = cleaned["listed_in"].apply(
            lambda x: to_pipe_separated(split_and_normalize_categories(x))
        )

    if "rating" in cleaned.columns:
        cleaned["rating"] = cleaned["rating"].astype(str).str.strip().str.upper()
        cleaned["rating"] = cleaned["rating"].replace({"": "Unknown", "NAN": "Unknown"})

    return cleaned


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Run full Milestone 1 cleaning + normalization pipeline."""
    rows_before = int(len(df))
    duplicates_removed = int(df.duplicated().sum())

    cleaned = df.drop_duplicates().copy()
    cleaned = clean_missing_values(cleaned)
    cleaned = normalize_categorical_columns(cleaned)

    metrics = {
        "rows_before": rows_before,
        "duplicates_removed": duplicates_removed,
        "rows_after": int(len(cleaned)),
    }
    return cleaned, metrics


def save_dataset(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    """Save processed dataset to data/processed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    raw_df = load_dataset()
    cleaned_df, metrics = prepare_dataset(raw_df)
    save_dataset(cleaned_df)

    print("Milestone 1 preprocessing complete.")
    print(f"Rows before cleaning: {metrics['rows_before']}")
    print(f"Duplicate rows removed: {metrics['duplicates_removed']}")
    print(f"Rows after cleaning: {metrics['rows_after']}")
    print(f"Saved cleaned dataset to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
