from __future__ import annotations

from pathlib import Path
import re
from typing import Tuple

import numpy as np
import pandas as pd

PROCESSED_DATA_PATH = Path("data/processed/netflix_titles_cleaned.csv")
FEATURED_DATA_PATH = Path("data/processed/netflix_titles_featured.csv")


def load_dataset(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned Netflix dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {path}")
    return pd.read_csv(path)


def _parse_duration(duration: str | float | int | None) -> Tuple[float | None, str]:
    """Return (value, unit) extracted from the duration string."""
    if duration is None or (isinstance(duration, float) and np.isnan(duration)):
        return None, "Unknown"

    text = str(duration).strip()
    if not text or text.lower() == "unknown":
        return None, "Unknown"

    match = re.match(r"^(\d+)\s*(min|mins|minute|minutes|season|seasons)$", text, re.I)
    if not match:
        return None, "Unknown"

    value = float(match.group(1))
    unit = match.group(2).lower()
    if "season" in unit:
        return value, "seasons"
    return value, "min"


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived duration and origin features to the dataset."""
    featured = df.copy()

    duration_values = []
    duration_units = []
    for duration in featured.get("duration", []):
        value, unit = _parse_duration(duration)
        duration_values.append(value)
        duration_units.append(unit)

    featured["duration_value"] = duration_values
    featured["duration_unit"] = duration_units

    def length_category(row: pd.Series) -> str:
        value = row.get("duration_value")
        unit = row.get("duration_unit")
        if value is None or pd.isna(value) or unit == "Unknown":
            return "Unknown"
        if unit == "min":
            if value < 60:
                return "Short"
            if value <= 120:
                return "Medium"
            return "Long"
        if unit == "seasons":
            if value <= 1:
                return "Short"
            if value <= 3:
                return "Medium"
            return "Long"
        return "Unknown"

    featured["length_category"] = featured.apply(length_category, axis=1)

    def content_origin(row: pd.Series) -> str:
        release_year = row.get("release_year")
        if pd.isna(release_year):
            return "Unknown"
        added_year = pd.to_datetime(row.get("date_added"), errors="coerce").year
        if pd.isna(added_year):
            return "Unknown"
        return "Original" if int(added_year) == int(release_year) else "Licensed"

    featured["content_origin"] = featured.apply(content_origin, axis=1)

    return featured


def save_dataset(df: pd.DataFrame, path: Path = FEATURED_DATA_PATH) -> None:
    """Save featured dataset to data/processed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    df = load_dataset()
    featured = add_feature_columns(df)
    save_dataset(featured)
    print(f"Saved featured dataset to: {FEATURED_DATA_PATH}")


if __name__ == "__main__":
    main()
