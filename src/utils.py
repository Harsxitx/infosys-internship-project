from __future__ import annotations

from typing import Iterable, List


def normalize_text(value: str | None, *, title_case: bool = False) -> str:
    """Return a consistently formatted string value."""
    if value is None:
        return "Unknown"

    cleaned = " ".join(str(value).strip().split())
    if not cleaned:
        return "Unknown"
    return cleaned.title() if title_case else cleaned


def split_and_normalize_categories(
    value: str | None, *, title_case: bool = False
) -> List[str]:
    """
    Split comma-separated categorical values and normalize each token.
    Example: "United States,  india " -> ["India", "United States"]
    """
    if value is None:
        return ["Unknown"]

    parts = [normalize_text(item, title_case=title_case) for item in str(value).split(",")]
    cleaned = sorted({item for item in parts if item and item != "Unknown"})
    return cleaned if cleaned else ["Unknown"]


def to_pipe_separated(values: Iterable[str], *, title_case: bool = False) -> str:
    """Persist multi-valued categories in a compact, deterministic format."""
    cleaned = [normalize_text(v, title_case=title_case) for v in values if v is not None]
    unique_values = sorted({v for v in cleaned if v and v != "Unknown"})
    return "|".join(unique_values) if unique_values else "Unknown"
