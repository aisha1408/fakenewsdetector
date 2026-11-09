"""
Load Fake.csv and True.csv, add labels, merge, shuffle, combine title + text.

The Kaggle "Fake and Real News Dataset" stores real articles in True.csv
and fake articles in Fake.csv. We convert these into numeric labels:
  - 0 = Real
  - 1 = Fake

Articles from True.csv originally come from Reuters wire service and frequently
start with location / agency tags such as:
  "WASHINGTON (Reuters) - The White House said..."
We strip these lead-ins and any other Reuters source identifiers so the model
learns from actual linguistic and topical content rather than memorizing the wire tag.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FAKE_CSV,
    LABEL_FAKE,
    LABEL_REAL,
    RANDOM_SEED,
    TEXT_COLUMNS,
    TRUE_CSV,
)

# Regex patterns for removing source leakage
# 1. Lead-in tag at the very start of the article body:
#    e.g., "WASHINGTON (Reuters) - ", "WEST PALM BEACH, Fla./WASHINGTON (Reuters) - "
_REUTERS_LEADIN = re.compile(
    r"^[A-Z][A-Za-z\s\.\,\-\/'\&]*\(\s*Reuters\s*\)\s*[-–—:]\s*",
    flags=re.IGNORECASE,
)
# 2. Any other Reuters tags or mentions
_REUTERS_TAG = re.compile(r"\(\s*Reuters\s*\)", flags=re.IGNORECASE)
_REUTERS_WORD = re.compile(r"\bReuters\b", flags=re.IGNORECASE)

# 3. Common blog / fake news artifact at the end of articles
_FEATURED_IMAGE = re.compile(r"featured\s+image.*$", flags=re.IGNORECASE)


def _require_dataset_files() -> None:
    missing = [str(path) for path in (FAKE_CSV, TRUE_CSV) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset files not found:\n  - "
            + "\n  - ".join(missing)
            + "\n\nDownload the Kaggle dataset "
            "'Fake and Real News Dataset' (clmentbisaillon/fake-and-real-news-dataset)\n"
            "and place Fake.csv and True.csv inside the data/ folder."
        )


def strip_source_leakage(text: str) -> str:
    """
    Remove Reuters lead-ins, source tags, and blog image artifacts.
    This prevents the model from trivially classifying based on agency bylines.
    """
    if not isinstance(text, str):
        return ""
    cleaned = _REUTERS_LEADIN.sub("", text.strip())
    cleaned = _REUTERS_TAG.sub("", cleaned)
    cleaned = _REUTERS_WORD.sub("", cleaned)
    cleaned = _FEATURED_IMAGE.sub("", cleaned)
    return cleaned.strip()


def _prepare_split(csv_path: Path, label: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for column in TEXT_COLUMNS:
        if column not in df.columns:
            raise ValueError(
                f"{csv_path.name} is missing column '{column}'. "
                f"Found columns: {list(df.columns)}"
            )

    df = df[list(TEXT_COLUMNS)].copy()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    # Strip source leakage from the body text before concatenating with title
    cleaned_body = df["text"].map(strip_source_leakage)

    # Combine headline + cleaned body into one document
    df["content"] = (df["title"] + " " + cleaned_body).str.strip()
    df["label"] = label

    # Keep only articles with non-empty content
    df = df[df["content"].str.len() > 0]
    return df[["content", "label"]]


def load_dataset() -> pd.DataFrame:
    """
    Load, label, merge and shuffle the two CSV files.

    Returns a DataFrame with:
      - content: cleaned title + article body
      - label: 0 (real) or 1 (fake)
    """
    _require_dataset_files()

    fake_df = _prepare_split(FAKE_CSV, LABEL_FAKE)
    true_df = _prepare_split(TRUE_CSV, LABEL_REAL)

    data = pd.concat([fake_df, true_df], ignore_index=True)
    data = data.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    print(
        f"Loaded {len(data):,} articles "
        f"({(data['label'] == LABEL_FAKE).sum():,} fake, "
        f"{(data['label'] == LABEL_REAL).sum():,} real)"
    )
    return data


def split_xy(data: pd.DataFrame):
    """Return numpy arrays X (text) and y (labels)."""
    X = data["content"].to_numpy()
    y = data["label"].to_numpy(dtype=np.int32)
    return X, y
