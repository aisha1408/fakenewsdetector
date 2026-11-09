"""
Load the saved model + TF-IDF vectorizer and classify new text.

Python:
    from src.predict import predict
    print(predict("Some news article ..."))

CLI (from the project root):
    python -m src.predict "Paste the article text here"
"""

from __future__ import annotations

import argparse
import os
import sys

# Configure backend fallback if tensorflow is not installed
if "KERAS_BACKEND" not in os.environ:
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        os.environ["KERAS_BACKEND"] = "torch"

try:
    import keras
except ImportError:
    from tensorflow import keras

import joblib
import numpy as np

from src.config import (
    CLASSIFICATION_THRESHOLD,
    LABEL_FAKE,
    LABEL_NAMES,
    MODEL_PATH,
    VECTORIZER_PATH,
)
from src.preprocess import clean_text

_model = None
_vectorizer = None


def _missing_artifacts_message() -> str:
    return (
        "Trained files were not found.\n"
        f"  Expected model:      {MODEL_PATH}\n"
        f"  Expected vectorizer: {VECTORIZER_PATH}\n"
        "Train first with:  python -m src.train"
    )


def load_artifacts():
    """Load model and vectorizer once, then reuse them (faster for the app)."""
    global _model, _vectorizer
    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(_missing_artifacts_message())

    _model = keras.models.load_model(MODEL_PATH)
    _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def predict(text: str, threshold: float = CLASSIFICATION_THRESHOLD) -> dict:
    """
    Classify one article.

    Returns:
        label: "Real" or "Fake"
        confidence: probability of the predicted class, 0 to 1
        fake_probability: model output P(fake)
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide a non-empty news article.")

    model, vectorizer = load_artifacts()
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("The text was empty after cleaning. Try a longer article.")

    features = vectorizer.transform([cleaned]).toarray().astype(np.float32)
    fake_probability = float(model.predict(features, verbose=0)[0][0])
    is_fake = fake_probability >= threshold
    label_id = LABEL_FAKE if is_fake else 0
    confidence = fake_probability if is_fake else (1.0 - fake_probability)

    return {
        "label": LABEL_NAMES[label_id],
        "confidence": confidence,
        "fake_probability": fake_probability,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify a news article as Real or Fake."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Article text. If omitted, the program reads from stdin.",
    )
    args = parser.parse_args(argv)

    text = args.text if args.text else sys.stdin.read()
    try:
        result = predict(text)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Prediction:        {result['label']}")
    print(f"Confidence:        {result['confidence'] * 100:.2f}%")
    print(f"Fake probability:  {result['fake_probability']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
