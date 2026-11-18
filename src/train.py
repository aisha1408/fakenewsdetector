"""
Full training pipeline.

Run from the project root:
    python -m src.train

Steps:
  1. Load and shuffle the Kaggle CSVs
  2. Clean the text (lowercase, URLs, HTML, punctuation, numbers, stopwords, lemmatization)
  3. Stratified 70 / 15 / 15 split
  4. Fit TF-IDF on train only
  5. Train the neural network with early stopping
  6. Evaluate on the test set
  7. Train a Logistic Regression baseline on the same features
  8. Save model, vectorizer, plots, and metrics.json
"""

from __future__ import annotations

import os
import random

# Configure backend fallback if tensorflow is not installed
if "KERAS_BACKEND" not in os.environ:
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        os.environ["KERAS_BACKEND"] = "torch"

try:
    import keras
    from keras.callbacks import EarlyStopping
except ImportError:
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FAST_WEIGHTS_PATH,
    LABEL_FAKE,
    LABEL_REAL,
    LR_C,
    LR_MAX_ITER,
    MODELS_DIR,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    VECTORIZER_PATH,
)
from src.data_loader import load_dataset, split_xy
from src.evaluate import (
    comparison_table,
    plot_confusion_matrix,
    plot_training_curves,
    predict_proba,
    print_report,
    save_metrics,
    scores_from_proba,
)
from src.features import (
    SparseToDenseSequence,
    build_vectorizer,
    fit_transform_train,
    transform,
)
from src.model import build_model
from src.preprocess import clean_corpus


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Make runs as reproducible as possible on a given machine."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def stratified_splits(X, y):
    """
    70% train, 15% validation, 15% test, preserving the fake/real ratio.

    sklearn only splits into two parts at a time, so we:
      - first peel off the test set (15%)
      - then peel validation off the remainder so that val is 15% of the original
    """
    if abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) > 1e-6:
        raise ValueError("TRAIN_SIZE + VAL_SIZE + TEST_SIZE must equal 1.0")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    # val fraction of the remaining (train+val) data
    val_fraction_of_rest = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction_of_rest,
        stratify=y_train_val,
        random_state=RANDOM_SEED,
    )
    print(
        f"Split sizes: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}"
    )
    print(
        f"  Train fake/real: {(y_train == LABEL_FAKE).sum():,}/{(y_train == LABEL_REAL).sum():,}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train() -> None:
    set_seeds()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/8] Loading dataset...")
    data = load_dataset()
    X_raw, y = split_xy(data)

    print("\n[2/8] Preprocessing text...")
    X_clean = np.array(clean_corpus(X_raw), dtype=object)

    print("\n[3/8] Splitting 70/15/15 (stratified)...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(X_clean, y)

    print("\n[4/8] Building TF-IDF features...")
    vectorizer = build_vectorizer()
    X_train_tfidf = fit_transform_train(vectorizer, X_train)
    X_val_tfidf = transform(vectorizer, X_val)
    X_test_tfidf = transform(vectorizer, X_test)
    print(f"  Val TF-IDF shape:  {X_val_tfidf.shape}")
    print(f"  Test TF-IDF shape: {X_test_tfidf.shape}")

    print("\n[5/8] Training the neural network...")
    model = build_model(input_dim=X_train_tfidf.shape[1])
    model.summary()

    train_seq = SparseToDenseSequence(
        X_train_tfidf, y_train, batch_size=BATCH_SIZE, shuffle=True
    )
    val_seq = SparseToDenseSequence(
        X_val_tfidf, y_val, batch_size=BATCH_SIZE, shuffle=False
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1,
    )

    print("\n[6/8] Saving model and vectorizer...")
    model.save(MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    np.savez_compressed(FAST_WEIGHTS_PATH, *model.get_weights())
    print(f"  Model:        {MODEL_PATH}")
    print(f"  Fast weights: {FAST_WEIGHTS_PATH}")
    print(f"  Vectorizer:   {VECTORIZER_PATH}")

    print("\n[7/8] Evaluating the neural network on the test set...")
    plot_training_curves(history)
    nn_proba = predict_proba(model, X_test_tfidf, BATCH_SIZE)
    nn_scores = scores_from_proba(y_test, nn_proba)
    print_report("Neural network", y_test, nn_scores["y_pred"])
    plot_confusion_matrix(y_test, nn_scores["y_pred"])

    print("\n[8/8] Training Logistic Regression baseline on the same TF-IDF features...")
    lr = LogisticRegression(
        C=LR_C,
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_SEED,
        solver="liblinear",
    )
    lr.fit(X_train_tfidf, y_train)
    lr_proba = lr.predict_proba(X_test_tfidf)[:, 1]
    lr_scores = scores_from_proba(y_test, lr_proba)
    print_report("Logistic Regression", y_test, lr_scores["y_pred"])

    table = comparison_table(nn_scores, lr_scores)
    print("\n=== Test-set comparison ===")
    print(table)

    save_metrics(
        {
            "neural_network": {
                "accuracy": nn_scores["accuracy"],
                "precision": nn_scores["precision"],
                "recall": nn_scores["recall"],
                "f1": nn_scores["f1"],
            },
            "logistic_regression": {
                "accuracy": lr_scores["accuracy"],
                "precision": lr_scores["precision"],
                "recall": lr_scores["recall"],
                "f1": lr_scores["f1"],
            },
            "split": {
                "train": int(len(X_train)),
                "val": int(len(X_val)),
                "test": int(len(X_test)),
            },
            "notes": (
                "Precision/recall/F1 are reported for the positive class (fake = 1). "
                "Very high scores on this Kaggle set can still reflect source style, "
                "not genuine 'truth' detection."
            ),
        }
    )
    print("\nTraining complete.")


if __name__ == "__main__":
    train()
