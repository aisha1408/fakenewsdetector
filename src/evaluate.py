"""
Evaluation helpers: metrics, confusion matrix, training curves, baseline comparison.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    CLASSIFICATION_THRESHOLD,
    CONFUSION_MATRIX_PATH,
    LABEL_NAMES,
    METRICS_PATH,
    OUTPUTS_DIR,
    TRAINING_CURVES_PATH,
)
from src.features import SparseToDenseSequence


def _ensure_output_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def predict_proba(model, X_sparse, batch_size: int) -> np.ndarray:
    """Run the Keras model on a sparse TF-IDF matrix in small dense batches."""
    sequence = SparseToDenseSequence(X_sparse, y=None, batch_size=batch_size, shuffle=False)
    probs = model.predict(sequence, verbose=0)
    return probs.ravel()


def scores_from_proba(y_true, y_proba, threshold: float = CLASSIFICATION_THRESHOLD) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "y_pred": y_pred,
    }


def print_report(name: str, y_true, y_pred) -> None:
    target_names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    print(f"\n=== {name} classification report ===")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))


def plot_confusion_matrix(y_true, y_pred, path=CONFUSION_MATRIX_PATH) -> None:
    _ensure_output_dir()
    matrix = confusion_matrix(y_true, y_pred)
    labels = [LABEL_NAMES[0], LABEL_NAMES[1]]
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {path}")


def plot_training_curves(history, path=TRAINING_CURVES_PATH) -> None:
    _ensure_output_dir()
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history_dict["loss"], label="Train loss")
    axes[0].plot(epochs, history_dict["val_loss"], label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, history_dict["accuracy"], label="Train accuracy")
    axes[1].plot(epochs, history_dict["val_accuracy"], label="Val accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle("Training vs validation curves")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves to {path}")


def comparison_table(nn_metrics: dict, lr_metrics: dict) -> str:
    """Pretty text table so the viva demo is easy to read."""
    header = f"{'Model':<28} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    line = "-" * len(header)
    nn_row = (
        f"{'Feedforward NN':<28} "
        f"{nn_metrics['accuracy']:>10.4f} "
        f"{nn_metrics['precision']:>10.4f} "
        f"{nn_metrics['recall']:>10.4f} "
        f"{nn_metrics['f1']:>10.4f}"
    )
    lr_row = (
        f"{'Logistic Regression':<28} "
        f"{lr_metrics['accuracy']:>10.4f} "
        f"{lr_metrics['precision']:>10.4f} "
        f"{lr_metrics['recall']:>10.4f} "
        f"{lr_metrics['f1']:>10.4f}"
    )
    return "\n".join([header, line, nn_row, lr_row])


def save_metrics(payload: dict, path=METRICS_PATH) -> None:
    _ensure_output_dir()
    serializable = {
        key: value
        for key, value in payload.items()
        if key != "y_pred"
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Saved metrics to {path}")
