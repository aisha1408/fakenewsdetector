"""
TF-IDF feature extraction.

TF-IDF (Term Frequency – Inverse Document Frequency) turns each article into
a numeric vector. Words that are common in one article but rare across the
whole training set get a higher score.

Important: the vectorizer is fitted on the TRAINING set only. If we fitted on
all data, information from the test set would leak into training.
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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

from src.config import BATCH_SIZE, MAX_DF, MAX_FEATURES, MIN_DF, NGRAM_RANGE

SequenceBase = getattr(keras.utils, "PyDataset", getattr(keras.utils, "Sequence", object))


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        # Text is already cleaned; do not re-tokenize aggressively
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )


def fit_transform_train(vectorizer: TfidfVectorizer, train_texts):
    """Fit TF-IDF on train texts and return a sparse matrix."""
    print("Fitting TF-IDF on the training set only...")
    X_train = vectorizer.fit_transform(train_texts)
    print(f"  Train TF-IDF shape: {X_train.shape}")
    return X_train


def transform(vectorizer: TfidfVectorizer, texts):
    """Transform texts with an already-fitted vectorizer (val / test / new input)."""
    return vectorizer.transform(texts)


class SparseToDenseSequence(SequenceBase):
    """
    Keras data generator that converts one sparse TF-IDF batch to dense at a time.

    A full dense matrix of ~45,000 x 5,000 floats would use hundreds of MB.
    Converting batch-by-batch keeps memory use low.
    """

    def __init__(self, X_sparse, y=None, batch_size: int = BATCH_SIZE, shuffle: bool = False):
        super().__init__()
        self.X = X_sparse
        self.y = None if y is None else np.asarray(y, dtype=np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(X_sparse.shape[0])
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.X.shape[0])
        batch_idx = self.indices[start:end]
        X_batch = self.X[batch_idx].toarray().astype(np.float32)
        if self.y is None:
            return X_batch
        return X_batch, self.y[batch_idx]

    def on_epoch_end(self):
        # Uses the global NumPy RNG, which train.py seeds once at startup.
        if self.shuffle:
            np.random.shuffle(self.indices)
