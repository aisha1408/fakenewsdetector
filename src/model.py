"""
Build the feedforward neural network.

Architecture (from the project spec):
  Input(5000 TF-IDF features)
    -> Dense(128, relu) + Dropout(0.5)
    -> Dense(64, relu)  + Dropout(0.3)
    -> Dense(1, sigmoid)   # probability of class "fake"

Sigmoid + binary cross-entropy is the standard setup for a yes/no (fake vs real) task.
Dropout randomly turns off neurons during training so the network cannot over-rely
on a few TF-IDF features (which also reduces overfitting).
"""

import os

if "KERAS_BACKEND" not in os.environ:
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        os.environ["KERAS_BACKEND"] = "torch"

try:
    import keras
    from keras import layers
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers

from src.config import (
    DROPOUT_1,
    DROPOUT_2,
    HIDDEN_UNITS_1,
    HIDDEN_UNITS_2,
    LEARNING_RATE,
    MAX_FEATURES,
)


def build_model(input_dim: int = MAX_FEATURES) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(HIDDEN_UNITS_1, activation="relu"),
            layers.Dropout(DROPOUT_1),
            layers.Dense(HIDDEN_UNITS_2, activation="relu"),
            layers.Dropout(DROPOUT_2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="fake_news_ffnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
