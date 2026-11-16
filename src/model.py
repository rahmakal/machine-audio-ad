"""The autoencoder and its reconstruction score."""

from pathlib import Path

import numpy as np
from tensorflow import keras

from .features import FEATURE_SIZE


HIDDEN_WIDTHS = (128, 128, 128, 128, 8, 128, 128, 128, 128)


def build_autoencoder() -> keras.Model:
    """Build the reference dense autoencoder with an eight-unit bottleneck."""
    inputs = keras.Input(shape=(FEATURE_SIZE,), dtype="float32")
    hidden = inputs
    for width in HIDDEN_WIDTHS:
        hidden = keras.layers.Dense(width)(hidden)
        hidden = keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001)(hidden)
        hidden = keras.layers.Activation("relu")(hidden)
    outputs = keras.layers.Dense(FEATURE_SIZE)(hidden)
    return keras.Model(inputs, outputs, name="toycar_autoencoder")


def load_autoencoder(path: Path) -> keras.Model:
    """Restore the full model from the HDF5 or a Keras checkpoint."""
    if not path.is_file():
        raise ValueError(f"Model file not found: {path}")

    try:
        model = keras.models.load_model(str(path), compile=False)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Unable to load Keras model from {path}: {error}") from error

    expected_shape = (None, FEATURE_SIZE)
    if model.input_shape != expected_shape or model.output_shape != expected_shape:
        raise ValueError(f"Expected a {FEATURE_SIZE}-feature autoencoder in {path}.")
    return model


def reconstruction_errors(model: keras.Model, features: np.ndarray) -> np.ndarray:
    """Return the mean squared reconstruction error for each feature window."""
    reconstruction = model.predict(features, verbose=0)
    return np.mean(np.square(features - reconstruction), axis=1)


def reconstruction_score(model: keras.Model, features: np.ndarray) -> float:
    """Average squared reconstruction error across features, then across windows."""
    return float(np.mean(reconstruction_errors(model, features)))


def describe_model() -> None:
    build_autoencoder().summary()
