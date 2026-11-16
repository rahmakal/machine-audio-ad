"""Train the ToyCar autoencoder on normal recordings."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .features import extract_features


def find_recordings(directories: list[Path]) -> list[Path]:
    recordings = set()
    for directory in directories:
        paths = [path for path in directory.glob("normal_*.wav") if path.is_file()]
        if not paths:
            raise ValueError(f"No normal_*.wav recordings found in {directory}")
        recordings.update(path.resolve() for path in paths)
    if len(recordings) < 2:
        raise ValueError("Need at least two normal recordings for the training and validation split.")
    return sorted(recordings)


def load_training_windows(recordings: list[Path]) -> np.ndarray:
    windows = [extract_features(path).astype(np.float32) for path in recordings]
    return np.concatenate(windows)


def train(
    train_dirs: Sequence[str | Path],
    checkpoint: str | Path,
    epochs: int,
    batch_size: int,
    validation_fraction: float,
    seed: int,
) -> None:
    """Fit the autoencoder and save the epoch with the lowest validation loss."""
    checkpoint = Path(checkpoint)
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if checkpoint.suffix != ".keras":
        raise ValueError("Training checkpoint path must end in .keras.")

    history_path = checkpoint.with_suffix(".history.csv")
    for path in (checkpoint, history_path):
        if path.exists():
            raise ValueError(f"Output already exists: {path}")

    recordings = find_recordings([Path(directory) for directory in train_dirs])
    train_files, validation_files = train_test_split(
        recordings, test_size=validation_fraction, random_state=seed
    )
    training = load_training_windows(train_files)
    validation = load_training_windows(validation_files)

    from tensorflow import keras

    from .model import build_autoencoder

    keras.utils.set_random_seed(seed)
    model = build_autoencoder()
    model.compile(optimizer="adam", loss="mean_squared_error")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        initial_value_threshold=float("inf"),
    )
    print(f"Training on {len(train_files)} recordings, validating on {len(validation_files)}.")
    model.fit(
        training,
        training,
        validation_data=(validation, validation),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[best_checkpoint, keras.callbacks.CSVLogger(history_path)],
    )
    if not checkpoint.is_file():
        raise ValueError("Training produced no checkpoint with an improved validation loss.")
    print(f"Best validation loss {best_checkpoint.best:.6g} saved to {checkpoint}")
