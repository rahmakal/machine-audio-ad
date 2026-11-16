"""Score and optionally label one WAV."""

import math
from pathlib import Path


def predict(
    model_path: str | Path,
    audio_path: str | Path,
    threshold: float | None = None,
) -> None:
    """Print the reconstruction score and an optional threshold-based label."""
    model_path = Path(model_path)
    audio_path = Path(audio_path)
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("Threshold must be a finite number.")
    if not model_path.is_file():
        raise ValueError(f"Model file not found: {model_path}")
    if not audio_path.is_file():
        raise ValueError(f"Unable to read WAV file: {audio_path}")

    from .features import extract_features

    features = extract_features(audio_path)

    from .model import load_autoencoder, reconstruction_score

    score = reconstruction_score(load_autoencoder(model_path), features)
    print(score)
    if threshold is not None:
        print("anomalous" if score >= threshold else "normal")
