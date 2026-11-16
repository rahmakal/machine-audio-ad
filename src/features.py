"""The reference log-mel features for the acoustic anomaly detector."""

import audioop
import sys
from pathlib import Path

import audioread
import librosa
import numpy as np


N_MELS = 128
CONTEXT_FRAMES = 5
FEATURE_SIZE = CONTEXT_FRAMES * N_MELS
FFT_SIZE = 1024
HOP_LENGTH = 512
POWER = 2.0
FRAME_START = 50
FRAME_STOP = 250
MIN_SAMPLES = (FRAME_START + CONTEXT_FRAMES - 1) * HOP_LENGTH


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Decode at the file's sample rate through the upstream PCM16 path."""
    try:
        with audioread.audio_open(str(path.resolve())) as audio:
            if audio.channels != 1:
                raise ValueError(f"Expected a mono WAV file: {path}")
            sample_rate = audio.samplerate
            pcm = np.frombuffer(b"".join(audio), dtype="<i2")
    except (audioread.DecodeError, audioop.error, EOFError, OSError) as error:
        raise ValueError(f"Unable to read WAV file: {path}") from error

    samples = pcm.astype(np.float32) / 32768.0
    return samples, sample_rate


def extract_features(path: Path) -> np.ndarray:
    """Return one row of 640 features per five consecutive log-mel frames."""
    samples, sample_rate = read_wav(path)
    if samples.size < MIN_SAMPLES:
        raise ValueError(
            f"Audio is too short. {path} has {samples.size} samples. "
            f"Feature extraction needs at least {MIN_SAMPLES} at the native sample rate."
        )

    # Librosa 0.6.0 reflects the full half-window on both ends before framing.
    padded_samples = np.pad(samples, FFT_SIZE // 2, mode="reflect")
    mel_power = librosa.feature.melspectrogram(
        y=padded_samples,
        sr=sample_rate,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH,
        win_length=FFT_SIZE,
        window="hann",
        center=False,
        power=POWER,
        n_mels=N_MELS,
        norm="slaney",
        dtype=np.float64,
    )
    log_mel = (20.0 / POWER) * np.log10(mel_power + sys.float_info.epsilon)
    frames = log_mel[:, FRAME_START:FRAME_STOP].T

    window_count = len(frames) - CONTEXT_FRAMES + 1
    features = np.empty((window_count, FEATURE_SIZE), dtype=np.float64)
    # Each row stores all mel bins of the earliest frame first.
    for offset in range(CONTEXT_FRAMES):
        first_bin = offset * N_MELS
        features[:, first_bin:first_bin + N_MELS] = frames[offset:offset + window_count]
    return features
