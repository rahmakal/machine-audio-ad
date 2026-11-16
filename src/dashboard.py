"""Local browser dashboard for inspecting and replaying ToyCar recordings."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from threading import Lock

import numpy as np
from flask import Flask, jsonify, send_file, send_from_directory

from .evaluation import RECORDING_NAME, auc_metrics
from .features import CONTEXT_FRAMES, FRAME_START, HOP_LENGTH, extract_features, read_wav


MAX_FPR = 0.1
NORMAL_SCORE_QUANTILE = 1 - MAX_FPR


def load_saved_scores(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"filename", "score"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Evaluation CSV needs filename and score columns: {path}")
        scores = {}
        for row in reader:
            score = float(row["score"])
            if math.isfinite(score):
                scores[row["filename"]] = score
        return scores


def recording_catalog(recordings_dir: Path, scores: dict[str, float]) -> dict[str, dict]:
    recordings = {}
    for path in sorted(recordings_dir.glob("*.wav")):
        match = RECORDING_NAME.fullmatch(path.stem)
        if match and path.is_file():
            label, machine_id = match.groups()
            recordings[path.name] = {
                "filename": path.name,
                "machine": machine_id,
                "label": label,
                "saved_score": scores.get(path.name),
            }
    return recordings


def summarize_machines(recordings: dict[str, dict]) -> list[dict]:
    by_machine = defaultdict(list)
    for recording in recordings.values():
        by_machine[recording["machine"]].append(recording)

    summaries = []
    for machine, clips in sorted(by_machine.items()):
        normal_scores = [
            clip["saved_score"] for clip in clips
            if clip["label"] == "normal" and clip["saved_score"] is not None
        ]
        anomaly_scores = [
            clip["saved_score"] for clip in clips
            if clip["label"] == "anomaly" and clip["saved_score"] is not None
        ]
        auc, pauc = None, None
        if normal_scores and anomaly_scores:
            labels = [0] * len(normal_scores) + [1] * len(anomaly_scores)
            auc, pauc = auc_metrics(labels, normal_scores + anomaly_scores, MAX_FPR)
        summaries.append({
            "id": machine,
            "count": len(clips),
            "auc": auc,
            "pauc": pauc,
            "demo_limit": float(np.quantile(normal_scores, NORMAL_SCORE_QUANTILE))
            if normal_scores else None,
        })
    return summaries


def create_app(model_path: Path, recordings_dir: Path, scores_path: Path) -> Flask:
    app = Flask(__name__)
    recordings_dir = recordings_dir.resolve()
    scores_path = scores_path.resolve()
    saved_scores = load_saved_scores(scores_path)
    recordings = recording_catalog(recordings_dir, saved_scores)
    summaries = summarize_machines(recordings)
    model = None
    model_lock = Lock()

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/catalog")
    def catalog():
        return jsonify(
            recordings=list(recordings.values()),
            machines=summaries,
            checkpoint=model_path.name,
            checkpoint_available=model_path.is_file(),
            recordings_dir=str(recordings_dir),
            scores_available=bool(saved_scores),
        )

    @app.get("/api/recordings/<filename>/audio")
    def audio(filename):
        if filename not in recordings:
            return jsonify(error="Recording not found."), 404
        return send_from_directory(recordings_dir, filename, mimetype="audio/wav")

    @app.get("/api/recordings/<filename>/waveform")
    def waveform(filename):
        if filename not in recordings:
            return jsonify(error="Recording not found."), 404
        try:
            samples, sample_rate = read_wav(recordings_dir / filename)
            if not samples.size or sample_rate <= 0:
                raise ValueError("Recording contains no playable samples.")
        except ValueError as error:
            return jsonify(error=str(error)), 422
        blocks = np.array_split(samples, min(samples.size, 230))
        peaks = [[float(block.min()), float(block.max())] for block in blocks]
        return jsonify(peaks=peaks, sample_rate=sample_rate, duration=samples.size / sample_rate)

    @app.post("/api/recordings/<filename>/score")
    def score(filename):
        nonlocal model
        if filename not in recordings:
            return jsonify(error="Recording not found."), 404
        if not model_path.is_file():
            return jsonify(error=f"Model file not found: {model_path}"), 503
        try:
            features = extract_features(recordings_dir / filename)
            from .model import load_autoencoder, reconstruction_errors

            with model_lock:
                if model is None:
                    model = load_autoencoder(model_path)
                errors = reconstruction_errors(model, features)
            if not np.isfinite(errors).all():
                raise ValueError("The model produced an invalid reconstruction error.")
        except ValueError as error:
            return jsonify(error=str(error)), 422
        return jsonify(
            score=float(np.mean(errors)),
            errors=errors.tolist(),
            first_center_sample=(FRAME_START + (CONTEXT_FRAMES - 1) / 2) * HOP_LENGTH,
            hop_samples=HOP_LENGTH,
        )

    @app.get("/api/evaluation.csv")
    def evaluation_csv():
        if not scores_path.is_file():
            return jsonify(error="The evaluation CSV is unavailable."), 404
        return send_file(scores_path, mimetype="text/csv", as_attachment=True)

    return app


def serve_dashboard(model_path: str, recordings_dir: str, scores_path: str, port: int) -> None:
    app = create_app(Path(model_path), Path(recordings_dir), Path(scores_path))
    app.run(host="127.0.0.1", port=port, use_reloader=False)
