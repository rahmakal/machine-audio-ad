"""Score a ToyCar test directory and report ROC AUC for labeled recordings."""

import csv
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean

from sklearn.metrics import roc_auc_score

from .features import extract_features


RECORDING_NAME = re.compile(r"(?:(normal|anomaly)_)?(id_\d{2})_.+")


def auc_metrics(
    labels: Sequence[int], scores: Sequence[float], max_fpr: float
) -> tuple[float, float]:
    return float(roc_auc_score(labels, scores)), float(
        roc_auc_score(labels, scores, max_fpr=max_fpr)
    )


def report_metrics(scores_by_id: dict[str, list[tuple[int, float]]], max_fpr: float) -> None:
    aucs = []
    partial_aucs = []
    for machine_id, labeled_scores in sorted(scores_by_id.items()):
        labels, scores = zip(*labeled_scores)
        if len(set(labels)) < 2:
            print(f"{machine_id} needs both normal and anomalous recordings for AUC.")
            continue
        auc, partial_auc = auc_metrics(labels, scores, max_fpr)
        aucs.append(auc)
        partial_aucs.append(partial_auc)
        print(f"{machine_id}  AUC {auc:.6f}  pAUC {partial_auc:.6f}")
    if aucs:
        print(f"Mean over machine IDs  AUC {mean(aucs):.6f}  pAUC {mean(partial_aucs):.6f}")


def evaluate(
    model_path: str | Path,
    test_dir: str | Path,
    output: str | Path,
    max_fpr: float,
) -> None:
    """Write per-WAV reconstruction scores and report metrics for labeled data."""
    test_dir = Path(test_dir)
    output = Path(output)
    if output.suffix != ".csv":
        raise ValueError("Evaluation output path must end in .csv.")
    if not 0 < max_fpr <= 1:
        raise ValueError("max_fpr must be greater than 0 and at most 1.")

    recordings = sorted(path for path in test_dir.glob("*.wav") if path.is_file())
    if not recordings:
        raise ValueError(f"No WAV recordings found in {test_dir}")

    from .model import load_autoencoder, reconstruction_score

    rows = []
    scores_by_id = defaultdict(list)
    autoencoder = load_autoencoder(Path(model_path))
    for path in recordings:
        match = RECORDING_NAME.fullmatch(path.stem)
        if match is None:
            raise ValueError(f"Expected a DCASE ToyCar test filename: {path.name}")
        label, machine_id = match.groups()
        score = reconstruction_score(autoencoder, extract_features(path))
        rows.append((path.name, machine_id, label or "", score))
        if label is not None:
            scores_by_id[machine_id].append((int(label == "anomaly"), score))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "machine_id", "label", "score"))
        writer.writerows(rows)

    print(f"Saved {len(rows)} scores to {output}")
    report_metrics(scores_by_id, max_fpr)
