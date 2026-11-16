# machine-audio-ad

Implementation of an ML model for acoustic anomaly detection.
It includes the autoencoder architecture, training and checkpoint selection,
dataset evaluation, and single-WAV inference.

## Setup on Ubuntu under WSL

With Miniconda installed in WSL, run these Bash commands from the repository
root. The environment uses Python 3.12, CPU TensorFlow, and FFmpeg.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env create --file environment.yml
conda activate machine-audio-ad
```

If the environment already exists, activate it and run `python -m pip install -e .`
to install the package and its Hydra dependency.

## Project layout

```text
main.py                       Hydra entry point
configs/config.yaml           Default task and shared checkpoint paths
configs/task/                 Prediction, training, evaluation, and inspection settings
src/
    __init__.py               Package marker
    features.py               WAV decoding and reference log-mel features
    model.py                  Autoencoder architecture, loading, and scoring
    prediction.py             Single-WAV prediction
    training.py               Training and validation-based checkpoint selection
    evaluation.py             ToyCar test-set scoring and metrics
```

Select a task with `task=...` and override settings with `task.setting=value`.
Use `--cfg job` to inspect the composed configuration without running the task.
Hydra saves each run's configuration and overrides under `outputs/`.

## Score a recording

```bash
python main.py task=predict task.audio_path=recording.wav
```

Standard output contains one raw score, the mean squared reconstruction error
averaged over all feature windows. Set `task.threshold=VALUE` to print a second line
with `anomalous` for scores at or above that value and `normal` for lower scores.
Audio is processed at its original sample rate and needs at least 27,648 samples
(1.728 seconds at 16 kHz).

## Model, training, and evaluation

`python main.py task=inspect` prints the locally defined architecture. It maps 640
log-mel inputs through four 128-unit blocks, an 8-unit bottleneck, four more
128-unit blocks, and a linear 640-unit output. Each hidden block applies a
dense layer, batch normalization, and ReLU.

To run your own training later, extract the ToyCar development dataset under
`dev_data/ToyCar/` so it contains `train/` and `test/`. The training directory
contains normal recordings. [DCASE 2020 development dataset](https://zenodo.org/records/3678171)

```bash
python main.py task=train
python main.py task=evaluate
```

Training uses Adam and mean squared error for 100 epochs by default, with
batches of 512 feature windows. It holds out 10% of the recordings for
validation before extracting windows. The epoch with the lowest validation
loss is saved to `checkpoints/toycar-best.keras`, with per-epoch losses in
`checkpoints/toycar-best.history.csv`.

Evaluation writes per-WAV scores to `results/toycar-scores.csv` and reports
ROC AUC and partial AUC at a maximum false positive rate of 0.1 for each labeled
machine ID. 
