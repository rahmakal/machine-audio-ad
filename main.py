"""Run a machine-audio-ad task with Hydra."""

import os

import hydra
from hydra.utils import call
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    call(config.task, _convert_="all")


if __name__ == "__main__":
    main()
