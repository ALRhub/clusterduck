import os
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print("Job config")
    print(OmegaConf.to_yaml(cfg))
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    rng = np.random.default_rng()
    duration = rng.integers(10)
    print(f"Waiting {duration} seconds")
    time.sleep(duration)


if __name__ == "__main__":
    my_app()
