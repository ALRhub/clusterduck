import logging
import os
import time

import hydra
import numpy as np
import psutil
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger("my_app")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info("Job config:")
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    log.info(f"CPU affinity: {psutil.Process().cpu_affinity()}")

    rng = np.random.default_rng()
    duration = rng.integers(10)
    log.info(f"Waiting {duration} seconds...")
    time.sleep(duration)

    log.info("Job finished!")


if __name__ == "__main__":
    my_app()
