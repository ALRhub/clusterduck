import logging
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("train")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> float:
    logger.info(f"Job config:\n{OmegaConf.to_yaml(cfg)}")

    duration = np.random.default_rng().integers(10)
    logger.info(f"Waiting {duration} seconds...")
    time.sleep(duration)

    # Results for instance useful for optuna optimizations:
    a: float = cfg.a
    b: float = cfg.b
    result = a**2 + b**2

    logger.info(f"Job finished with {result=}!")

    return result


if __name__ == "__main__":
    train()
