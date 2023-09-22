import logging
import os
import time

import hydra
import numpy as np
import psutil
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("training")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    logger.info(
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(ALL)')}"
    )
    logger.info(f"PyTorch detected {torch.cuda.device_count()} devices.")
    logger.info(f"CPU affinity: {psutil.Process().cpu_affinity()}")

    logger.info(f"Job config:\n{OmegaConf.to_yaml(cfg)}")

    duration = np.random.default_rng().integers(10)
    logger.info(f"Waiting {duration} seconds...")
    time.sleep(duration)

    logger.info("Job finished!")


if __name__ == "__main__":
    train()
