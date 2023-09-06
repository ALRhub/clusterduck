import logging
import os
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger("my_app")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info("Job config:")
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    rng = np.random.default_rng()
    duration = rng.integers(10)
    log.info(f"Waiting {duration} seconds...")
    time.sleep(duration)
    log.info("Job finished!")

    import torch
    print(torch.cuda.is_available())
    torch.cuda.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_capability(device))
    a = torch.randn(10000, 10000).to(device)
    b = torch.randn(10000, 10000).to(device)
    c = torch.matmul(a, b)
    print(c.sum())
    print(os.environ.get("CUDA_VISIBLE_DEVICES", "No GPU"))
    time.sleep(0.1)

    return "apple"


if __name__ == "__main__":
    my_app()
