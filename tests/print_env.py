import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def print_env(cfg: DictConfig) -> None:
    import logging
    import os

    logger = logging.getLogger()

    logger.info(f"Job config:\n{OmegaConf.to_yaml(cfg)}")

    for key in cfg.env_vars:
        logger.info(f"Env variable {key}={os.environ.get(key, '[UNSET]')}")

    if cfg.cpu.detect_w_psutil:
        import psutil

        logger.info(f"[psutil] CPU affinity: {psutil.Process().cpu_affinity()}")

    if cfg.cpu.detect_w_os:
        logger.info(f"[os] CPU affinity: {list(os.sched_getaffinity(0))}")

    if cfg.gpu.detect_w_pycuda:
        import pycuda.driver as cuda

        cuda.init()
        for i in cuda.Device.count():
            logger.info(f"[pycuda] GPU {i} at address {cuda.Device(i).pci_bus_id()}")

    if cfg.gpu.detect_w_pytorch:
        import torch

        logger.info(f"[pytorch] {torch.cuda.device_count()} GPUs detected.")

    logger.info(f"Job finished.")


if __name__ == "__main__":
    print_env()
