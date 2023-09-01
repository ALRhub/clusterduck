# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time

import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    import torch
    print(torch.cuda.is_available(), torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_capability(device))
    a = torch.randn(10000, 10000).to(device)
    b = torch.randn(10000, 10000).to(device)
    c = torch.matmul(a, b)
    print(c.sum())
    print(os.environ.get("CUDA_VISIBLE_DEVICES", "No GPU"))
    time.sleep(0.1)

    print(OmegaConf.to_yaml(cfg))
    return "Hello World!"


if __name__ == "__main__":
    my_app()
