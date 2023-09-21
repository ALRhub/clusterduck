from __future__ import annotations

import logging
import os
import time
from typing import Sequence

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class Resource:
    def apply(self):
        raise NotImplementedError


class ResourcePool:
    _subclasses: dict[str, type[ResourcePool]] = {}

    def __init_subclass__(cls, *, kind: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._subclasses[kind] = cls

    def __new__(cls, *args, kind: str, **kwargs) -> ResourcePool:
        try:
            subcls = cls._subclasses[kind]
        except KeyError:
            raise ValueError(f"No ResourcePool registered under {kind=}.")
        return super().__new__(subcls)

    def get(self, index: int) -> Resource:
        raise NotImplementedError


class CUDAResource(Resource):
    def __init__(self, cuda_devices: Sequence[int]):
        self.cuda_devices = cuda_devices

    def apply(self):
        logger.debug(f"Setting CUDA devices to {self.cuda_devices}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.cuda_devices))


class CUDAResourcePool(ResourcePool, kind="cuda"):
    def __init__(
        self,
        kind: str,
        n_workers: int,
        gpus: Sequence[int] | None = None,
    ) -> None:
        if gpus is None:
            gpus = self.get_available_gpus()
            logger.info(f"Auto-detected the following CUDA devices: {gpus}")

        n_gpus = len(gpus)

        if n_workers >= n_gpus:
            if n_workers % n_gpus:
                raise ValueError(
                    f"The number of workers ({n_workers}) must be divisible by the number of GPUs ({n_gpus})."
                )
        else:
            if n_gpus % n_workers:
                raise ValueError(
                    f"The number of workers ({n_workers}) must evenly divide the number of GPUs ({n_gpus})."
                )
        workers_per_gpu = max(1, n_workers // n_gpus)
        gpus_per_worker = max(1, n_gpus // n_workers)

        gpu_groups: list[list[int]] = (
            np.array(gpus).repeat(workers_per_gpu).reshape(-1, gpus_per_worker).tolist()
        )
        self.gpu_resources = [CUDAResource(gpus) for gpus in gpu_groups]

    def get(self, index: int) -> Resource:
        return self.gpu_resources[index]

    @staticmethod
    def get_available_gpus() -> list[int]:
        # unset any masking of CUDA devices
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        try:
            import torch

            return list(range(torch.cuda.device_count()))
        except ImportError:
            pass

        try:
            import tensorflow as tf

            return [
                int(device.name) for device in tf.config.list_physical_devices("GPU")
            ]
        except ImportError:
            pass

        try:
            import pycuda.autoinit
            import pycuda.driver as cuda

            return list(range(cuda.Device.count()))
        except ImportError:
            pass

        raise ImportError(
            "Either PyTorch, Tensorflow or PyCUDA must be installed to use CUDAResource"
        )


class CPUResource(Resource):
    def __init__(self, cpus: Sequence[int]):
        self.cpus = list(cpus)

    def apply(self):
        logger.debug(f"Setting CPU affinity to {self.cpus}")
        psutil.Process().cpu_affinity(self.cpus)


class CPUResourcePool(ResourcePool, kind="cpu"):
    def __init__(
        self,
        kind: str,
        n_workers: int,
        cpus: Sequence[int] | None = None,
    ) -> None:
        if cpus is None:
            cpus = self.get_available_cpus()
            logger.info(f"Auto-detected the following CPUs: {cpus}")

        n_cpus = len(cpus)

        if n_workers > n_cpus:
            raise ValueError(
                f"Cannot have more workers ({n_workers}) than CPUs ({n_cpus})!"
            )

        if n_cpus % n_workers:
            raise ValueError(
                f"The number of workers ({n_workers}) must evenly divide the number of CPUs ({n_cpus})."
            )

        cpus_per_worker = n_cpus // n_workers
        cpu_groups: list[list[int]] = (
            np.array(cpus).reshape(-1, cpus_per_worker).tolist()
        )
        self.cpu_resources = [CPUResource(cpus) for cpus in cpu_groups]

    def get(self, index: int) -> Resource:
        return self.cpu_resources[index]

    @staticmethod
    def get_available_cpus() -> list[int]:
        import psutil

        psutil.Process().cpu_affinity([])  # "clear" process affinity
        cpus = psutil.Process().cpu_affinity()
        assert cpus is not None
        return cpus


class StaggeredStart(Resource):
    def __init__(self, delay):
        self.delay = delay

    def apply(self):
        if self.delay > 0:
            logger.debug(f"Sleeping for {self.delay} to stagger job start.")
            time.sleep(self.delay)


class StaggeredStarts(ResourcePool, kind="stagger"):
    def __init__(self, kind: str, n_workers: int, delay: float) -> None:
        delays = [delay * i for i in range(n_workers)]
        self.delays = [StaggeredStart(delay=delay) for delay in delays]

    def get(self, index: int) -> Resource:
        delay = self.delays[index]
        # delays should only be applied on each worker once
        self.delays[index] = StaggeredStart(delay=0)
        return delay
