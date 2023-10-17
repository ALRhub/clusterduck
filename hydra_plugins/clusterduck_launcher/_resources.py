from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import psutil
from omegaconf import DictConfig

from ._logging import get_logger

logger = get_logger(__name__)


def create_resource_pools_from_cfg(
    config: DictConfig,
    n_workers: int,
) -> list[ResourcePool]:
    resource_pools = []
    for kind, resource_cfg in config.items():
        logger.debug(f"Scheduling {kind} resources with config: {resource_cfg}")
        # e.g.
        # resources_config:
        #   cuda:
        # or
        # resources_config:
        #   cuda:
        #     gpus: [0, 1, 2, 3]
        resource_cfg = resource_cfg or {}
        resource_pools.append(
            ResourcePool(
                kind=kind,
                n_workers=n_workers,
                **resource_cfg,
            )
        )
    return resource_pools


class Resource:
    def apply(self):
        raise NotImplementedError


class ResourcePool:
    _subclasses: dict[str, type[ResourcePool]] = {}

    def __init_subclass__(cls, *, kind: str | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if kind is not None:
            cls._subclasses[kind] = cls

    def __new__(cls, *args, kind: str | None = None, **kwargs) -> ResourcePool:
        if kind is None:
            # called during unpickling with no arguments, but cls is the class we need to create
            if cls is ResourcePool:
                raise TypeError(
                    "ResourcePool.__new__() missing 1 required keyword-only argument: 'kind'"
                )
            return super().__new__(cls)
        try:
            subcls = cls._subclasses[kind]
        except KeyError:
            raise ValueError(f"No ResourcePool registered under {kind=}.")
        return super().__new__(subcls)

    def get(self, index: int) -> Resource:
        raise NotImplementedError


class GpuResources(ResourcePool):
    @staticmethod
    def get_available_gpus() -> list[int]:
        if (gpus := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
            # SLURM sets CUDA_VISIBLE_DEVICES for each job and each step
            # https://slurm.schedmd.com/gres.html#GPU_Management
            logger.debug(
                "Using `CUDA_VISIBLE_DEVICES` to determine the available GPU resources."
            )
            return [int(gpu) for gpu in gpus.split(",")]

        if (gpus := os.environ.get("SLURM_JOB_GPUS")) is not None:
            logger.debug(
                "Using `SLURM_JOB_GPUS` to determine the available GPU resources."
            )
            return [int(gpu) for gpu in gpus.split(",")]

        logger.debug(
            "Trying to detect number of gpus using an ML library (e.g. pytorch)"
        )

        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Manager() as manager:
            gpus = manager.list()
            p = ctx.Process(
                target=GpuResources._get_available_gpus_destructive,
                args=(gpus,),
            )
            p.start()
            p.join()

            if gpus and gpus[0] is None:
                raise RuntimeError(
                    "Either pycuda, Pytorch, or TensorFlow must be installed in order to auto-detect GPU resources."
                )

            return list(gpus)

    @staticmethod
    def _get_available_gpus_destructive(gpus: list[int | None]) -> None:
        """These imports modify program state, so we do them in a separate process."""
        try:
            import pycuda.autoinit
            import pycuda.driver as cuda

            n_gpus = cuda.Device.count()
            logger.debug(f"Found {n_gpus} GPUs using pycuda")
            gpus.extend(range(n_gpus))
            return
        except ImportError:
            pass

        try:
            import torch

            n_gpus = torch.cuda.device_count()
            logger.debug(f"Found {n_gpus} GPUs using Pytorch")
            gpus.extend(range(n_gpus))
            return
        except ImportError:
            pass

        try:
            import tensorflow as tf

            devices = tf.config.list_physical_devices("GPU")
            logger.debug(f"Found the following GPUs using TensorFlow: {devices}")
            gpus.extend(int(device.name) for device in devices)
            return
        except ImportError:
            pass

        # None signals that auto-detection of GPUs has failed
        gpus.append(None)


@dataclass
class CudaDevices(Resource):
    gpus: list[int]

    def apply(self):
        logger.debug(f"Setting CUDA devices to {self.gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))


class Cuda(GpuResources, kind="cuda"):
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
        if n_workers > 1:
            logger.debug(
                f"CUDA will be allocated the following GPU groups: {gpu_groups}"
            )
        self.gpu_resources = [CudaDevices(gpus) for gpus in gpu_groups]

    def get(self, index: int) -> CudaDevices:
        return self.gpu_resources[index]


@dataclass
class HeadlessRenderDevice(Resource):
    gpu: int

    def apply(self):
        logger.debug(
            f"Setting device for EGL rendering (`EGL_DEVICE_ID`) to {self.gpu}"
        )
        os.environ["EGL_DEVICE_ID"] = str(self.gpu)
        # Reference:
        # https://github.com/mmatl/pyrender/issues/92
        # https://pyrender.readthedocs.io/en/latest/_modules/pyrender/offscreen.html
        logger.debug(
            f"Setting device for pyglet headless rendering (`PYGLET_HEADLESS_DEVICE`) to {self.gpu}"
        )
        os.environ["PYGLET_HEADLESS_DEVICE"] = str(self.gpu)
        # reference:
        # https://pyglet.readthedocs.io/en/latest/programming_guide/options.html#pyglet.options
        # https://github.com/pyglet/pyglet/issues/51
        logger.debug(
            f"Setting device for Mujoco headless rendering (`MUJOCO_EGL_DEVICE_ID`) to {self.gpu}"
        )
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(self.gpu)
        # reference:
        # https://github.com/google-deepmind/dm_control/issues/345
        # https://pytorch.org/rl/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html#common-issues-during-import-or-when-rendering-mujoco-environments


class HeadlessRendering(GpuResources, kind="rendering"):
    def __init__(
        self,
        kind: str,
        n_workers: int,
        gpus: Sequence[int] | None = None,
    ) -> None:
        if gpus is None:
            gpus = self.get_available_gpus()
            logger.info(f"Auto-detected the following GPU devices: {gpus}")

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
        gpus_allocs: list[int] = (
            np.array(gpus).repeat(workers_per_gpu)[::gpus_per_worker].tolist()
        )
        if n_workers > 1:
            logger.debug(
                f"EGL contexts will be allocated the following GPUs: {gpus_allocs}"
            )
        self.gpu_resources = [HeadlessRenderDevice(gpu) for gpu in gpus_allocs]

    def get(self, index: int) -> HeadlessRenderDevice:
        return self.gpu_resources[index]


@dataclass
class CpuGroup(Resource):
    cpus: list[int]

    def apply(self):
        logger.debug(f"Setting CPU affinity to {self.cpus}")
        psutil.Process().cpu_affinity(self.cpus)


class Cpus(ResourcePool, kind="cpu"):
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
        if n_workers > 1:
            logger.debug(f"Allocated the following CPU groups: {cpu_groups}")
        self.cpu_resources = [CpuGroup(cpus) for cpus in cpu_groups]

    def get(self, index: int) -> CpuGroup:
        return self.cpu_resources[index]

    @staticmethod
    def get_available_cpus() -> list[int]:
        import psutil

        psutil.Process().cpu_affinity([])  # "clear" process affinity
        cpus = psutil.Process().cpu_affinity()
        assert cpus is not None
        return cpus


@dataclass
class StaggeredStart(Resource):
    delay: float

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
