from __future__ import annotations

import logging
import os
import shlex
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

PARAMETER_SYNONYMS = {
    "name": "job_name",
    "timeout_min": "time",
    "mem_gb": "mem",
    "tasks_per_node": "ntasks_per_node",
}

SPECIAL_KWARGS = ["array_count", "array_parallelism", "stderr_to_stdout"]


log = logging.getLogger("clusterduck")


def clean_slurm_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Cleans up the kwargs by mapping parameter synonyms to their canonical
    names and replacing underscores with hyphens.
    """
    # Map parameter synonyms to their canonical names
    kwargs = {PARAMETER_SYNONYMS.get(key, key): value for key, value in kwargs.items()}

    # Replace underscores with hyphens, unless the key is in SPECIAL_KWARGS
    # (these are used internally by make_sbatch_string and are not passed to
    # sbatch/srun)
    kwargs = {
        key.replace("_", "-") if key not in SPECIAL_KWARGS else key: value
        for key, value in kwargs.items()
    }

    return kwargs


def make_sbatch_string(
    command: list[str],
    log_folder: str | Path,
    sbatch_kwargs: dict[str, Any] | None = None,
    srun_kwargs: dict[str, Any] | None = None,
    setup: list[str] | None = None,
    teardown: list[str] | None = None,
    use_srun: bool = True,
) -> str:
    log_folder = Path(log_folder)
    sbatch_kwargs = sbatch_kwargs or {}
    srun_kwargs = srun_kwargs or {}

    # Filter out None values (unset values)
    sbatch_kwargs = {
        key: value for key, value in sbatch_kwargs.items() if value is not None
    }
    srun_kwargs = {
        key: value for key, value in srun_kwargs.items() if value is not None
    }

    array_count = sbatch_kwargs.pop("array_count", 1)
    array_parallelism = sbatch_kwargs.pop("array_parallelism", 256)
    stderr_to_stdout = sbatch_kwargs.pop("stderr_to_stdout", False)

    # TODO: cleanup log file paths
    # TODO: make these paths configurable via launcher config
    if array_count == 1:
        stdout = log_folder / "%j_out.log"
        stderr = log_folder / "%j_err.log"
        srun_stdout = log_folder / "%j_%t_out.log"
        srun_stderr = log_folder / "%j_%t_err.log"
    else:
        sbatch_kwargs["array"] = (
            f"0-{array_count - 1}%{min(array_count, array_parallelism)}"
        )
        stdout = log_folder / "%A_%a_out.log"
        stderr = log_folder / "%A_%a_err.log"
        srun_stdout = log_folder / "%A_%a_%t_out.log"
        srun_stderr = log_folder / "%A_%a_%t_err.log"

    sbatch_kwargs["output"] = str(stdout)
    if not stderr_to_stdout:
        sbatch_kwargs["error"] = str(stderr)

    srun_kwargs["output"] = str(srun_stdout)
    if not stderr_to_stdout:
        srun_kwargs["error"] = str(srun_stderr)

    sbatch_kwargs["open-mode"] = "append"
    srun_kwargs["open-mode"] = "append"

    lines = ["#!/bin/bash", "", "# Parameters"]
    for key, value in sbatch_kwargs.items():
        lines.append(as_sbatch_flag(key, value))

    # environment setup:
    if setup:
        lines += ["", "# setup"] + setup

    if use_srun:
        srun = ["srun", "--unbuffered"]
        for key, value in srun_kwargs.items():
            srun.append(as_srun_args(key, value))

        command = srun + command

    lines += ["", "# command", shlex.join(command), ""]

    # environment teardown:
    if teardown:
        lines += ["", "# teardown"] + teardown

    return "\n".join(lines)


def as_sbatch_flag(key: str, value) -> str:
    if value is True:
        return f"#SBATCH --{key}"

    value = shlex.quote(str(value))
    return f"#SBATCH --{key}={value}"


def as_srun_args(key: str, value) -> str:
    if value is True:
        return f"--{key}"

    value = shlex.quote(str(value))
    return f"--{key}={value}"


sentinel = object()


@dataclass
class SlurmJobEnvironment:

    ENV_VARS_TO_LOG = [
        "HOSTNAME",
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_STEP_ID",
        "SLURM_PROCID",
        "SLURM_NTASKS",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "CUDA_VISIBLE_DEVICES",
        "EGL_DEVICE_ID",
    ]

    def __post_init__(self):
        self.detect_cpu()
        self.detect_mem()
        self.detect_gpu()

        for key in self.ENV_VARS_TO_LOG:
            log.debug(f"Env variable {key}={os.environ.get(key, '[UNSET]')}")

    @cached_property
    def job_id(self) -> str:
        """Fetches the job id from the environment variable set by slurm"""
        return self.try_get_env_var("SLURM_JOB_ID", default="LOCAL")

    @cached_property
    def array_job_id(self) -> str:
        """Fetches the job id from the environment variable set by slurm."""
        return self.try_get_env_var("SLURM_ARRAY_JOB_ID", default="LOCAL")

    @cached_property
    def array_index(self) -> int:
        """Fetches the array id from the environment variable set by slurm."""
        return int(self.try_get_env_var("SLURM_ARRAY_TASK_ID", default=0))

    @cached_property
    def task_index(self) -> int:
        """Fetches the task index from the environment variable set by slurm.

        Note: SLURM_LOCALID is the index of the task on the current node,
        whereas SLURM_PROCID is the global index of the task across all
        nodes. (only relevant for multi-node jobs).
        """
        return int(self.try_get_env_var("SLURM_PROCID", default=0))

    @cached_property
    def n_tasks(self) -> int:
        """Fetches the number of tasks per job from the environment variable
        set by slurm.

        Note: An array job with N subjobs will launch this many tasks per subjob.
        """
        return int(self.try_get_env_var("SLURM_NTASKS", default=1))

    @property
    def global_rank(self) -> int:
        """Fetches the global rank of the task from the environment variable set by slurm."""
        return self.array_index * self.n_tasks + self.task_index

    @staticmethod
    def try_get_env_var(key: str, default: Any = sentinel) -> Any:
        try:
            return os.environ[key]
        except KeyError:
            if default is not sentinel:
                log.warning(
                    f"Could not find {key} in environment variables. Using default value: {default}"
                )
                return default
            else:
                raise RuntimeError(
                    f"Could not find {key} in environment variables. Make sure that the job is running inside a slurm allocation."
                )

    @staticmethod
    def detect_cpu() -> None:
        try:
            import psutil

            log.debug(f"[psutil] total #CPUs: {psutil.cpu_count()}")
            log.debug(
                f"[psutil] available #CPUs: {len(psutil.Process().cpu_affinity())}"
            )
            log.debug(f"[psutil] CPU affinity: {psutil.Process().cpu_affinity()}")

            return

        except ImportError:
            pass

        import multiprocessing as mp

        log.debug(f"[mp] total #CPUs: {mp.cpu_count()}")
        log.debug(f"[os] available #CPUs: {len(os.sched_getaffinity(0))}")
        log.debug(f"[os] CPU affinity: {list(os.sched_getaffinity(0))}")

    @staticmethod
    def detect_mem() -> None:
        try:
            import psutil

            log.debug(
                f"[psutil] Total memory: {psutil.virtual_memory().total / (1024 ** 3):.3f}G"
            )

            return

        except ImportError:
            pass

        log.debug(
            f"[os] Total memory: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 ** 3):.3f}G"
        )

    @staticmethod
    def detect_gpu() -> None:
        try:
            import pycuda.driver as cuda

            cuda.init()
            log.debug(f"[pycuda] {cuda.Device.count()} GPU(s) detected:")
            for i in range(cuda.Device.count()):
                log.debug(f"[pycuda] GPU {i} at address {cuda.Device(i).pci_bus_id()}")

            return

        except ImportError:
            pass

        try:
            import torch

            log.debug(f"[pytorch] {torch.cuda.device_count()} GPU(s) detected.")

            return

        except ImportError:
            pass

        log.debug("Could not detect GPUs. Neither pycuda nor pytorch is installed.")
