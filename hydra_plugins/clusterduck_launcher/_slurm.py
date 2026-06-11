from __future__ import annotations

import logging
import os
import re
import shlex
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from ._utils import run_command

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
    sbatch_kwargs: dict[str, Any] | None = None,
    srun_kwargs: dict[str, Any] | None = None,
    setup: list[str] | None = None,
    teardown: list[str] | None = None,
    use_srun: bool = True,
) -> str:
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

    if array_count > 1:
        array_command = f"0-{array_count - 1}"
        if array_parallelism < array_count:
            # maximum number of simultaneously running tasks from the job array
            array_command += f"%{array_parallelism}"
        sbatch_kwargs["array"] = array_command

    lines = ["#!/bin/bash", "", "# Parameters"]
    for key, value in sbatch_kwargs.items():
        lines.append(as_sbatch_flag(key, value))

    # environment setup:
    if setup:
        lines += ["", "# setup"] + setup

    lines += ["", "# command"]
    command_str = shlex.join(command)

    if use_srun:
        lines.append("srun --unbuffered \\")
        for key, value in srun_kwargs.items():
            lines.append(f"    {as_srun_args(key, value)} \\")

        lines.append("    " + command_str)

    else:
        lines.append(command_str)

    # environment teardown:
    if teardown:
        lines += ["", "# teardown"] + teardown

    lines.append("")
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


def extract_job_id(sbatch_output: str) -> str:
    """Extracts the job id from the output of sbatch command."""
    # sbatch output is expected to be in the format "Submitted batch job <job_id>"
    try:
        return sbatch_output.strip().split()[-1]
    except Exception as e:
        raise RuntimeError(
            f"Could not extract job id from sbatch output: {sbatch_output}"
        ) from e


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
        self.detect_alloc_mem()
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

    def detect_alloc_mem(self) -> None:
        """Detects the memory allocated by slurm for the current job. This is
        useful to detect if we are running in an environment with memory limits
        set by slurm, and to log the allocated memory for debugging purposes.

        Note: This is not necessarily the same as the total memory available on
        the node, since slurm can be configured to allocate only a subset of
        the total memory to each job.
        """
        if self.job_id == "LOCAL":
            # skip detection if we are not running inside a slurm job
            return

        try:
            scontrol_output = run_command(["scontrol", "show", "job", str(self.job_id)])
        except RuntimeError as e:
            log.debug(f"Could not run scontrol to detect allocated memory: {e}")
            return

        mem = extract_tres_allocated_memory(scontrol_output)
        if mem:
            log.debug(f"[scontrol] job allocated memory: {mem}")
        else:
            log.debug(
                f"Could not detect allocated memory from scontrol output: \n\n {scontrol_output}"
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


def extract_tres_allocated_memory(scontrol_output: str) -> str | None:
    """
    Parses scontrol output to find the TRES memory allocation.
    Makes minimal assumptions about line numbers, ordering, or spacing.
    """
    # 1. Locate the TRES field. \bTRES=(\S+) looks for 'TRES='
    # and captures all non-whitespace characters belonging to that field.
    tres_match = re.search(r"\bTRES=(\S+)", scontrol_output)
    if not tres_match:
        return None

    tres_value = tres_match.group(1)

    # 2. Within the TRES string (e.g., "cpu=64,mem=128000M,node=1"),
    # extract the value following "mem=" up until the next comma or end of string.
    mem_match = re.search(r"\bmem=([^,]+)", tres_value)
    if mem_match:
        return mem_match.group(1)

    return None
