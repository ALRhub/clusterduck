from __future__ import annotations

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

    # Map parameter synonyms to their canonical names
    sbatch_kwargs = {
        PARAMETER_SYNONYMS.get(key, key): value for key, value in sbatch_kwargs.items()
    }
    srun_kwargs = {
        PARAMETER_SYNONYMS.get(key, key): value for key, value in srun_kwargs.items()
    }

    array_count = sbatch_kwargs.pop("array_count", 1)
    array_parallelism = sbatch_kwargs.pop("array_parallelism", 256)
    stderr_to_stdout = sbatch_kwargs.pop("stderr_to_stdout", False)

    # TODO: cleanup log file paths
    if array_count == 1:
        stdout = log_folder / "%j/%j_out.log"
        stderr = log_folder / "%j/%j_err.log"
        srun_stdout = log_folder / "%j/%j_%t_out.log"
        srun_stderr = log_folder / "%j/%j_%t_err.log"
    else:
        sbatch_kwargs["array"] = (
            f"0-{array_count - 1}%{min(array_count, array_parallelism)}"
        )
        stdout = log_folder / "%A_%a/%A_%a_out.log"
        stderr = log_folder / "%A_%a/%A_%a_err.log"
        srun_stdout = log_folder / "%A_%a/%A_%a_%t_out.log"
        srun_stderr = log_folder / "%A_%a/%A_%a_%t_err.log"

    sbatch_kwargs["output"] = str(stdout)
    if not stderr_to_stdout:
        sbatch_kwargs["error"] = str(stderr)

    srun_kwargs["output"] = str(srun_stdout)
    if not stderr_to_stdout:
        srun_kwargs["error"] = str(srun_stderr)

    signal_delay_s = sbatch_kwargs.pop("signal_delay_s", 120)

    lines = ["#!/bin/bash", "", "# Parameters"]
    for key, value in sbatch_kwargs.items():
        lines.append(as_sbatch_flag(key, value))

    # environment setup:
    if setup is not None:
        lines += ["", "# setup"] + setup

    srun = ["srun", "--unbuffered"]
    for key, value in srun_kwargs.items():
        srun.append(as_srun_args(key, value))

    srun.extend(command)
    srun = shlex.join(srun)
    lines += ["", "# command", srun, ""]

    # environment teardown:
    if teardown is not None:
        lines += ["", "# teardown"] + teardown
    return "\n".join(lines)


def as_sbatch_flag(key: str, value) -> str:
    key = key.replace("_", "-")
    if value is True:
        return f"#SBATCH --{key}"

    value = shlex.quote(str(value))
    return f"#SBATCH --{key}={value}"


def as_srun_args(key: str, value) -> str:
    key = key.replace("_", "-")
    if value is True:
        return f"--{key}"

    value = shlex.quote(str(value))
    return f"--{key}={value}"


@dataclass
class SlurmJobEnvironment:
    @cached_property
    def job_id(self) -> int:
        """Fetches the job id from the environment variable set by slurm"""
        try:
            return int(os.environ["SLURM_JOB_ID"])
        except KeyError:
            raise RuntimeError(
                "Could not find SLURM_JOB_ID in environment variables. "
                "Make sure that the job is running inside a slurm allocation."
            )

    @cached_property
    def array_job_id(self) -> int:
        """Fetches the job id from the environment variable set by slurm."""
        try:
            return int(os.environ["SLURM_ARRAY_JOB_ID"])
        except KeyError:
            raise RuntimeError(
                "Could not find SLURM_ARRAY_JOB_ID in environment variables. "
                "Make sure that the job is running inside a slurm allocation."
            )

    @cached_property
    def array_index(self) -> int:
        """Fetches the array id from the environment variable set by slurm."""
        try:
            return int(os.environ["SLURM_ARRAY_TASK_ID"])
        except KeyError:
            raise RuntimeError(
                "Could not find SLURM_ARRAY_TASK_ID in environment variables. "
                "Make sure that the job is running inside a slurm allocation."
            )

    @cached_property
    def task_index(self) -> int:
        """Fetches the task index from the environment variable set by slurm.

        Note: SLURM_LOCALID is the index of the task on the current node,
        whereas SLURM_PROCID is the global index of the task across all
        nodes. (only relevant for multi-node jobs).
        """
        try:
            return int(os.environ["SLURM_PROCID"])
        except KeyError:
            raise RuntimeError(
                "Could not find SLURM_PROCID in environment variables. "
                "Make sure that the job is running inside a slurm allocation."
            )

    @cached_property
    def n_tasks(self) -> int:
        """Fetches the number of tasks per job from the environment variable
        set by slurm.

        Note: An array job with N subjobs will launch this many tasks per subjob.
        """
        try:
            return int(os.environ.get("SLURM_NTASKS", 1))
        except KeyError:
            raise RuntimeError(
                "Could not find SLURM_NTASKS in environment variables. "
                "Make sure that the job is running inside a slurm allocation."
            )

    @property
    def global_rank(self) -> int:
        """Fetches the global rank of the task from the environment variable set by slurm."""
        return self.array_index * self.n_tasks + self.task_index
