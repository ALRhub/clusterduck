from __future__ import annotations

import os
import re
import shlex
import typing as tp
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


# pylint: disable=too-many-arguments,unused-argument, too-many-locals
def make_sbatch_string(
    command: str,
    folder: tp.Union[str, Path],
    job_name: str = "submitit",
    partition: tp.Optional[str] = None,
    time: int = 5,
    nodes: int = 1,
    ntasks_per_node: tp.Optional[int] = None,
    cpus_per_task: tp.Optional[int] = None,
    cpus_per_gpu: tp.Optional[int] = None,
    num_gpus: tp.Optional[int] = None,  # legacy
    gpus_per_node: tp.Optional[int] = None,
    gpus_per_task: tp.Optional[int] = None,
    qos: tp.Optional[str] = None,  # quality of service
    setup: tp.Optional[tp.List[str]] = None,
    teardown: tp.Optional[tp.List[str]] = None,
    mem: tp.Optional[str] = None,
    mem_per_gpu: tp.Optional[str] = None,
    mem_per_cpu: tp.Optional[str] = None,
    signal_delay_s: int = 90,
    comment: tp.Optional[str] = None,
    constraint: tp.Optional[str] = None,
    exclude: tp.Optional[str] = None,
    account: tp.Optional[str] = None,
    gres: tp.Optional[str] = None,
    mail_type: tp.Optional[str] = None,
    mail_user: tp.Optional[str] = None,
    nodelist: tp.Optional[str] = None,
    dependency: tp.Optional[str] = None,
    exclusive: tp.Optional[tp.Union[bool, str]] = None,
    array_parallelism: int = 256,
    wckey: str = "submitit",
    stderr_to_stdout: bool = False,
    map_count: tp.Optional[int] = None,  # used internally
    additional_parameters: tp.Optional[tp.Dict[str, tp.Any]] = None,
    srun_args: tp.Optional[tp.Iterable[str]] = None,
    use_srun: bool = True,
) -> str:
    """Creates the content of an sbatch file with provided parameters

    Parameters
    ----------
    See slurm sbatch documentation for most parameters:
    https://slurm.schedmd.com/sbatch.html

    Below are the parameters that differ from slurm documentation:

    folder: str/Path
        folder where print logs and error logs will be written
    signal_delay_s: int
        delay between the kill signal and the actual kill of the slurm job.
    setup: list
        a list of command to run in sbatch before running srun
    teardown: list
        a list of command to run in sbatch after running srun
    map_size: int
        number of simultaneous map/array jobs allowed
    additional_parameters: dict
        Forces any parameter to a given value in sbatch. This can be useful
        to add parameters which are not currently available in submitit.
        Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}
    srun_args: List[str]
        Add each argument in the list to the srun call

    Raises
    ------
    ValueError
        In case an erroneous keyword argument is added, a list of all eligible parameters
        is printed, with their default values
    """
    nonslurm = [
        "nonslurm",
        "folder",
        "command",
        "map_count",
        "array_parallelism",
        "additional_parameters",
        "setup",
        "teardown",
        "signal_delay_s",
        "stderr_to_stdout",
        "srun_args",
        "use_srun",  # if False, un python directly in sbatch instead of through srun
    ]
    parameters = {
        k: v for k, v in locals().items() if v is not None and k not in nonslurm
    }
    # rename and reformat parameters
    parameters["signal"] = f"{SlurmJobEnvironment.USR_SIG}@{signal_delay_s}"
    if num_gpus is not None:
        warnings.warn(
            '"num_gpus" is deprecated, please use "gpus_per_node" instead (overwritting with num_gpus)'
        )
        parameters["gpus_per_node"] = parameters.pop("num_gpus", 0)
    if "cpus_per_gpu" in parameters and "gpus_per_task" not in parameters:
        warnings.warn(
            '"cpus_per_gpu" requires to set "gpus_per_task" to work (and not "gpus_per_node")'
        )
    # add necessary parameters
    paths = utils.JobPaths(folder=folder)
    stdout = str(paths.stdout)
    stderr = str(paths.stderr)
    # Job arrays will write files in the form  <ARRAY_ID>_<ARRAY_TASK_ID>_<TASK_ID>
    if map_count is not None:
        assert isinstance(map_count, int) and map_count
        parameters["array"] = f"0-{map_count - 1}%{min(map_count, array_parallelism)}"
        stdout = stdout.replace("%j", "%A_%a")
        stderr = stderr.replace("%j", "%A_%a")
    parameters["output"] = stdout.replace("%t", "0")
    if not stderr_to_stdout:
        parameters["error"] = stderr.replace("%t", "0")
    parameters["open-mode"] = "append"
    if additional_parameters is not None:
        parameters.update(additional_parameters)
    # now create
    lines = ["#!/bin/bash", "", "# Parameters"]
    for k in sorted(parameters):
        lines.append(as_sbatch_flag(k, parameters[k]))
    # environment setup:
    if setup is not None:
        lines += ["", "# setup"] + setup
    # commandline (this will run the function and args specified in the file provided as argument)
    # We pass --output and --error here, because the SBATCH command doesn't work as expected with a filename pattern

    if use_srun:
        # using srun has been the only option historically,
        # but it's not clear anymore if it is necessary, and using it prevents
        # jobs from scheduling other jobs
        stderr_flags = [] if stderr_to_stdout else ["--error", stderr]
        if srun_args is None:
            srun_args = []
        srun_cmd = shlex.join(
            ["srun", "--unbuffered", "--output", stdout, *stderr_flags, *srun_args]
        )
        command = " ".join((srun_cmd, command))

    lines += [
        "",
        "# command",
        "export SUBMITIT_EXECUTOR=slurm",
        # The input "command" is supposed to be a valid shell command
        command,
        "",
    ]

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


def get_job_id_from_submission_command(string: bytes | str) -> str:
    """Returns the job ID from the output of sbatch string"""
    if not isinstance(string, str):
        string = string.decode()
    output = re.search(r"job (?P<id>[0-9]+)", string)
    if output is None:
        raise ValueError(
            f'Could not make sense of sbatch output "{string}"\n'
            "Job instance will not be able to fetch status\n"
            "(you may however set the job job_id manually if needed)"
        )
    return output.group("id")


@dataclass
class JobEnvironment:
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
    def array_id(self) -> int:
        """Fetches the array id from the environment variable set by slurm"""
        try:
            return int(os.environ["SLURM_ARRAY_TASK_ID"])
        except KeyError:
            raise RuntimeError(
                "Could not find SLURM_ARRAY_TASK_ID in environment variables. "
                "Make sure that the job is running inside a slurm allocation."
            )

    @cached_property
    def task_id(self) -> int:
        """Fetches the node id from the environment variable set by slurm"""
        try:
            return int(os.environ["SLURM_PROCID"])
        except KeyError:
            raise RuntimeError(
                "Could not find SLURM_PROCID in environment variables. "
                "Make sure that the job is running inside a slurm allocation."
            )

    @cached_property
    def tasks_per_node(self) -> int:
        """Fetches the number of tasks per node from the environment variable set by slurm"""
        try:
            return int(os.environ.get("SLURM_NTASKS", 1))
        except KeyError:
            raise RuntimeError(
                "Could not find SLURM_NTASKS in environment variables. "
                "Make sure that the job is running inside a slurm allocation."
            )
