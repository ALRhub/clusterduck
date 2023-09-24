# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class BaseQueueConf:
    """Configuration shared by all executors"""

    submitit_folder: str = "${hydra.sweep.dir}/.submitit/%j"

    # maximum time for the job in minutes
    timeout_min: int = 60
    # number of cpus to use for each task
    cpus_per_task: Optional[int] = None
    # number of gpus to use on each node
    gpus_per_node: Optional[int] = None
    # number of tasks to spawn on each node
    tasks_per_node: int = 1
    # memory to reserve for the job on each node (in GB)
    mem_gb: Optional[int] = None
    # number of nodes to use for the job
    nodes: int = 1
    # name of the job
    name: str = "${hydra.job.name}"
    # redirect stderr to stdout
    stderr_to_stdout: bool = False

    # Following parameters are clusterduck specific
    # number of tasks (i.e. hydra jobs) to spawn in parallel on each node
    parallel_runs_per_node: int = 1
    # number of tasks (i.e. hydra jobs) to complete in total in each slurm job
    # leave None to execute all overrides in a single slurm job
    total_runs_per_node: Optional[int] = None
    # wait until slurm jobs finish before exiting Python script
    wait_for_completion: bool = False
    # resources that should be divided up among parallel task executions
    # e.g. resources_config: [cuda, cpu]
    # additional configuration for resources should be included as a DictConfig
    # beneath the resource name
    # e.g.
    # resources_config:
    #   - stagger:
    #       delay: 10
    resources_config: dict[str, Optional[dict]] = field(default_factory=dict)
    # whether to print debug statements to the SLURM stdout log file
    verbose: bool = False


@dataclass
class SlurmQueueConf(BaseQueueConf):
    """Slurm configuration overrides and specific parameters"""

    _target_: str = "hydra_plugins.clusterduck_launcher.clusterduck_launcher.ClusterDuckSlurmLauncher"

    # Params are used to configure sbatch, for more info check:
    # https://github.com/facebookincubator/submitit/blob/main/submitit/slurm/slurm.py

    # Following parameters are slurm specific
    # More information: https://slurm.schedmd.com/sbatch.html
    #
    # slurm partition to use on the cluster
    partition: Optional[str] = None
    qos: Optional[str] = None
    comment: Optional[str] = None
    constraint: Optional[str] = None
    exclude: Optional[str] = None
    gres: Optional[str] = None
    cpus_per_gpu: Optional[int] = None
    gpus_per_task: Optional[int] = None
    mem_per_gpu: Optional[str] = None
    mem_per_cpu: Optional[str] = None
    account: Optional[str] = None

    # Following parameters are submitit specifics
    #
    # USR1 signal delay before timeout
    signal_delay_s: int = 120
    # Maximum number of retries on job timeout.
    # Change this only after you confirmed your code can handle re-submission
    # by properly resuming from the latest stored checkpoint.
    # check the following for more info on slurm_max_num_timeout
    # https://github.com/facebookincubator/submitit/blob/main/docs/checkpointing.md
    max_num_timeout: int = 0
    # Useful to add parameters which are not currently available in the plugin.
    # Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}
    additional_parameters: Dict[str, Any] = field(default_factory=dict)
    # Maximum number of jobs running in parallel
    array_parallelism: int = 256
    # A list of commands to run in sbatch befure running srun
    setup: Optional[List[str]] = None
    # Any additional arguments that should be passed to srun
    srun_args: Optional[List[str]] = None


@dataclass
class LocalQueueConf(BaseQueueConf):
    _target_: str = "hydra_plugins.clusterduck_launcher.clusterduck_launcher.ClusterDuckLocalLauncher"


# finally, register two different choices:
ConfigStore.instance().store(
    group="hydra/launcher",
    name="clusterduck_local",
    node=LocalQueueConf(),
    provider="clusterduck",
)


ConfigStore.instance().store(
    group="hydra/launcher",
    name="clusterduck_slurm",
    node=SlurmQueueConf(),
    provider="clusterduck",
)
