from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class ClusterDuckLauncherConf:

    _target_: str = (
        "hydra_plugins.clusterduck_launcher.clusterduck_launcher.ClusterDuckLauncher"
    )

    log_folder: str = "${hydra.sweep.dir}/slurm"

    # maximum time for the job in minutes
    timeout_min: int = 60
    # number of cpus to use for each task
    cpus_per_task: Optional[int] = None
    # number of gpus to use on each node
    gpus_per_node: Optional[int] = None
    # number of tasks to spawn on each node
    tasks_per_node: Optional[int] = None
    # memory to reserve for the job on each node (in GB)
    mem_gb: Optional[int] = None
    # number of nodes to use for the job
    nodes: int = 1
    # name of the job
    name: str = "${hydra.job.name}"
    # redirect stderr to stdout
    stderr_to_stdout: bool = False

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
    # Maximum number of jobs running in parallel
    array_parallelism: int = 256
    # A list of commands to run in sbatch befure running srun
    setup: Optional[List[str]] = None
    sbatch_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Any additional arguments that should be passed to srun
    srun_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Following parameters are clusterduck specific
    # number of tasks (i.e. hydra jobs) to run in parallel on each node
    parallel_tasks_per_job: int = 1
    # number of tasks (i.e. hydra jobs) to run sequentially in each resource slot
    sequential_tasks_per_job: int = 1
    # whether to use srun to launch the command or just run it directly in sbatch (useful for debugging)
    use_srun: bool = True
    # whether to actually submit the job using sbatch or just generate the files
    do_submit: bool = True
    # shortcut for use_srun=False and do_submit=False, which generates a script that can be executed locally for debugging
    local_debug: bool = False


# finally, register the config type:
ConfigStore.instance().store(
    group="hydra/launcher",
    name="clusterduck_slurm",
    node=ClusterDuckLauncherConf(),
    provider="clusterduck",
)
