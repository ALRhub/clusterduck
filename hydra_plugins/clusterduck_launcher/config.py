from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class ClusterDuckLauncherConf:

    _target_: str = (
        "hydra_plugins.clusterduck_launcher.clusterduck_launcher.ClusterDuckLauncher"
    )

    ## Slurm Settings (see https://slurm.schedmd.com/sbatch.html)
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

    ## Clusterduck Settings
    # Folder where the submission script, pickle and slurm logs will be stored.
    log_folder: str = "${hydra.sweep.dir}/slurm"
    # If `True`, redirect the standard error of the job to the same file as standard output.
    stderr_to_stdout: bool = True
    # Throttle array jobs to only have this many jobs running at once
    array_parallelism: int = 256
    # Any additional arguments that should be passed to sbatch
    sbatch_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Any additional arguments that should be passed to srun
    srun_kwargs: Dict[str, Any] = field(default_factory=dict)
    # A list of commands to run in sbatch befure running srun
    setup: Optional[List[str]] = None
    # A list of commands to run in sbatch after running srun
    teardown: Optional[List[str]] = None
    # If these environment variables are set and there are multiple tasks, clusterduck will create a subfolder for each task and set the environment variable to point to that subfolder. This is useful for avoiding conflicts between tasks when writing temporary files.
    tmpdir_vars: Optional[List[str]] = field(default_factory=lambda: ["TMP", "TMPDIR"])

    ## Debugging Settings
    # If `True`, the python command will be launched by srun. If `False`, the python command is run directly inside the job.
    use_srun: bool = True
    # If `False`, create the submission file but do not actually submit it.
    do_submit: bool = True
    # If `True`, this is a shortcut for `use_srun=False` and `do_submit=False`. This generates a script that can be executed locally as a standard shell script.
    local_debug: bool = False


# finally, register the config type:
ConfigStore.instance().store(
    group="hydra/launcher",
    name="clusterduck_slurm",
    node=ClusterDuckLauncherConf(),
    provider="clusterduck",
)
