import logging
from typing import Any, Dict, Sequence
import os

from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, configure_log, run_job, setup_globals
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, open_dict

from ._slurm import SlurmJobEnvironment

log = logging.getLogger("clusterduck")


def execute_job(
    initial_job_idx: int,
    job_overrides: Sequence[Sequence[str]],
    hydra_context: HydraContext,
    config: DictConfig,
    task_function: TaskFunction,
    singleton_state: Dict[Any, Any],
) -> JobReturn:

    setup_globals()
    Singleton.set_state(singleton_state)

    # Setup hydra logging for operations before the job starts. Because the
    # hydra.runtime.output_dir and other hydra variables are not yet populated,
    # this only logs to the console and not to any files. However, this is
    # still captured in the slurm logs.
    configure_log(config.hydra.hydra_logging, config.hydra.verbose)

    slurm = SlurmJobEnvironment()
    task_id = slurm.global_rank + initial_job_idx
    overrides = job_overrides[task_id]
    log.debug(f"Executing task with global rank {task_id}")

    # TODO: configure signal handlers?
    os.environ["RANK"] = "0"

    sweep_config = hydra_context.config_loader.load_sweep_config(
        config, list(overrides)
    )
    with open_dict(sweep_config.hydra.job) as job:
        # Populate new job variables
        job.id = slurm.job_id
        sweep_config.hydra.job.num = task_id
    HydraConfig.instance().set_config(sweep_config)

    return run_job(
        task_function=task_function,
        config=sweep_config,
        job_dir_key="hydra.sweep.dir",
        job_subdir_key="hydra.sweep.subdir",
        hydra_context=hydra_context,
    )
