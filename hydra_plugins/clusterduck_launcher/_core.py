import logging
from typing import Any, Dict, Sequence

from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, configure_log, run_job, setup_globals
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, open_dict

from ._slurm import SlurmJobEnvironment

log = logging.getLogger(__name__)


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

    # setup hydra logging for operations before the job starts
    # TODO: verify
    configure_log(config.hydra.hydra_logging, config.hydra.verbose)

    # TODO: configure signal handlers?

    slurm = SlurmJobEnvironment()
    task_id = slurm.global_rank + initial_job_idx
    overrides = job_overrides[task_id]

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
