# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from multiprocessing import Manager, Pool, current_process, Queue
from joblib import Parallel, delayed, wrap_non_picklable_objects  # type: ignore
import joblib

from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, filter_overrides, run_job, setup_globals
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf, open_dict

from .config import BaseQueueConf

log = logging.getLogger(__name__)



def execute_job(
    idx: int,
    overrides: Sequence[str],
    hydra_context: HydraContext,
    config: DictConfig,
    task_function: TaskFunction,
    singleton_state: Dict[Any, Any],
) -> JobReturn:
    """Calls `run_job` in parallel"""
    setup_globals()
    Singleton.set_state(singleton_state)

    sweep_config = hydra_context.config_loader.load_sweep_config(config, list(overrides))
    sweep_config = OmegaConf.create(sweep_config)
    with open_dict(sweep_config):
        sweep_config.hydra.job.id = f"{sweep_config.hydra.job.name}_{idx}"
        sweep_config.hydra.job.num = idx
    HydraConfig.instance().set_config(sweep_config)

    ret = run_job(
        hydra_context=hydra_context,
        config=sweep_config,
        task_function=task_function,
        job_dir_key="hydra.sweep.dir",
        job_subdir_key="hydra.sweep.subdir",
    )

    return ret


class BaseClusterDuckLauncher(Launcher):
    _EXECUTOR = "abstract"

    def __init__(
            self,
            num_of_overrides_per_job: int = 1,
            parallel_executions_in_job: int = 1,
            exclusive_gpu_per_execution: bool = False,
            **params: Any
    ) -> None:
        self.params = {}
        for k, v in params.items():
            if OmegaConf.is_config(v):
                v = OmegaConf.to_container(v, resolve=True)
            self.params[k] = v

        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.sweep_configs: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

        self.num_of_overrides_per_job = num_of_overrides_per_job
        self.parallel_executions_in_job = parallel_executions_in_job
        self.exclusive_gpu_per_execution = exclusive_gpu_per_execution


    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def __call__(
        self,
        sweep_overrides_list: List[List[str]],
        job_dir_key: str,
        job_num: int,
        job_id: str,
        singleton_state: Dict[type, Singleton],
    ) -> JobReturn:
        from dask.distributed import Client as DaskClient
        from dask_cuda import LocalCUDACluster


        assert self.hydra_context is not None
        assert self.config is not None
        assert self.task_function is not None

        Singleton.set_state(singleton_state)
        setup_globals()

        cluster = LocalCUDACluster()
        client = DaskClient(cluster)

        assert self.parallel_executions_in_job <= len(cluster.workers), f"Number of workers {len(cluster.workers)} is less than parallel_executions_in_job {self.parallel_executions_in_job}"

        with joblib.parallel_config(backend="dask"):
            runs = Parallel(
                n_jobs=self.parallel_executions_in_job,
                prefer="processes",
            )(
            delayed(execute_job)(
                    job_num,
                    overrides,
                    self.hydra_context,
                    self.config,
                    self.task_function,
                    singleton_state,
                )
                for idx, overrides in enumerate(sweep_overrides_list)
            )

        assert isinstance(runs, List)
        for run in runs:
            assert isinstance(run, JobReturn)
        return runs

    def checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        """Resubmit the current callable at its current state with the same initial arguments."""
        # lazy import to ensure plugin discovery remains fast
        import submitit

        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        # lazy import to ensure plugin discovery remains fast
        import submitit
        import math

        assert self.config is not None

        num_jobs = math.ceil(len(job_overrides)/self.num_of_overrides_per_job)
        assert num_jobs > 0
        params = self.params
        # build executor
        init_params = {"folder": self.params["submitit_folder"]}
        specific_init_keys = {"max_num_timeout"}

        init_params.update(
            **{
                f"{self._EXECUTOR}_{x}": y
                for x, y in params.items()
                if x in specific_init_keys
            }
        )
        init_keys = specific_init_keys | {"submitit_folder"}
        executor = submitit.AutoExecutor(cluster=self._EXECUTOR, **init_params)

        # specify resources/parameters
        baseparams = set(OmegaConf.structured(BaseQueueConf).keys())
        params = {
            x if x in baseparams else f"{self._EXECUTOR}_{x}": y
            for x, y in params.items()
            if x not in init_keys
        }
        executor.update_parameters(**params)

        log.info(
            f"Submitit '{self._EXECUTOR}' sweep output dir : "
            f"{self.config.hydra.sweep.dir}"
        )
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: List[Any] = []
        for idx in range(num_jobs):
            job_overrides_sublist = job_overrides[idx*self.num_of_overrides_per_job:(idx+1)*self.num_of_overrides_per_job]
            assert len(job_overrides_sublist) > 0 and len(job_overrides_sublist) <= self.num_of_overrides_per_job

            filtered_job_overrides_sublist = [" ".join(filter_overrides(overrides)) for overrides in job_overrides_sublist]
            log.info(f"Launching {filtered_job_overrides_sublist} in job with id {initial_job_idx+idx}")

            job_params.append(
                (
                    job_overrides_sublist,
                    "hydra.sweep.dir",
                    idx+initial_job_idx,
                    f"job_id_for_{idx}",
                    Singleton.get_state(),
                )
            )

        jobs = executor.map_array(self, *zip(*job_params))
        results = [j.result() for j in jobs]
        results = [item for sublist in results for item in sublist]
        return results
        # return [j.results()[0] for j in jobs]


class LocalLauncher(BaseClusterDuckLauncher):
    _EXECUTOR = "local"


class SlurmLauncher(BaseClusterDuckLauncher):
    _EXECUTOR = "slurm"
