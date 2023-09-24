from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hydra.core.singleton import Singleton
from hydra.core.utils import (
    JobReturn,
    JobStatus,
    filter_overrides,
    run_job,
    setup_globals,
)
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from .config import BaseQueueConf

log = logging.getLogger(__name__)


class BaseClusterDuckLauncher(Launcher):
    _EXECUTOR = "abstract"

    def __init__(
        self,
        parallel_runs_per_node: int,
        total_runs_per_node: int | None,
        wait_for_completion: bool,
        resources_config: ListConfig,
        verbose: bool,
        **params: Any,
    ) -> None:
        self.n_workers = parallel_runs_per_node
        self.total_runs_per_node = total_runs_per_node
        self.wait_for_completion = wait_for_completion
        self.resources_config = resources_config
        self.verbose = verbose

        # parameters used by submitit
        self.params = {}
        for k, v in params.items():
            if OmegaConf.is_config(v):
                v = OmegaConf.to_container(v, resolve=True)
            self.params[k] = v

        self.config: Optional[DictConfig] = None
        self.task_function: Optional[bytes] = None
        self.sweep_configs: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        import cloudpickle

        self.config = config
        self.hydra_context = hydra_context
        self.task_function = cloudpickle.dumps(task_function)

    def run_workers(
        self,
        sweep_overrides_list: List[List[str]],
        job_dir_key: str,
        job_nums: range,
        job_id: str,
        singleton_state: Dict[type, Singleton],
    ) -> list[JobReturn]:
        from ._logging import configure_log, get_logger
        from ._resources import ResourcePool
        from ._worker_pool import WorkerPool

        configure_log(self.verbose)
        logger = get_logger()

        kwargs_list = [
            dict(
                sweep_overrides=sweep_overrides,
                job_dir_key=job_dir_key,
                job_num=job_num,
                job_id=job_id,
                singleton_state=singleton_state,
            )
            for sweep_overrides, job_num in zip(sweep_overrides_list, job_nums)
        ]

        # TODO: make sure this is removed
        import sys

        if "torch" in sys.modules:
            logger.debug(
                "Package `torch` has already been imported before resource creation."
            )
        else:
            logger.debug(
                "Package `torch` has not yet been imported before resource creation."
            )

        resource_pools = []
        for resource in self.resources_config:
            if isinstance(resource, str):
                resource_pools.append(
                    ResourcePool(kind=resource, n_workers=self.n_workers)
                )
            elif isinstance(resource, DictConfig):
                items = list(resource.items())
                assert len(items) == 1
                kind, kwargs = items[0]
                resource_pools.append(
                    ResourcePool(kind=kind, n_workers=self.n_workers, **kwargs)
                )
            else:
                raise ValueError(f"Unexpected resource configuration {resource}")

        if "torch" in sys.modules:
            logger.debug(
                "Package `torch` has already been imported after resource creation."
            )
        else:
            logger.debug(
                "Package `torch` has not yet been imported after resource creation."
            )

        process_manager = WorkerPool(
            n_workers=self.n_workers,
            resource_pools=resource_pools,
            # if not using fork, all hydra objects need to be serialized with cloudpickle
            start_method="fork",
        )
        results = process_manager.execute(
            target=self,
            kwargs_list=kwargs_list,
        )

        exceptions = [
            result.return_value
            for result in results
            if result.status == JobStatus.FAILED
        ]
        if exceptions:
            # TODO: ExceptionGroup exists in Python 3.11 and up
            raise RuntimeError(
                f"{len(exceptions)} workers failed due to errors!"
            ) from exceptions[0]

        return results

    def __call__(
        self,
        sweep_overrides: List[str],
        job_dir_key: str,
        job_num: int,
        job_id: str,
        singleton_state: Dict[type, Singleton],
    ) -> JobReturn:
        # lazy import to ensure plugin discovery remains fast
        import pickle

        import submitit

        from ._logging import configure_log, get_logger

        configure_log(self.verbose)
        logger = get_logger()

        assert self.hydra_context is not None
        assert self.config is not None
        assert self.task_function is not None

        task_function = pickle.loads(self.task_function)

        Singleton.set_state(singleton_state)
        setup_globals()
        sweep_config = self.hydra_context.config_loader.load_sweep_config(
            self.config, sweep_overrides
        )

        with open_dict(sweep_config.hydra.job) as job:
            # Populate new job variables
            job.id = submitit.JobEnvironment().job_id  # type: ignore
            sweep_config.hydra.job.num = job_num

        logger.info(
            f"Running job {job_num} with overrides {' '.join(filter_overrides(sweep_overrides))}"
        )
        ret = run_job(
            hydra_context=self.hydra_context,
            task_function=task_function,
            config=sweep_config,
            job_dir_key=job_dir_key,
            job_subdir_key="hydra.sweep.subdir",
        )
        logger.info(f"Job {job_num} completed.")
        return ret

    def checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        """Resubmit the current callable at its current state with the same initial arguments."""
        # lazy import to ensure plugin discovery remains fast
        import submitit

        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        # lazy import to ensure plugin discovery remains fast
        import math

        import submitit

        assert self.config is not None

        total_runs_per_node = self.total_runs_per_node or len(job_overrides)

        num_jobs = math.ceil(len(job_overrides) / total_runs_per_node)
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
            f"Clusterduck '{self._EXECUTOR}' sweep output dir : "
            f"{self.config.hydra.sweep.dir}"
        )
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: List[Any] = []
        for idx in range(num_jobs):
            job_overrides_sublist = job_overrides[
                idx * total_runs_per_node : (idx + 1) * total_runs_per_node
            ]
            assert (
                len(job_overrides_sublist) > 0
                and len(job_overrides_sublist) <= total_runs_per_node
            )

            log.info(f"\tJob #{idx} :")
            for overrides in job_overrides_sublist:
                lst = " ".join(filter_overrides(overrides))
                log.info(f"\t\t{lst}")

            job_params.append(
                (
                    job_overrides_sublist,
                    "hydra.sweep.dir",  # job_dir_key
                    range(  # job_nums
                        idx * total_runs_per_node + initial_job_idx,
                        (idx + 1) * total_runs_per_node + initial_job_idx,
                    ),
                    f"job_id_for_{idx}",  # job_id
                    Singleton.get_state(),  # singleton_state
                )
            )

        # launch all
        jobs = executor.map_array(self.run_workers, *zip(*job_params))

        if self.wait_for_completion:
            return [result for j in jobs for result in j.results()[0]]
        else:
            # we do our best to emulate what hydra.core.utils.run_job would return by doing most
            # of what BaseSubmititLauncher.__call__ does but with a no-op task_function
            assert self.hydra_context is not None
            assert self.config is not None
            assert self.task_function is not None

            no_op = lambda config: None
            job_nums = range(initial_job_idx, initial_job_idx + len(job_overrides))
            results: list[JobReturn] = []
            for job_num, override in zip(job_nums, job_overrides):
                sweep_config = self.hydra_context.config_loader.load_sweep_config(
                    self.config, override
                )

                with open_dict(sweep_config.hydra.job) as job:
                    # Populate new job variables
                    # Cannot set job ID to slurm job ID as we are not inside slurm job
                    sweep_config.hydra.job.num = job_num

                result = run_job(
                    hydra_context=self.hydra_context,
                    task_function=no_op,
                    config=sweep_config,
                    job_dir_key="hydra.sweep.dir",
                    job_subdir_key="hydra.sweep.subdir",
                    configure_logging=False,
                )

                results.append(result)
            return results


class ClusterDuckLocalLauncher(BaseClusterDuckLauncher):
    _EXECUTOR = "local"


class ClusterDuckSlurmLauncher(BaseClusterDuckLauncher):
    _EXECUTOR = "slurm"
