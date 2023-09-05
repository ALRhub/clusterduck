from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

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
from omegaconf import DictConfig, OmegaConf, open_dict

from .config import BaseQueueConf
from .process_manager import ProcessManager

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.connection import Connection

log = logging.getLogger("clusterduck")


class BaseClusterDuckLauncher(Launcher):
    _EXECUTOR = "abstract"

    def __init__(
        self,
        num_of_overrides_per_job: int,
        parallel_executions_in_job: int,
        wait_for_completion: bool,
        resources_config: DictConfig,
        **params: Any,
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
        self.wait_for_completion = wait_for_completion
        self.resources_config = resources_config

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

    def process_manager_proxy(
        self,
        sweep_overrides_list: List[List[str]],
        job_dir_key: str,
        job_nums: range,
        job_id: str,
        singleton_state: Dict[type, Singleton],
    ) -> list[JobReturn]:
        process_manager = ProcessManager(parallel_executions_in_job=self.parallel_executions_in_job, target_fn=self, resources_config=self.config.hydra.launcher.resources_config)
        return process_manager.run(
            sweep_overrides_list,
            job_dir_key,
            job_nums,
            job_id,
            singleton_state,
        )

    def __call__(
        self,
        sweep_overrides: List[str],
        job_dir_key: str,
        job_num: int,
        job_id: str,
        singleton_state: Dict[type, Singleton],
        pipe: Connection,
    ) -> None:
        # lazy import to ensure plugin discovery remains fast
        import cloudpickle
        import submitit

        assert self.hydra_context is not None
        assert self.config is not None
        assert self.task_function is not None

        Singleton.set_state(singleton_state)
        setup_globals()
        sweep_config = self.hydra_context.config_loader.load_sweep_config(
            self.config, sweep_overrides
        )

        with open_dict(sweep_config.hydra.job) as job:
            # Populate new job variables
            job.id = submitit.JobEnvironment().job_id  # type: ignore
            sweep_config.hydra.job.num = job_num


        # TODO: separate clusterduck logging (global across overrides) from job logging (override-local)
        logger = logging.getLogger("clusterduck")
        logger.info(f"Running job {job_num}")
        ret = run_job(
            hydra_context=self.hydra_context,
            task_function=self.task_function,
            config=sweep_config,
            job_dir_key=job_dir_key,
            job_subdir_key="hydra.sweep.subdir",
        )
        logger.info(f"Job {job_num} completed.")
        pipe.send(cloudpickle.dumps(ret))

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

        num_jobs = math.ceil(len(job_overrides) / self.num_of_overrides_per_job)
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
                idx
                * self.num_of_overrides_per_job : (idx + 1)
                * self.num_of_overrides_per_job
            ]
            assert (
                len(job_overrides_sublist) > 0
                and len(job_overrides_sublist) <= self.num_of_overrides_per_job
            )

            filtered_job_overrides_sublist = [
                " ".join(filter_overrides(overrides))
                for overrides in job_overrides_sublist
            ]
            log.info(f"Launching {filtered_job_overrides_sublist} in job with id {idx}")

            job_params.append(
                (
                    job_overrides_sublist,
                    "hydra.sweep.dir",  # job_dir_key
                    range(  # job_nums
                        idx * self.num_of_overrides_per_job + initial_job_idx,
                        (idx + 1) * self.num_of_overrides_per_job + initial_job_idx,
                    ),
                    f"job_id_for_{idx}",  # job_id
                    Singleton.get_state(),  # singleton_state
                )
            )

        # launch all
        jobs = executor.map_array(self.process_manager_proxy, *zip(*job_params))

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
