from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from ._wrapped_task import WrappedTaskFunction
    from ._resources import Resource

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

log = logging.getLogger("clusterduck.launcher")


class BaseClusterDuckLauncher(Launcher):
    _EXECUTOR = "abstract"

    def __init__(
        self,
        parallel_runs_per_node: int,
        total_runs_per_node: int | None,
        wait_for_completion: bool,
        resources_config: DictConfig,
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
        self.task_function: Optional[WrappedTaskFunction] = None
        self.sweep_configs: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        from ._wrapped_task import WrappedTaskFunction

        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def run_workers(
        self,
        sweep_overrides_list: List[List[str]],
        job_dir_key: str,
        job_nums: range,
        job_id: str,
        singleton_state: Dict[type, Singleton],
    ) -> list[JobReturn]:
        """This method runs inside the SLURM job and starts worker processes to run hydra jobs.
        At this point, no Hydra logging has been configured, only submitit's logging, which logs
        to stdout and stderr.
        """
        from ._logging import configure_log
        from ._resources import create_resource_pools_from_cfg
        from ._worker_pool import WorkerPool

        configure_log(self.verbose)

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

        resource_pools = create_resource_pools_from_cfg(
            self.resources_config, self.n_workers
        )

        process_manager = WorkerPool(
            n_workers=self.n_workers,
            resource_pools=resource_pools,
            # we use fork because many Hydra objects are only pickable with cloudpickle
            start_method="fork",
        )
        results = process_manager.execute(
            target_fn=self,
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
        job_overrides_sublist: Sequence[Sequence[str]],
        job_dir_key: str,
        initial_job_num: int,
        job_id: str,
        singleton_state: Dict[type, Singleton],
    ) -> JobReturn:
        """This method runs inside the SLURM job inside a fresh process forked by `run_workers`.
        When this method starts, no logging of any kind has been configured. Hydra job logging
        is configured inside `run_job`, so we delay important operations like applying resource
        configurations and unpickling the task function until the __call__ method of the
        WrappedTaskFunction, which is called from there.
        """
        # lazy import to ensure plugin discovery remains fast
        import submitit

        assert self.hydra_context is not None
        assert self.config is not None
        assert self.task_function is not None

        Singleton.set_state(singleton_state)
        setup_globals()

        task_id = int(os.environ["SLURM_LOCALID"])
        sweep_overrides = job_overrides_sublist[task_id]

        sweep_config = self.hydra_context.config_loader.load_sweep_config(
            self.config, list(sweep_overrides)
        )

        with open_dict(sweep_config.hydra.job) as job:
            # Populate new job variables
            job.id = submitit.JobEnvironment().job_id  # type: ignore
            sweep_config.hydra.job.num = initial_job_num + task_id

        return run_job(
            hydra_context=self.hydra_context,
            task_function=self.task_function,
            config=sweep_config,
            job_dir_key=job_dir_key,
            job_subdir_key="hydra.sweep.subdir",
        )

    def checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        """Resubmit the current callable at its current state with the same initial arguments."""
        # lazy import to ensure plugin discovery remains fast
        import submitit

        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        """This method runs inside the main Hydra process on the login node. At this point, only
        Hydra logging (not job logging) has been configured, which by default logs everything of
        level INFO and higher to stdout under the tag [HYDRA].
        """

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
                    idx * total_runs_per_node + initial_job_idx,  # initial_job_num
                    f"job_id_for_{idx}",  # job_id
                    Singleton.get_state(),  # singleton_state
                )
            )

        # launch all
        jobs = executor.map_array(self, *zip(*job_params))

        if self.wait_for_completion:
            return [j.results()[0] for j in jobs]
        else:
            # we do our best to emulate what BaseSubmititLauncher.__call__ would do but with a
            # no-op task_function
            assert self.hydra_context is not None
            assert self.config is not None
            assert self.task_function is not None

            no_op = lambda config: None
            job_nums = range(initial_job_idx, initial_job_idx + len(job_overrides))
            results: list[JobReturn] = []
            for job_num, override in zip(job_nums, job_overrides):
                sweep_config = self.hydra_context.config_loader.load_sweep_config(
                    self.config, list(override)
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
