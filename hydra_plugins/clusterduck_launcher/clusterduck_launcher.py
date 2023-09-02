import logging
import multiprocessing as mp
import os
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, filter_overrides, run_job, setup_globals
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf, open_dict

from .config import BaseQueueConf

log = logging.getLogger(__name__)


class BaseClusterDuckLauncher(Launcher):
    _EXECUTOR = "abstract"

    def __init__(
        self,
        num_of_overrides_per_job: int = 1,
        parallel_executions_in_job: int = 1,
        exclusive_gpu_per_execution: bool = False,
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

    def process_manager(
        self,
        sweep_overrides_list: List[List[str]],
        job_dir_key: str,
        job_num: int,
        job_id: str,
        singleton_state: Dict[type, Singleton],
    ) -> list[JobReturn]:
        processes: list[mp.Process] = []
        n_processes = min(self.parallel_executions_in_job, len(sweep_overrides_list))
        for i in range(n_processes):
            # TODO: consider using forkserver context to create processes
            process = mp.Process(
                target=self,
                kwargs=dict(
                    sweep_overrides=sweep_overrides_list[i],
                    job_dir_key=job_dir_key,
                    job_num=job_num,
                    job_id=job_id,
                    singleton_state=singleton_state,
                    resources=i,
                ),
            )
            process.start()
            processes.append(process)

        submitted_overrides = n_processes
        while submitted_overrides < len(sweep_overrides_list):
            # wait for next override to finish
            sentinels = [proc.sentinel for proc in processes]
            done_processes = wait(sentinels)

            for done_process in done_processes:
                resource_id = sentinels.index(done_process)
                processes[resource_id].join()
                processes[resource_id] = mp.Process(
                    target=self,
                    kwargs=dict(
                        sweep_overrides=sweep_overrides_list[submitted_overrides],
                        job_dir_key=job_dir_key,
                        job_num=job_num,
                        job_id=job_id,
                        singleton_state=singleton_state,
                        resources=resource_id,
                    ),
                )
                processes[resource_id].start()
                submitted_overrides += 1
                if submitted_overrides == len(sweep_overrides_list):
                    break

        for process in processes:
            process.join()

        # TODO: get return value from overrides and return here
        return [JobReturn() for _ in sweep_overrides_list]

    def __call__(
        self,
        sweep_overrides: List[str],
        job_dir_key: str,
        job_num: int,
        job_id: str,
        singleton_state: Dict[type, Singleton],
        resources,
    ) -> JobReturn:
        # lazy import to ensure plugin discovery remains fast
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

        # assign resources
        # TODO: abstract this
        os.environ["CUDA_VISIBLE_DEVICES"] = str(resources)

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
            idx = initial_job_idx + idx

            filtered_job_overrides_sublist = [
                " ".join(filter_overrides(overrides))
                for overrides in job_overrides_sublist
            ]
            log.info(f"Launching {filtered_job_overrides_sublist} in job with id {idx}")

            job_params.append(
                (
                    job_overrides_sublist,
                    "hydra.sweep.dir",
                    idx,  # TODO: adjust so that each override has unique idx?
                    f"job_id_for_{idx}",
                    Singleton.get_state(),
                )
            )

        # launch all
        jobs = executor.map_array(self.process_manager, *zip(*job_params))

        # TODO: how to give a return value without waiting for job completion
        for job in jobs:
            log.info(f"Job has {len(job.results())} results:")
            for i, result in enumerate(job.results()):
                log.info(f"Result {i} has length {len(result)}")
                for j, job_result in enumerate(result):
                    log.info(f"Job result {j}: {job_result}")

        return [
            result for j in jobs for job_result in j.results() for result in job_result
        ]


class ClusterDuckLocalLauncher(BaseClusterDuckLauncher):
    _EXECUTOR = "local"


class ClusterDuckSlurmLauncher(BaseClusterDuckLauncher):
    _EXECUTOR = "slurm"
