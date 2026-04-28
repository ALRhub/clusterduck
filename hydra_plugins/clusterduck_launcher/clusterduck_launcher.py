from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from hydra.core.utils import JobReturn
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger("clusterduck")


class ClusterDuckLauncher(Launcher):
    SBATCH_FILENAME = "submission.sh"
    PICKLE_FILENAME = "submitted.pkl"

    def __init__(
        self,
        log_folder: str,
        parallel_tasks_per_job: int = 1,
        sequential_tasks_per_job: int = 1,
        do_submit: bool = True,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self.log_folder = Path(log_folder)
        self.parallel_tasks_per_job = parallel_tasks_per_job
        self.sequential_tasks_per_job = sequential_tasks_per_job
        self.do_submit = do_submit
        self.verbose = verbose

        # parameters used by submitit
        self.kwargs = {
            key: (
                OmegaConf.to_container(value, resolve=True)
                if OmegaConf.is_config(value)
                else value
            )
            for key, value in kwargs.items()
        }

        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

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

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        # lazy import to ensure plugin discovery remains fast
        import functools
        import math
        import pickle
        import sys

        import cloudpickle
        from hydra.core.singleton import Singleton
        from hydra.core.utils import configure_log, filter_overrides, setup_globals

        from ._core import execute_job
        from ._slurm import make_sbatch_string
        from ._utils import run_command

        setup_globals()
        assert self.config is not None
        assert self.task_function is not None
        assert self.hydra_context is not None

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Launching jobs, sweep output dir : {sweep_dir}")
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        task = functools.partial(
            execute_job,
            initial_job_idx,
            job_overrides,
            self.hydra_context,
            self.config,
            self.task_function,
            singleton_state=Singleton.get_state(),
        )

        self.log_folder.mkdir(parents=True, exist_ok=True)

        # We create one pickle file per job array, then decide which override
        # to apply based on the array index.
        pickle_path = self.log_folder / self.PICKLE_FILENAME
        with open(pickle_path, "wb") as ofile:
            cloudpickle.dump(task, ofile, pickle.HIGHEST_PROTOCOL)

        submission_path = self.log_folder / self.SBATCH_FILENAME

        num_tasks = len(job_overrides)  # 1 task per override
        tasks_per_job = self.parallel_tasks_per_job * self.sequential_tasks_per_job
        tasks_per_job = min(tasks_per_job, num_tasks)  # limit to number of overrides
        num_jobs = math.ceil(num_tasks / tasks_per_job)  # group into jobs
        assert num_tasks > 0 and num_jobs > 0

        sbatch_kwargs = self.kwargs.pop("sbatch_kwargs", {}) or {}
        srun_kwargs = self.kwargs.pop("srun_kwargs", {}) or {}
        setup = self.kwargs.pop("setup", None)

        # remaining fields are assumed to be sbatch parameters
        sbatch_kwargs.update(self.kwargs)

        sbatch_kwargs["array_count"] = num_jobs

        if tasks_per_job > 1:
            # Launch n parallel instances of the task inside each job (node)
            sbatch_kwargs["ntasks"] = srun_kwargs["ntasks"] = tasks_per_job
            # Ensure that each task inside the node has exclusive access to its
            # resources, e.g. each GPU is only visible to one task.
            srun_kwargs["exclusive"] = True

        python_command = [
            sys.executable,
            "-u",  # Force the stdout and stderr streams to be unbuffered
            "-m",
            "hydra_plugins.clusterduck_launcher._run",
            str(pickle_path),
        ]

        sbatch_text = make_sbatch_string(
            command=python_command,
            log_folder=self.log_folder,
            sbatch_kwargs=sbatch_kwargs,
            srun_kwargs=srun_kwargs,
            setup=setup,
        )

        with submission_path.open("w") as f:
            f.write(sbatch_text)

        if self.do_submit:
            submission_command = ["sbatch", str(submission_path)]
            run_command(submission_command)
        else:
            log.info(
                f"Generated submission script at {submission_path}, not submitting"
            )

        return []
