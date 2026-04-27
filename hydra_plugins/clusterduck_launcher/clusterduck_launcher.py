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
from omegaconf import DictConfig, OmegaConf, open_dict

log = logging.getLogger("clusterduck.launcher")


class ClusterDuckLauncher(Launcher):
    SBATCH_FILENAME = "submission.sh"
    PICKLE_FILENAME = "submitted.pkl"

    def __init__(
        self,
        log_folder: str,
        parallel_tasks_per_job: int = 1,
        sequential_tasks_per_job: int = 1,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self.parallel_tasks_per_job = parallel_tasks_per_job
        self.sequential_tasks_per_job = sequential_tasks_per_job
        self.log_folder = Path(log_folder)
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
        import shlex
        import sys

        import cloudpickle
        from hydra.core.singleton import Singleton
        from hydra.core.utils import (
            JobReturn,
            configure_log,
            filter_overrides,
            setup_globals,
        )

        from ._core import execute_job
        from ._slurm import as_sbatch_flag, as_srun_args
        from ._utils import run_command
        from .config import RESOURCE_FIELDS, SBATCH_FIELDS

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

        # job_parameters = {
        #     "partition": "accelerated",
        #     "time": 30,
        #     "nodes": 1,
        #     "ntasks_per_node": 4,
        #     "gres": "gpu:4",
        #     "open-mode": "append",
        # }
        # srun_parameters = {
        #     "cpus-per-task": 38,
        #     "exclusive": True,
        # }
        # self.stderr_to_stdout = False

        num_tasks = len(job_overrides)
        tasks_per_job = self.parallel_tasks_per_job * self.sequential_tasks_per_job
        num_jobs = math.ceil(num_tasks / tasks_per_job)
        assert num_tasks > 0 and num_jobs > 0

        # TODO: limit num_tasks to number of overrides, right?

        sbatch_kwargs = self.kwargs.pop("sbatch_kwargs", {}) or {}
        srun_kwargs = self.kwargs.pop("srun_kwargs", {}) or {}
        setup = self.kwargs.pop("setup", None)
        array_parallelism = self.kwargs.pop("array_parallelism", 256)
        stderr_to_stdout = self.kwargs.pop("stderr_to_stdout", False)
        signal_delay_s = self.kwargs.pop("signal_delay_s", 120)

        sbatch_kwargs.update(
            {key: self.kwargs.pop(key) for key in SBATCH_FIELDS if key in self.kwargs}
        )

        resource_kwargs = {
            key: self.kwargs.pop(key) for key in RESOURCE_FIELDS if key in self.kwargs
        }

        assert not self.kwargs

        if tasks_per_job == 1:
            # In this case, each job in the array submission requests the
            # specified resources and only runs a single task.
            sbatch_kwargs.update(resource_kwargs)

        else:
            # In this case, we assume that the cluster only allocates entire
            # nodes exclusively. Each job simply requests a node, and we forward
            # all resource requests to the srun command.
            srun_kwargs.update(resource_kwargs)
            # Launch n parallel instances of the task inside each job (node)
            sbatch_kwargs["ntasks"] = srun_kwargs["ntasks"] = tasks_per_job
            # Ensure that each task inside the node has exclusive access to its
            # resources, e.g. each GPU is only visible to one task.
            srun_kwargs["exclusive"] = True

        if num_jobs == 1:
            stdout = self.log_folder / "%j/%j_out.log"
            stderr = self.log_folder / "%j/%j_err.log"
            srun_stdout = self.log_folder / "%j/%j_%t_out.log"
            srun_stderr = self.log_folder / "%j/%j_%t_err.log"
        else:
            sbatch_kwargs["array"] = (
                f"0-{num_jobs - 1}%{min(num_jobs, array_parallelism)}"
            )
            stdout = self.log_folder / "%A_%a/%A_%a_out.log"
            stderr = self.log_folder / "%A_%a/%A_%a_err.log"
            srun_stdout = self.log_folder / "%A_%a/%A_%a_%t_out.log"
            srun_stderr = self.log_folder / "%A_%a/%A_%a_%t_err.log"

        sbatch_kwargs["output"] = str(stdout)
        if not stderr_to_stdout:
            sbatch_kwargs["error"] = str(stderr)

        srun_kwargs["output"] = str(srun_stdout)
        if not stderr_to_stdout:
            srun_kwargs["error"] = str(srun_stderr)

        # Filter out None values (unset values)
        sbatch_kwargs = {
            key: value for key, value in sbatch_kwargs.items() if value is not None
        }
        srun_kwargs = {
            key: value for key, value in srun_kwargs.items() if value is not None
        }

        lines = ["#!/bin/bash", "", "# Parameters"]
        for key, value in sbatch_kwargs.items():
            lines.append(as_sbatch_flag(key, value))

        # environment setup:
        if setup is not None:
            lines += ["", "# setup"] + setup

        srun = ["srun", "--unbuffered"]
        for key, value in srun_kwargs.items():
            srun.append(as_srun_args(key, value))

        python_command = [
            sys.executable,
            "-u",  # Force the stdout and stderr streams to be unbuffered
            "-m",
            "hydra_plugins.clusterduck_launcher._run",
            str(pickle_path),
        ]
        srun.extend(python_command)
        srun = shlex.join(srun)
        lines += ["", "# command", srun, ""]

        with submission_path.open("w") as f:
            f.write("\n".join(lines))

        submission_command = ["sbatch", str(submission_path)]
        run_command(submission_command)

        # if self.wait_for_completion:
        #     return [j.results()[0] for j in jobs]
        # else:
        #     # we do our best to emulate what BaseSubmititLauncher.__call__ would do but with a
        #     # no-op task_function
        #     assert self.hydra_context is not None
        #     assert self.config is not None
        #     assert self.task_function is not None

        #     no_op = lambda config: None
        #     job_nums = range(initial_job_idx, initial_job_idx + len(job_overrides))
        #     results: list[JobReturn] = []
        #     for job_num, override in zip(job_nums, job_overrides):
        #         sweep_config = self.hydra_context.config_loader.load_sweep_config(
        #             self.config, list(override)
        #         )

        #         with open_dict(sweep_config.hydra.job) as job:
        #             # Populate new job variables
        #             # Cannot set job ID to slurm job ID as we are not inside slurm job
        #             sweep_config.hydra.job.num = job_num

        #         result = run_job(
        #             hydra_context=self.hydra_context,
        #             task_function=no_op,
        #             config=sweep_config,
        #             job_dir_key="hydra.sweep.dir",
        #             job_subdir_key="hydra.sweep.subdir",
        #             configure_logging=False,
        #         )

        #         results.append(result)
        #     return results
