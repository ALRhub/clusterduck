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
        use_srun: bool = True,
        do_submit: bool = True,
        local_debug: bool = False,
        **kwargs: Any,
    ) -> None:
        self.log_folder = Path(log_folder).resolve()
        self.use_srun = use_srun and not local_debug
        self.do_submit = do_submit and not local_debug
        self.local_debug = local_debug

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
        from ._slurm import clean_slurm_kwargs, make_sbatch_string
        from ._utils import run_command

        setup_globals()
        assert self.config is not None
        assert self.task_function is not None
        assert self.hydra_context is not None

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        self.log_folder.mkdir(parents=True, exist_ok=True)
        kwargs = dict(self.kwargs)

        ## Create the task function executed by each task and pickle it
        task = functools.partial(
            execute_job,
            initial_job_idx,
            job_overrides,
            self.hydra_context,
            self.config,
            self.task_function,
            singleton_state=Singleton.get_state(),
        )

        # We create one pickle file per job array, then decide which override
        # to apply based on the array index.
        pickle_path = self.log_folder / self.PICKLE_FILENAME
        with open(pickle_path, "wb") as ofile:
            cloudpickle.dump(task, ofile, pickle.HIGHEST_PROTOCOL)

        ## Create the batch file for submitting the jobs to slurm
        submission_path = self.log_folder / self.SBATCH_FILENAME

        ## Compute number of tasks and jobs
        num_tasks = len(job_overrides)  # 1 task per override
        tasks_per_job = int(kwargs["tasks_per_node"])
        num_jobs = math.ceil(num_tasks / tasks_per_job)  # group into jobs
        assert num_tasks > 0 and tasks_per_job > 0 and num_jobs > 0

        ## Pretty print how tasks (overrides) are distributed across jobs
        log.info(f"Launching jobs, sweep output dir : {sweep_dir}")
        for idx in range(num_jobs):
            job_overrides_sublist = job_overrides[
                idx * tasks_per_job : (idx + 1) * tasks_per_job
            ]
            log.info(f"\tJob #{idx} :")
            for overrides in job_overrides_sublist:
                lst = " ".join(filter_overrides(overrides))
                log.info(f"\t\t{lst}")

        # TODO: move these into init args of class
        extra_sbatch_kwargs = kwargs.pop("sbatch_kwargs", {}) or {}
        srun_kwargs = kwargs.pop("srun_kwargs", {}) or {}
        setup = kwargs.pop("setup", []) or []
        teardown = kwargs.pop("teardown", []) or []
        tmpdir_vars = kwargs.pop("tmpdir_vars", ["TMP", "TMPDIR"]) or []
        # remaining fields are assumed to be sbatch parameters
        sbatch_kwargs = kwargs

        sbatch_kwargs["array_count"] = num_jobs

        if tasks_per_job > 1:
            # Ensure that each task inside the node has exclusive access to its
            # resources, e.g. each GPU is only visible to one task.
            srun_kwargs["exclusive"] = True

            # If RANK and LOCAL_RANK are not set, pytorch lightning uses
            # SLURM_PROCID to determine the rank of the process, which determines
            # e.g. whether the loggers should actually log. If we are not
            # doing multi-node or multi-gpu training, we set this to 0 to ensure
            # that all tasks log their output.
            setup.insert(0, "export RANK=0")
            setup.insert(0, "export LOCAL_RANK=0")

        ## construct log file names
        stderr_to_stdout = sbatch_kwargs.pop("stderr_to_stdout", True)
        # %A is actually always the correct job id, even with single jobs
        # %j returns a different id for each job in a job array, so we can only
        # use it if we have a single job
        # %a returns a nonsense integer for single jobs, so we can only use it if we have a job array
        log_name_prefix = "%j" if num_jobs == 1 else "%A_%a"

        sbatch_kwargs["output"] = str(self.log_folder / f"{log_name_prefix}_out.log")
        if not stderr_to_stdout:
            sbatch_kwargs["error"] = str(self.log_folder / f"{log_name_prefix}_err.log")

        sbatch_kwargs["open-mode"] = "append"

        # If we have multiple tasks per job, we need to add the task id to the
        # log file names to avoid different tasks overwriting each other's logs.
        # Otherwise we don't need to specify output and error for srun, because
        # it will inherit the ones from sbatch.
        if tasks_per_job > 1:
            log_name_prefix += "_task_%t"

            srun_kwargs["output"] = str(self.log_folder / f"{log_name_prefix}_out.log")
            if not stderr_to_stdout:
                srun_kwargs["error"] = str(
                    self.log_folder / f"{log_name_prefix}_err.log"
                )

            srun_kwargs["open-mode"] = "append"

        # Ensure that we get the full stack trace if the job fails
        setup.insert(0, "export HYDRA_FULL_ERROR=1")

        # ensure that kwargs use canonical parameter names so that merging works correctly
        sbatch_kwargs = clean_slurm_kwargs(sbatch_kwargs)
        extra_sbatch_kwargs = clean_slurm_kwargs(extra_sbatch_kwargs)
        srun_kwargs = clean_slurm_kwargs(srun_kwargs)

        # merge extra_sbatch_kwargs into sbatch_kwargs, with a warning if
        # existing parameters are overwritten
        for key, value in extra_sbatch_kwargs.items():
            if key in sbatch_kwargs:
                log.warning(
                    f"Overwriting sbatch parameter {key}={sbatch_kwargs[key]} with {key}={value} from sbatch_kwargs"
                )
            sbatch_kwargs[key] = value

        ## Create the python command to execute with srun
        python_command = [
            sys.executable,  # gets the path to the python interpreter in the currently active environment
            "-u",  # Force the stdout and stderr streams to be unbuffered
            "-m",
            "hydra_plugins.clusterduck_launcher._run",
            str(pickle_path),
        ]

        sbatch_text = make_sbatch_string(
            command=python_command,
            sbatch_kwargs=sbatch_kwargs,
            srun_kwargs=srun_kwargs,
            setup=setup,
            teardown=teardown,
            use_srun=self.use_srun,
        )

        with submission_path.open("w") as f:
            f.write(sbatch_text)

        if self.do_submit:
            submission_command = ["sbatch", str(submission_path)]
            submission_result = run_command(submission_command)
            log.info(submission_result)
        else:
            log.info(
                f"Generated submission script at {submission_path}, not submitting"
            )

        # return empty list of JobReturn since we don't wait for the jobs to complete
        return []
