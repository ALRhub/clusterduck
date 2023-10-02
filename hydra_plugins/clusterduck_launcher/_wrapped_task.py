from typing import Any, Callable, Sequence

import cloudpickle

from ._logging import get_logger
from ._resources import Resource

logger = get_logger(__name__)


class WrappedTaskFunction:
    def __init__(self, task_function: Callable) -> None:
        self.task_function = cloudpickle.dumps(task_function)
        self.resources = None

    def set_resources(self, resources: Sequence[Resource]) -> None:
        self.resources = resources

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """This method runs inside the SLURM job inside a fresh process forked by `run_workers`.
        This method is called by Hydra's `run_job`, so Hydra's job_logging has been configured.
        Because Hydra has been set up, the Hydra config is also accessible via
        `from hydra.core.hydra_config import HydraConfig`.
        """
        assert self.resources is not None

        logger.debug("Setting allocated resources...")

        for resource in self.resources:
            resource.apply()

        logger.debug("Unpickling task function...")

        task_function: Callable = cloudpickle.loads(self.task_function)

        logger.debug("Running task function...")

        result = task_function(*args, **kwds)

        logger.debug("Completed task function.")

        return result
