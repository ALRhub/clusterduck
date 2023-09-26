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
