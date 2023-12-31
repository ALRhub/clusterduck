import multiprocessing as mp
import pickle
from multiprocessing.connection import Connection, wait
from typing import Any, Callable, Literal, Sequence, TypeVar

import cloudpickle

from ._logging import get_logger
from ._resources import Resource, ResourcePool

ReturnType = TypeVar("ReturnType")

logger = get_logger(__name__)


class WorkerPool:
    def __init__(
        self,
        n_workers: int,
        resource_pools: Sequence[ResourcePool],
        start_method: Literal["fork", "spawn", "forkserver"] = "fork",
    ) -> None:
        self.n_workers = n_workers
        self.resource_pools = resource_pools
        self.start_method = start_method

    def worker(
        self,
        target: Callable,
        resources: Sequence[Resource],
        pipe: Connection,
        **kwargs: Any,
    ):
        try:
            ret = target(resources=resources, **kwargs)
            pipe.send(cloudpickle.dumps(ret))

        except Exception as e:
            import traceback

            traceback.print_exc()
            pipe.send(cloudpickle.dumps(e))

    def execute(
        self,
        target_fn: Callable[..., ReturnType],
        kwargs_list: Sequence[dict],
    ) -> list[ReturnType]:
        ctx = mp.get_context(self.start_method)
        processes: list[mp.Process] = []
        n_processes = min(self.n_workers, len(kwargs_list))
        manager_pipes, worker_pipes = zip(*[ctx.Pipe() for _ in range(n_processes)])

        for i in range(n_processes):
            resources = [resource_pool.get(i) for resource_pool in self.resource_pools]

            process = ctx.Process(
                target=self.worker,
                kwargs=dict(
                    target=target_fn,
                    resources=resources,
                    pipe=worker_pipes[i],
                    **kwargs_list[i],
                ),
            )
            logger.info(
                f"Starting process #{i} in slot #{i} with arguments: {kwargs_list[i]}"
            )
            logger.debug(f"This process will be given resources: {resources}")
            process.start()
            processes.append(process)

        results = []
        submitted_overrides = n_processes
        while submitted_overrides < len(kwargs_list):
            # wait for next override to finish or fail
            sentinels = [proc.sentinel for proc in processes]
            done_processes = wait(sentinels)

            for done_process in done_processes:
                worker_id = sentinels.index(done_process)
                processes[worker_id].join()

                if not manager_pipes[worker_id].poll():
                    logger.error(
                        f"Process {submitted_overrides - 1} crashed with no return value."
                    )
                    raise RuntimeError("Worker process sent no return value.")
                result = pickle.loads(manager_pipes[worker_id].recv())

                if isinstance(result, Exception):
                    logger.error(
                        f"Process {submitted_overrides - 1} completed with an uncaught exception."
                    )
                    raise result
                logger.debug(f"Process {submitted_overrides - 1} completed normally.")
                results.append(result)

                resources = [
                    resource_pool.get(worker_id)
                    for resource_pool in self.resource_pools
                ]
                processes[worker_id] = ctx.Process(
                    target=self.worker,
                    kwargs=dict(
                        target=target_fn,
                        resources=resources,
                        pipe=worker_pipes[worker_id],
                        **kwargs_list[submitted_overrides],
                    ),
                )
                logger.info(
                    f"Starting process #{submitted_overrides} in slot #{worker_id} with arguments: {kwargs_list[submitted_overrides]}"
                )
                logger.debug(f"This process will be given resources: {resources}")
                processes[worker_id].start()
                submitted_overrides += 1
                if submitted_overrides == len(kwargs_list):
                    break

        for worker_id, process in enumerate(processes):
            process.join()

            if not manager_pipes[worker_id].poll():
                # TODO: maybe do not throw an Exception here, as this stops job
                raise RuntimeError("Worker process sent no return value.")
            result = pickle.loads(manager_pipes[worker_id].recv())

            if isinstance(result, Exception):
                raise result
            results.append(result)

        for manager_pipe, worker_pipe in zip(manager_pipes, worker_pipes):
            manager_pipe.close()
            worker_pipe.close()

        return results
