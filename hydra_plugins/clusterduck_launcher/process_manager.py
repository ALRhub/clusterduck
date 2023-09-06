from typing import List, Dict

from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, JobStatus
from omegaconf import DictConfig

from hydra_plugins.clusterduck_launcher.resource_manager import ResourceManager


class ProcessManager:
    def __init__(
            self,
            parallel_executions_in_job: int,
            target_fn: callable,
            resources_config: DictConfig,
    ):
        self.parallel_executions_in_job = parallel_executions_in_job
        self.target_fn = target_fn
        self.resource_manager = ResourceManager(resources_config)


    def target_fn_proxy(self, resources, singleton_state, **kwargs):
        import cloudpickle
        singleton_state = cloudpickle.loads(singleton_state)

        try:
            print(kwargs.keys(), flush=True)
            for resource in resources.values():
                resource.apply()
            return self.target_fn(singleton_state=singleton_state, **kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return e

    def run(
        self,
        sweep_overrides_list: List[List[str]],
        job_dir_key: str,
        job_nums: range,
        job_id: str,
        singleton_state: Dict[type, Singleton],
    ) -> list[JobReturn]:
        import multiprocessing as mp
        import pickle
        from multiprocessing.connection import wait
        import cloudpickle


        mp.set_start_method('spawn')

        processes: list[mp.Process] = []
        n_processes = min(self.parallel_executions_in_job, len(sweep_overrides_list))
        manager_pipes, worker_pipes = zip(*[mp.Pipe() for _ in range(n_processes)])
        job_nums_iter = iter(job_nums)
        for i in range(n_processes):
            # TODO: consider using forkserver context to create processes
            process = mp.Process(
                target=self.target_fn_proxy,
                kwargs=dict(
                    sweep_overrides=sweep_overrides_list[i],
                    job_dir_key=job_dir_key,
                    job_num=next(job_nums_iter),
                    job_id=job_id,
                    singleton_state=cloudpickle.dumps(singleton_state),
                    pipe=worker_pipes[i],
                    resources=self.resource_manager.get_resource(i),
                ),
            )
            process.start()
            processes.append(process)

        results: list[JobReturn] = []
        submitted_overrides = n_processes
        while submitted_overrides < len(sweep_overrides_list):
            # wait for next override to finish or fail
            sentinels = [proc.sentinel for proc in processes]
            done_processes = wait(sentinels)

            for done_process in done_processes:
                resource_id = sentinels.index(done_process)
                processes[resource_id].join()

                if not manager_pipes[resource_id].poll():
                    raise RuntimeError("Worker process sent no return value.")
                result = manager_pipes[resource_id].recv()

                results.append(pickle.loads(result))
                processes[resource_id] = mp.Process(
                    target=self.target_fn_proxy,
                    kwargs=dict(
                        sweep_overrides=sweep_overrides_list[submitted_overrides],
                        job_dir_key=job_dir_key,
                        job_num=next(job_nums_iter),
                        job_id=job_id,
                        singleton_state=cloudpickle.dumps(singleton_state),
                        pipe=worker_pipes[resource_id],
                        resources=self.resource_manager.get_resource(resource_id),
                    ),
                )
                processes[resource_id].start()
                submitted_overrides += 1
                if submitted_overrides == len(sweep_overrides_list):
                    break

        for resource_id, process in enumerate(processes):
            process.join()
            if not manager_pipes[resource_id].poll():
                raise RuntimeError("Worker process sent no return value.")
            result = manager_pipes[resource_id].recv()

            results.append(pickle.loads(result))

        for manager_pipe, worker_pipe in zip(manager_pipes, worker_pipes):
            manager_pipe.close()
            worker_pipe.close()

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
