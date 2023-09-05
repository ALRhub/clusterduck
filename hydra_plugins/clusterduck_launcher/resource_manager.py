from omegaconf import DictConfig

from hydra_plugins.clusterduck_launcher.resources import Resource, CPUResource, CUDAResource, StragglerResource
import math
class ResourceManager:
    def __init__(self, config):
        self.config = config
        self.resource_combinations = []

        self.n_processes = config.n_processes
        self.n_processes_per_resource = config.n_processes_per_resource

        self.cpu_available_resources = []
        self.cuda_available_resources = []
        self.straggler_available_resources = []

        self.get_available_resources()
        self.distribute_resources()

    def get_available_resources(self):
        self.cpu_available_resources = CPUResource.get_available()
        self.cuda_available_resources = CUDAResource.get_available()
        self.straggler_available_resources = StragglerResource.get_available(self.n_processes, **self.config.straggler_resource)


    def get_resource_combinations(self, resources_list):
        n_resources = len(resources_list)
        n_unique_combinations = math.ceil(self.n_processes / self.n_processes_per_resource)
        resource_combinations = []

        for i in range(self.n_processes):
            resource_combinations.append([])
            for j in range(n_resources):
                if j % n_unique_combinations == i % n_unique_combinations:
                    resource_combinations[i].append(resources_list[j])

        return resource_combinations

    def distribute_resources(self):
        cpu_combinations = self.get_resource_combinations(self.cpu_available_resources)
        cuda_combinations = self.get_resource_combinations(self.cuda_available_resources)
        straggler_combinations = self.straggler_available_resources

        for cpu_combination, cuda_combination, straggler_combination in zip(cpu_combinations, cuda_combinations, straggler_combinations):
            self.resource_combinations.append({
                "cpu": CPUResource(cpu_combination),
                "cuda": CUDAResource(cuda_combination),
                "straggler": StragglerResource(straggler_combination)
            })

        return

    def get_resource(self, i):
        return self.resource_combinations[i]



if __name__ == "__main__":
    dc = DictConfig({"n_processes": 4, "n_processes_per_resource": 2, "straggler_resource": {"n_processes": 4, "min_delay": 0.0, "max_delay": 10.0}})
    rm = ResourceManager(dc)

    import numpy as np
    print(rm.get_resource_combinations(np.arange(15).tolist()))
