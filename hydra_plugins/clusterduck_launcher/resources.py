import os
from abc import ABC, abstractmethod


class Resource(ABC):
    @abstractmethod
    def apply(self):
        pass

    @staticmethod
    @abstractmethod
    def get_available():
        pass


class CUDAResource(Resource):
    def __init__(self, cuda_devices):
        self.cuda_devices = cuda_devices

    def apply(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.cuda_devices))

    @staticmethod
    def get_available():
        #TODO: add config for gpu affinity
        try:
            import torch
            return [i for i in range(torch.cuda.device_count())]
        except ImportError:
            pass

        try:
            import tensorflow as tf
            return tf.config.list_physical_devices("GPU")
        except ImportError:
            pass

        try:
            import pycuda.driver as cuda
            return [i for i in range(cuda.Device.count())]
        except ImportError:
            pass

        raise ImportError("Either PyTorch, Tensorflow or PyCUDA must be installed to use CUDAResource")

class CPUResource(Resource):
    def __init__(self, cpus):
        self.cpus = cpus

    def apply(self):
        import psutil
        psutil.Process().cpu_affinity(self.cpus)

    @staticmethod
    def get_available():
        #TODO: adding config for cpu affinity
        import psutil
        psutil.Process().cpu_affinity([])
        return psutil.Process().cpu_affinity()

class StragglerResource(Resource):
    def __init__(self, delay):
        self.delay = delay

    def apply(self):
        import time
        time.sleep(self.delay)

    @staticmethod
    def get_available(n_processes, min_delay, max_delay):
        import numpy as np
        return np.linspace(min_delay, max_delay, n_processes).tolist()
