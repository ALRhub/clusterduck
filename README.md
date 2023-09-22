# clusterduck

clusterduck is a hydra launcher plugin for running jobs in batches on a SLURM cluster. It is intended for small tasks on clusters where jobs have exclusive access to a node, such that submitting a single task to a node would be wasteful.

### Examples

To run the example script locally, e.g. looping over both model types twice each, use:
```bash
python example/train.py --multirun model=convnet,transformer +iteration="range(2)"
```

To run the example script with the submitit backend but locally without a cluster, specify the platform like this:
```bash
python example/train.py --multirun model=convnet,transformer +iteration="range(2)" +platform=slurm_debug
```

To run the example script on the HoreKa cluster, use:
```bash
python example/train.py --multirun model=convnet,transformer +iteration="range(2)" +platform=horeka
```

### Development
PyCUDA is a helpful tool for working with CUDA devices outside of the context of a machine learning library like pytorch. We recommend installing it with conda:
```bash
conda install pycuda
```

Install additional requirements for development using:
```bash
pip install ".[all]"
```


## Configuration Options
This plugin is heavily inspired by the hydra-launcher-submitit plugin, it provides all parameters of that original plugin. 
See [here](https://hydra.cc/docs/plugins/submitit_launcher/) for the documentation of the original plugin.

### Additional Parameters
The following parameters are added by this plugin:

We specify a task as a single experiment, i.e. a single run of the script with a single set of parameters.

- **parallel_runs_per_node:**  
The number of parallel running tasks per node, i.e. the number of experiments that will run simultaneously in a single SLURM job.
- **total_tasks_per_node:**  
The total number of tasks per node, i.e. the number of experiments that will run in a single SLURM job.  
If not specified all tasks will be run in a single job. However only as many tasks will be run in parallel as specified in `parallel_runs_per_node`.
- **wait_for_completion:**  
If set to true, the launcher will wait for all jobs to finish before exiting. Otherwise it will keep running in your login node until all experiments are finished.
- **resources_config:**  
A list of resources that will be passed to the SLURM job. Currently available are following options configurable resources:
  - **cpu:** This will spread the tasks over the available CPUs.
  - **cuda:** With this you can specify the GPUs that will be used for the tasks.
  - **stagger:** This will delay the start of each task by the specified amount of seconds. This can be useful if you want to avoid starting all tasks at the same time, e.g. to avoid overloading the file system.

Here an example of a `hydra/launcher` config that uses all of the above options:
```yaml
hydra:
  launcher:
    # launcher specific options
    timeout_min: 5
    partition: dev_accelerated
    gres: gpu:4
    
    # clusterduck specific options
    parallel_runs_per_node: 2
    total_tasks_per_node: 8
    wait_for_completion: True
    resources_config:
      - cpu
      - cuda:
          gpus: [0, 1, 2, 3]
      - stagger:
          delay: 5
```

Further look into the example folder for a working example with multiple example configurations.