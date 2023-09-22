# clusterduck

clusterduck is a hydra launcher plugin for running jobs in batches on a SLURM cluster. It is intended for small tasks on clusters where jobs have exclusive access to a node, such that submitting a single task to a node would be wasteful.

### Installation
Install clusterduck with `pip install .`
```bash
pip install .
```

An editable install is only allowed if it is strict, using the following command:
```bash
pip install -e . --config-settings editable_mode=strict
```

### Examples
The example script requires a few additional dependencies. Install with:
```bash
pip install ".[examples]"
```

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
This plugin is heavily inspired by the [hydra-submitit-launcher plugin](https://hydra.cc/docs/plugins/submitit_launcher/), and provides all parameters of that original plugin. See their documentation for details about those parameters.

Both plugins rely on [submitit](https://github.com/facebookincubator/submitit) for the real heavy lifting. See their documentation for more information.

### Additional Parameters
The following parameters are added by this plugin:

We refer to a hydra job, i.e. one execution of the hydra main function with a set of overrides, as a *run*, to differentiate it from both jobs and tasks as defined by SLURM.

- **parallel_runs_per_node:**  
The number of parallel executions per node, i.e. the number of experiments that will run simultaneously in a single SLURM job.
This will depend on the available resources in a node.
- **total_runs_per_node:**  
The total number of executions per node, i.e. the number of experiments that will run in a single SLURM job.
This will depend on the duration of a run, the `parallel_runs_per_node` setting, and the time limit you set for the job in SLURM.
If not specified, all executions will be run in a single job.
However only `parallel_runs_per_node` of these executions will be running at any given time.
- **wait_for_completion:**  
If set to true, the launcher will keep running in your login node until all SLURM jobs before exiting.
Otherwise it will submit the SLURM jobs into the queue and then exit.
- **resources_config:**  
A list of resources that will be divided up among parallel runs within a SLURM job.
Currently available are following options configurable resources:
  - **cpu** This will divide the runs over the available CPUs.
    - Optional argument `cpus` specifies the CPU ids available to the job. Leave blank to auto-detect.
  - **cuda** This will divide the runs over the available GPUs. With this you can specify the GPUs that will be used for the tasks.
    - Optional argument `gpus` specifies the GPU ids available to the job. Leave blank to auto-detect.
  - **stagger:** This will delay the start of each task by the specified amount of seconds. This can be useful if you want to avoid starting all tasks at the same time, e.g. to avoid overloading the file system.
    - Argument `delay` specifies the delay amount in seconds.

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
    total_runs_per_node: 8
    wait_for_completion: True
    resources_config:
      - cpu
      - cuda:
          gpus: [0, 1, 2, 3]
      - stagger:
          delay: 5
```

Further look into the example folder for a working example with multiple example configurations.