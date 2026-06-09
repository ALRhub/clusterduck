# clusterduck

clusterduck is a hydra launcher plugin for running jobs on a SLURM cluster.

Unlike [hydra-submitit](https://hydra.cc/docs/plugins/submitit_launcher/), clusterduck also supports "batching" multiple tasks within one job.
This may be useful if:
1. your cluster only allocates exclusive nodes with multiple GPUs, but your tasks only use a single GPU
2. you have hundreds of small jobs but your cluster imposes a (potentially project-wide) limit on the number of queued jobs

In addition, clusterduck does not wait for your job to finish after submission!

## Installation

Install clusterduck with `pip install .`
```bash
pip install .
```

Developers should note that Hydra plugins are not compatible with new PEP 660-style editable installs.
In order to perform an editable install, either use [compatibility mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html#legacy-behavior):
```bash
pip install -e . --config-settings editable_mode=compat
```
or use [strict editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html#strict-editable-installs).
```bash
pip install -e . --config-settings editable_mode=strict
```
Be aware that strict mode installs do not expose new files created in the project until the installation is performed again.

## Usage

clusterduck essentially allows you to generate sbatch files programmatically.
We generate an sbatch script containing a single srun command that calls python.
Since every cluster is different, we do not try to be clever, but instead let the user set the arguments for sbatch and srun transparently.

Each hydra override becomes a slurm task.
One or more tasks may be grouped into a slurm job, and one or more jobs may be grouped into a slurm job array.
By default, we use one task per job and submit a job array if there are multiple tasks to run.

### Configuration

After clusterduck is installed, you can print out the available configuration options with the following command:

```bash
python your_app.py hydra/launcher=clusterduck_slurm --cfg hydra -p hydra.launcher
```

The majority of these arguments are passed to sbatch.
In addition, `sbatch_kwargs` allows for adding and overriding arbitrary sbatch arguments, while `srun_kwargs` does the same for srun.
Lastly `setup` allows for arbitrary shell commands to be executed before python is called (useful for environment variables, etc.).

See example configs for cluster platforms under `example/conf/platform`.

### Batching Tasks Within Jobs

To batch multiple tasks into a single job, set `tasks_per_node`>1.
(We use `tasks_per_node` instead of `ntasks`, because `ntasks`>1 is not compatible with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).)
Beyond that, you may need to adjust the config so that GPUs and CPUs are divided correctly among the tasks. Ideally, you can specify all required resources with e.g. `cpus_per_task` and `gpus_per_task`, in which case everything is handled automatically.
Unfortunately, not all clusters support this.
Please see the examples (`example/conf/platform`) and experiment with your cluster to see what works.

### Verbose Logging

To debug resource allocation, please install the optional dependencies with `pip install ".[dev]"`.
Then, use [hydra's verbose logging feature](https://hydra.cc/docs/tutorials/basic/running_your_app/logging/) to activate verbose logging in clusterduck.
Either add `hydra.verbose=clusterduck` to your command or add the following to your config:
```yaml
hydra:
  verbose: clusterduck
```

Afterwards, check the slurm logs your job produces to see which GPUs and CPUs and how much memory your job is assigned.

### Debugging

We also provide the following non-slurm options for debugging:

- **use_srun:**  
If `True`, the python command will be launched by srun. If `False`, the python command is run directly inside the job. (default: `True`)
- **do_submit:**  
If `False`, create the submission file but do not actually submit it. (default: `True`)
- **local_debug:**  
If `True`, this is a shortcut for `use_srun=False` and `do_submit=False`. This generates a script that can be executed locally as a standard shell script. (default: `False`)

## Example

The example script does not requires additional dependencies, but they are nice to have. Install with:
```bash
pip install ".[dev]"
```

To run the example script locally, e.g. looping over both model types twice each, use:
```bash
python example/train.py --multirun model=convnet,transformer +iteration="range(2)"
```

To run the example script with the submitit backend but locally without a cluster, specify the platform like this:
```bash
python example/train.py --multirun model=convnet,transformer +iteration="range(2)" +platform=local_debug
```

To run the example script on the HoreKa cluster, use:
```bash
python example/train.py --multirun model=convnet,transformer +iteration="range(2)" +platform=horeka
```

## Sharp Edges

### Hydra Sweepers

Because the clusterduck launcher does not wait for the jobs to complete, it is not compatible with any sweepers that optimize some returned value.
