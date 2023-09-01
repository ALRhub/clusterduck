# clusterduck

clusterduck is a hydra launcher plugin for running jobs in batches on a SLURM cluster. It is intended for small tasks on clusters where jobs have exclusive access to a node, such that submitting a single task to a node would be wasteful.

### Current test command:
```bash
python example/my_app.py --multirun db=postgresql,mysql
```

### Dask Version:
```bash
# Install dask
mamba install -c conda-forge dask dask-cuda

python example/my_app.py --config-name=config_bwcluster --multirun db=postgresql,mysql hydra/launcher=clusterduck_slurm '+seed=range(1,20)'
```