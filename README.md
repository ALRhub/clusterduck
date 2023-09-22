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
