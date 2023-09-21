# clusterduck

clusterduck is a hydra launcher plugin for running jobs in batches on a SLURM cluster. It is intended for small tasks on clusters where jobs have exclusive access to a node, such that submitting a single task to a node would be wasteful.

### Current test command:
```bash
python example/train.py --multirun model=convnet,transformer +platfrom=horeka
```
