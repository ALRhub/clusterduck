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
