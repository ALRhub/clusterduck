import argparse
import pickle
import time
from pathlib import Path

try:  # loading numpy before loading the pickle, to avoid unexpected interactions
    # pylint: disable=unused-import
    import numpy  # type: ignore  # noqa
except ImportError:
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a job")
    parser.add_argument(
        "path",
        type=str,
        help="Path to pickled task callable.",
    )
    args = parser.parse_args()
    pickle_path = Path(args.path)

    # wait for pickle to be created
    wait_time = 60
    for _ in range(wait_time):
        if pickle_path.exists():
            break
        time.sleep(1)
    if not pickle_path.exists():
        raise RuntimeError(
            f"Waited for {wait_time} seconds but could not find submitted jobs in path:\n{pickle_path}"
        )

    # unpickle the task callable
    with open(pickle_path, "rb") as ifile:
        task_fn = pickle.load(ifile)

    # run task
    result = task_fn()
