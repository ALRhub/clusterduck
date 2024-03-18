import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a job")
    parser.add_argument(
        "path",
        type=str,
        help="Path to pickled task callable.",
    )
    args = parser.parse_args()
    pickle_path = args.path

    # configure logging

    # wait for pickle to be created

    try:
        # unpickle the task callable
        with open(pickle_path, "rb") as ifile:
            task = pickle.load(ifile)

        # configure signal handlers

        # save result via pickle

    except Exception as e:
        # dump error and traceback
        # logging
        raise e
