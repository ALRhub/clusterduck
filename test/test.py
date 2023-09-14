import os
from time import sleep
from loky import get_reusable_executor



def say_hello(k):

    pid = os.getpid()
    print(f"Hello from {pid} with arg {k}")
    sleep(.01)
    return pid

if __name__ == "__main__":
    # Create an executor with 4 worker processes, that will
    # automatically shutdown after idling for 2s
    executor = get_reusable_executor(max_workers=1, timeout=200)

    res = executor.submit(say_hello, 1)
    print("Got results:", res.result())

    results = executor.map(say_hello, range(50))
    n_workers = len(set(results))
    print("Number of used processes:", n_workers)
    assert n_workers == 4