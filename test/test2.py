from time import sleep
import multiprocessing as mp
import loky
from loky import get_reusable_executor

# Store the initialization status in a global variable of a module.
loky._INITIALIZER_STATUS = "uninitialized"


def initializer(x):
    print(f"[{mp.current_process().name}] init")
    print(mp.current_process())
    loky._INITIALIZER_STATUS = x


def return_initializer_status(delay=0):
    print(f"[{mp.current_process().name}] return")
    print(mp.current_process())

    sleep(delay)

    return getattr(loky, "_INITIALIZER_STATUS", "uninitialized")

if __name__ == "__main__":

    executor = get_reusable_executor(
        max_workers=2,
        initializer=initializer,
        initargs=("initialized",),
        context="loky",
        timeout=1000,
    )

    assert loky._INITIALIZER_STATUS == "uninitialized"
    executor.submit(return_initializer_status).result()
    assert executor.submit(return_initializer_status).result() == "initialized"

    # With reuse=True, the executor use the same initializer
    executor1 = get_reusable_executor(max_workers=4, reuse=True)
    for x in executor1.map(return_initializer_status, [0.5] * 4):
        assert x == "initialized"

    # With reuse='auto', the initializer is not used anymore as a new executor
    # is created.
    executor2 = get_reusable_executor(max_workers=4, reuse=False)
    for x in executor2.map(return_initializer_status, [0.1] * 4):
        assert x == "uninitialized"

    executor3 = get_reusable_executor(max_workers=4, reuse=False, kill_workers=True)
    for x in executor3.map(return_initializer_status, [0.1] * 4):
        assert x == "uninitialized"

    executor4 = get_reusable_executor(max_workers=4)
    for x in executor4.map(return_initializer_status, [0.1] * 4):
        assert x == "uninitialized"