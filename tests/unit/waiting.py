"""What a unit test waits with when another thread, task or process has to get somewhere first.

A passing test never waits FAILURE_BOUND out: it is how long a broken one takes to fail, so it is
generous — a CI runner stalls for seconds, and a correct test behind a 2 s bound failed on one
(2026-11-19). A wait that is expected to run out (nothing may happen meanwhile) is the opposite
case: it is paid in full by every passing run, stays a few milliseconds and takes its own number."""
import time

FAILURE_BOUND = 30.0


def wait_until(predicate, timeout: float = FAILURE_BOUND, interval: float = 0.005) -> bool:
    """True as soon as *predicate()* is, False once *timeout* seconds have passed without it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())
