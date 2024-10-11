from itertools import islice
from typing import Iterable, TypeVar

T = TypeVar("T")


def batch(it: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    """Batch an iterable into chunks of size `size`."""
    it = iter(it)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch
