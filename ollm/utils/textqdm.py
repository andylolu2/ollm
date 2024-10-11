from collections.abc import Sized
from time import time
from typing import Iterable, TypeVar

from absl import logging

T = TypeVar("T")


class textpbar:
    """A text-based version of a progress bar."""

    def __init__(self, total: int | None = None, period: float = 10.0):
        self.total = total
        self.period = period
        self.i = 0
        self.last_log = time()
        self.start = time()

    def update(self, n: int = 1):
        self.i += n
        if time() - self.last_log > self.period:
            self.last_log = time()
            rate = self.i / (time() - self.start)
            if self.total is not None:
                logging.info(
                    "Progress: %d / %d %.2f%% (Avg. rate: %.2f it/s)",
                    self.i,
                    self.total,
                    self.i / self.total * 100,
                    rate,
                )
            else:
                logging.info("Progress: %d (Avg. rate: %.2f it/s)", self.i, rate)


def textqdm(
    iterable: Iterable[T],
    total: int | None = None,
    period: float = 10.0,
) -> Iterable[T]:
    """A text-based version of tqdm."""
    if total is None and isinstance(iterable, Sized):
        total = len(iterable)

    pbar = textpbar(total, period)
    for item in iterable:
        pbar.update()
        yield item
