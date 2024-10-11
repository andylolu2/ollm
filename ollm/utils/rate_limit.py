import asyncio
from collections import deque
from datetime import datetime, timedelta


class Resource:
    def __init__(self, period: timedelta = timedelta(seconds=1), limit: float = 1):
        """Represent a resource that re-plenishes over time.

        Args:
            period (timedelta): The time period which the resource re-plenishes.
            limit (int): The amount of resource that re-plenishes.
        """
        self.period = period
        self.limit = limit
        self.min_wait = period / limit
        self.last_replenish = datetime.now()

        self.resources = limit
        self.waiters: deque[tuple[float, asyncio.Future]] = deque()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.release()

    async def replenish(self):
        """Periodically replinish the resource."""

        while True:
            self.resources = self.limit
            self.last_replenish = datetime.now()
            await self.wake_waiters()
            next_replenish = self.last_replenish + self.period
            await asyncio.sleep((next_replenish - datetime.now()).total_seconds())

    async def wake_waiters(self):
        """Wake up as many waiters as possible."""
        while self.waiters:
            resource_requested, waiter = self.waiters[0]
            if resource_requested > self.resources:
                break
            else:
                self.waiters.popleft()
                # self.resources -= resource_requested
                waiter.set_result(None)
                await asyncio.sleep(self.min_wait.total_seconds())

    async def acquire(self, resource_requested: int = 1):
        """Acquire the resource.

        Args:
            resource_requested (int, optional): Amount of resource requested. Defaults to 1.
        """
        if resource_requested > self.limit:
            raise ValueError(f"Requested more than the limit: {resource_requested}")

        if resource_requested > self.resources:
            waiter = asyncio.get_running_loop().create_future()
            self.waiters.append((resource_requested, waiter))
            await waiter

        self.resources -= resource_requested

    async def release(self, resource_released: int = 1):
        """Release the resource.

        Args:
            resource_released (int, optional): Amount of resource released. Defaults to 1.
        """
        self.resources += resource_released
        await self.wake_waiters()


if __name__ == "__main__":

    async def main():
        resource = Resource(timedelta(seconds=1), 2)

        async def worker(i):
            await asyncio.sleep(0.1)
            print(f"Worker {i} waiting")
            await resource.acquire()
            print(f"Worker {i} acquired")
            await asyncio.sleep(0.5)
            print(f"Worker {i} done")

        await asyncio.gather(*(worker(i) for i in range(10)))

    asyncio.run(main())
