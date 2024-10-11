import asyncio

import aiohttp
from absl import logging


async def wait_for_endpoint(url: str):
    # wait for the server to start
    async with aiohttp.ClientSession() as session:
        while True:
            logging.info("Waiting for %s to start", url)
            try:
                async with session.get(url) as resp:
                    break
            except aiohttp.ClientConnectorError:
                await asyncio.sleep(5)
    logging.info("Server started")
