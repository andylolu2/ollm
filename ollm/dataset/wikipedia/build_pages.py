import asyncio
import json
from pathlib import Path
from typing import Iterable

import aiohttp
from absl import app, flags, logging

from ollm.dataset import wikipedia
from ollm.utils import batch, setup_logging, textqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "categories_file",
    None,
    "File containing categories to get summaries for",
    required=True,
    short_name="c",
)
flags.DEFINE_string(
    "output_dir", None, "Directory to save output files", required=True, short_name="o"
)


async def get_pages_abstract(page_ids: set[int], out_file: Path):
    """Get summaries of pages.

    API reference: https://www.mediawiki.org/w/api.php?action=help&modules=query%2Bextracts
    """

    async def get_pages_summary(
        page_ids_batch: Iterable[int], session: aiohttp.ClientSession
    ):
        last_continue: dict = {}
        while True:
            params = {
                "action": "query",
                "pageids": "|".join(map(str, page_ids_batch)),
                "prop": "extracts",
                "format": "json",
                "formatversion": "2",
                "explaintext": "true",
                "exintro": "true",
                **last_continue,
            }
            result = await wikipedia.api_request(session, params)
            if "error" in result:
                raise RuntimeError(result["error"])
            if "warnings" in result:
                logging.warning(result["warnings"])
            if "query" in result:
                with open(out_file, "a") as f:
                    for page in result["query"]["pages"]:
                        if page.get("missing", False):
                            continue
                        if page.get("extract", "") == "":
                            continue
                        item = {
                            "id": page["pageid"],
                            "title": page["title"],
                            "abstract": page["extract"],
                        }
                        f.write(json.dumps(item) + "\n")
            if "continue" not in result:
                return
            last_continue = result["continue"]

    prev_results = set()
    if out_file.exists():
        with open(out_file, "r") as f:
            for line in f:
                item = json.loads(line)
                prev_results.add(item["id"])
    logging.info("Loaded %s seen pages", len(prev_results))
    page_ids = page_ids - prev_results

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        tasks = []
        for page_ids_batch in batch(textqdm(page_ids), 50):
            tasks.append(get_pages_summary(page_ids_batch, session))
        await asyncio.gather(*tasks)


async def async_main(_):
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, "build_pages", flags=FLAGS)

    asyncio.create_task(wikipedia.api_limit.replenish())

    # Get page ids to request from previous raw results
    categories_file = Path(FLAGS.categories_file)
    page_ids = set()
    with open(categories_file, "r") as f:
        for line in f:
            page_ids.update(json.loads(line)["pages"])
    logging.info("Getting summaries for %s pages", len(page_ids))

    await get_pages_abstract(page_ids, out_dir / "raw_pages.jsonl")


def main(_):
    asyncio.run(async_main(_))


if __name__ == "__main__":
    app.run(main)
