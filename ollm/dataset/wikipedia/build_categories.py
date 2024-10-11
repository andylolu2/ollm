import asyncio
import json
from pathlib import Path

import aiohttp
from absl import app, flags, logging

from ollm.dataset import wikipedia
from ollm.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_integer("max_depth", 2, "Max depth to traverse", short_name="d")
flags.DEFINE_string(
    "output_dir", None, "Directory to save output files", required=True, short_name="o"
)


async def get_pages_and_subcats(out_categories_file: Path, max_depth: int = 0):
    """Recursively get all pages and subcategories of a category.

    API reference: https://www.mediawiki.org/wiki/Special:MyLanguage/API:Categorymembers
    """

    seen = set()

    prev_categories: dict[int, dict] = {}
    if out_categories_file.exists():
        with open(out_categories_file, "r") as f:
            for line in f:
                item = json.loads(line)
                prev_categories[item["id"]] = item
    logging.info("Loaded %s seen categories", len(prev_categories))

    async def get_category_members(
        category_id: int, category_name: str, session: aiohttp.ClientSession
    ) -> dict:
        if category_id in prev_categories:
            return prev_categories[category_id]

        logging.info("Getting category members for %s (%s)", category_name, category_id)

        pages: list[int] = []
        sub_categories: list[dict] = []

        last_continue: dict = {}
        for _ in range(10):  # Get at most 10x500 items
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmpageid": category_id,
                "cmtype": "page|subcat",
                "cmprop": "ids|title|type",
                "format": "json",
                "formatversion": "2",
                "cmlimit": "max",
                **last_continue,
            }
            result = await wikipedia.api_request(session, params)

            if "error" in result:
                raise RuntimeError(result["error"])
            if "warnings" in result:
                logging.warning(result["warnings"])
            if "query" in result:
                for page in result["query"]["categorymembers"]:
                    if page.get("missing", False):
                        continue
                    if page["type"] == "page":
                        pages.append(page["pageid"])
                    elif page["type"] == "subcat":
                        sub_categories.append(
                            {
                                "id": page["pageid"],
                                "title": page["title"].removeprefix("Category:"),
                            }
                        )
                    else:
                        raise RuntimeError("Unknown page type: %s", page["type"])

            if "continue" not in result:
                break
            last_continue = result["continue"]

        return {
            "id": category_id,
            "title": category_name,
            "pages": pages,
            "sub_categories": sub_categories,
        }

    async def task(
        depth: int,
        category_id: int,
        category_name: str,
        session: aiohttp.ClientSession,
        task_group: asyncio.TaskGroup,
    ):
        category_item = await get_category_members(category_id, category_name, session)

        if category_id not in prev_categories:
            with open(out_categories_file, "a") as f:
                f.write(json.dumps(category_item) + "\n")

        for item in category_item["sub_categories"]:
            subcategory_id = item["id"]
            if subcategory_id not in seen and depth + 1 <= max_depth:
                seen.add(subcategory_id)
                task_group.create_task(
                    task(depth + 1, subcategory_id, item["title"], session, task_group)
                )

    seen.add(wikipedia.ROOT_CATEGORY_ID)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session, asyncio.TaskGroup() as task_group:
        await task(
            0,
            wikipedia.ROOT_CATEGORY_ID,
            wikipedia.ROOT_CATEGORY_NAME,
            session,
            task_group,
        )


async def async_main(_):
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, "build_categories", flags=FLAGS)

    asyncio.create_task(wikipedia.api_limit.replenish())

    raw_categories_file = out_dir / "raw_categories.jsonl"
    await get_pages_and_subcats(raw_categories_file, max_depth=FLAGS.max_depth)

    logging.info("Collected raw categories")


def main(_):
    asyncio.run(async_main(_))


if __name__ == "__main__":
    app.run(main)
