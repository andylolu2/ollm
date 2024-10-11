import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import dotenv
from absl import app, flags, logging

from ollm.utils import Resource, batch, setup_logging

dotenv.load_dotenv()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "output_dir", None, "Directory to save output files", required=True, short_name="o"
)
flags.DEFINE_string(
    "date_min",
    None,
    "Minimum date of papers to get citations for. Format: YYYY-MM-DD",
    required=True,
)
flags.DEFINE_string(
    "date_max",
    None,
    "Maximum date of papers to get citations for. Format: YYYY-MM-DD",
    required=True,
)

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
semantic_scholar_limit = Resource(period=timedelta(seconds=1), limit=1)


async def api_request(session: aiohttp.ClientSession, retries: int = 3, **kwargs):
    for i in range(retries):
        try:
            await semantic_scholar_limit.acquire()
            async with session.post(SEMANTIC_SCHOLAR_API_URL, **kwargs) as response:
                if response.status == 429:  # Too many requests
                    raise RuntimeError("Too many requests")
                result = await response.json()
                return result
        except Exception as e:
            if i == retries - 1:
                raise e
            else:
                logging.error("Request failed: %s. %d retries left", e, retries - i - 1)
                await asyncio.sleep(2**i)
    assert False  # Unreachable


def download_arxiv(save_dir: Path):
    file_path = save_dir / "arxiv-metadata-oai-snapshot.json"
    if not file_path.exists():
        import kaggle  # Lazy import so that dotenv is loaded

        logging.info("Downloading arXiv dataset")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "Cornell-University/arxiv", path=save_dir, unzip=True
        )
    return file_path


def preprocess_text(text: str) -> str:
    lines = text.split("\n")
    lines = list(filter(lambda x: x != "", map(lambda x: x.strip(), lines)))
    text = " ".join(lines)
    return text


def preprocess_arxiv(item: dict):
    # Get date
    # version = {"version": "v1", "created": "Mon, 1 Jan 0000 00:00:00 GMT"}
    first_version = min(item["versions"], key=lambda x: int(x["version"][1:]))
    date = datetime.strptime(first_version["created"], "%a, %d %b %Y %H:%M:%S %Z")

    return {
        "id": item["id"],
        "title": preprocess_text(item["title"]),
        "categories": item["categories"].split(" "),
        "abstract": preprocess_text(item["abstract"]),
        "date": date,
    }


def load_papers(
    file_path: Path, date_min: datetime, date_max: datetime, cache_dir: Path
):
    cache_file = cache_dir / "papers.jsonl"
    papers = []
    if cache_file.exists():
        logging.info("Loading papers from %s", cache_file)
        with open(cache_file, "r") as f:
            for line in f:
                papers.append(json.loads(line))
    else:
        logging.info("Building papers from %s", file_path)
        with open(file_path, "r") as f_in, open(cache_file, "w") as f_out:
            for line in f_in:
                item = json.loads(line)
                item = preprocess_arxiv(item)
                if date_min <= item["date"] <= date_max:
                    f_out.write(json.dumps(item, default=str) + "\n")
                    papers.append(item)

    logging.info("Loaded %d papers", len(papers))
    return papers


async def get_papers_with_citations(papers: list[dict], cache_dir: Path):
    cache_file = cache_dir / "papers_with_citations.jsonl"

    if cache_file.exists():
        logging.info("Results file %s exists. Skipping...", cache_file)
        return

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:

        async def coro(paper_batch):
            paper_ids = [f"ARXIV:{paper['id']}" for paper in paper_batch]
            response = await api_request(
                session,
                params={"fields": "citationCount"},
                json={"ids": paper_ids},
                headers={"x-api-key": os.getenv("SEMANTIC_SCHOLAR_API_KEY")},
            )
            logging.debug("Response: %s", response)
            for paper, result in zip(paper_batch, response):
                if result is None:
                    logging.info(
                        "No citation data for %s: %s",
                        paper["id"],
                        paper["title"],
                    )
                    continue
                item = {"citation_count": result["citationCount"], **paper}
                with open(cache_file, "a") as f:
                    f.write(json.dumps(item, default=str) + "\n")

        coros = [coro(paper_batch) for paper_batch in batch(papers, 500)]
        await asyncio.gather(*coros)


async def main(_):
    # Set up
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, "build_pages", flags=FLAGS)
    asyncio.create_task(semantic_scholar_limit.replenish())

    # Get arXiv data
    arxiv_file = download_arxiv(out_dir / "raw")
    papers = load_papers(
        arxiv_file,
        datetime.strptime(FLAGS.date_min, "%Y-%m-%d"),
        datetime.strptime(FLAGS.date_max, "%Y-%m-%d"),
        out_dir,
    )
    await get_papers_with_citations(papers, out_dir)


if __name__ == "__main__":
    app.run(lambda _: asyncio.run(main(_)))
