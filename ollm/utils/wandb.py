from typing import overload

import wandb
from wandb.apis.public import Run


def get_run_id(name: str) -> int:
    return int(name.split("-")[-1])


@overload
def load_runs(run_ids: str) -> Run: ...


@overload
def load_runs(run_ids: list[str]) -> list[Run]: ...


def load_runs(run_ids: list[str] | str):
    api = wandb.Api()  # type: ignore
    singular = isinstance(run_ids, str)

    if singular:
        run_ids = [run_ids]  # type: ignore

    runs = api.runs(
        path="andylolu2/llm-ol",
        filters={"display_name": {"$regex": rf"^({'|'.join(map(str, run_ids))})$"}},
    )

    if len(runs) != len(run_ids):
        run_names = [run.name for run in runs]
        raise ValueError(f"Request: {run_ids}, response: {run_names}")

    if singular:
        return runs[0]
    else:
        return dict(zip(run_ids, runs))
