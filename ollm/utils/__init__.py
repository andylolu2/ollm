import torch

from .data import batch
from .embedding import embed, load_embedding_model
from .http import wait_for_endpoint
from .jinja import load_template
from .logging import log_flags, setup_logging
from .plotting import sized_subplots
from .rate_limit import Resource
from .textqdm import textpbar, textqdm
from .torch_utils import cosine_sim, scaled_cosine_sim
from .types import Graph, PathLike
from .wandb import get_run_id, load_runs

device = "cuda" if torch.cuda.is_available() else "cpu"

__all__ = [
    "batch",
    "cosine_sim",
    "device",
    "get_run_id",
    "Graph",
    "load_runs",
    "load_template",
    "log_flags",
    "PathLike",
    "Resource",
    "scaled_cosine_sim",
    "setup_logging",
    "sized_subplots",
    "textpbar",
    "textqdm",
    "wait_for_endpoint",
    "load_embedding_model",
    "embed",
]
