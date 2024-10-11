from functools import cache

import torch
from absl import logging
from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


@cache
def load_embedding_model(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    logging.info("Loading embedding model %s", name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, device_map=device)
    return model, tokenizer


@torch.no_grad()
def embed(text: str | list[str], model, tokenizer, variant: str = "mean"):
    is_single = isinstance(text, str)
    if is_single:
        text = [text]

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    outputs = model(**inputs)
    if variant == "cls":
        embed = outputs.last_hidden_state[:, 0, :]
    elif variant == "mean":
        embed = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    else:
        raise ValueError(f"Invalid variant: {variant}")

    if is_single:
        embed = embed[0]
    return embed


def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    )
    return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
