import torch


def cosine_sim(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Cosine sim between x and y."""
    x = x / x.norm(dim=dim, keepdim=True)
    y = y / y.norm(dim=dim, keepdim=True)
    sim = torch.tensordot(x, y, dims=([dim], [dim]))  # type: ignore
    return sim


def scaled_cosine_sim(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Cosine sim between x and y, scaled to [0, 1] instead of [-1, 1]."""
    sim = cosine_sim(x, y, dim=dim)
    return (sim + 1) / 2
