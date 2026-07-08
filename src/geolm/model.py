"""Geometric core: token points, recurrent trajectory, nearest-point output."""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


@dataclass
class GeoConfig:
    """Model configuration. Stored inside the model file next to the weights."""
    vocab_size: int
    dim: int = 128        # dimensionality of the point cloud (the geometry)
    hidden: int = 512     # memory of the recurrent engine
    layers: int = 2       # number of recurrent layers
    dropout: float = 0.1
    max_new: int = 64     # generation length limit for answers

    def to_dict(self):
        return asdict(self)


class GeoModel(nn.Module):
    """Every token is a learnable point in a low-dimensional space.

    A sequence is read by a recurrent engine as a trajectory through those points.
    The model predicts the coordinates of the next point, and the next token is
    chosen by proximity to the points of a shared cloud. The same cloud serves
    both input and output, so there is no separate output vocabulary matrix.
    """

    def __init__(self, cfg: GeoConfig):
        super().__init__()
        self.cfg = cfg
        self.points = nn.Embedding(cfg.vocab_size, cfg.dim)
        nn.init.normal_(self.points.weight, std=0.1)
        self.rnn = nn.GRU(
            cfg.dim, cfg.hidden, num_layers=cfg.layers, batch_first=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        self.proj = nn.Linear(cfg.hidden, cfg.dim)
        self.log_scale = nn.Parameter(torch.zeros(1))

    def trajectory(self, tokens: torch.Tensor, state=None):
        """tokens [B, T] to predicted points [B, T, dim] plus recurrent state."""
        out, state = self.rnn(self.points(tokens), state)
        return self.proj(out), state

    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        """Points [..., dim] to vocabulary logits via negative squared distance."""
        emb = self.points.weight
        d2 = (feats * feats).sum(-1, keepdim=True) - 2 * feats @ emb.t() + (emb * emb).sum(-1)
        return -d2 * torch.exp(self.log_scale)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        feats, _ = self.trajectory(tokens)
        return self.logits(feats)


def pick_device(requested: str = "auto") -> torch.device:
    """auto: cuda when available, otherwise cpu (the reliable choice for GRU)."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
