"""This file defines the top-down bottom-up encoder block as detailed in https://arxiv.org/pdf/2203.07586v1.pdf."""
from torch import nn, Tensor
from torch.nn import functional as F

from .LocalAttention import LocalSelfAttention


class BottomUpTopDownEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, val: Tensor) -> Tensor:
        ...
