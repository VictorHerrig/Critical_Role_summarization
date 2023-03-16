import math

import torch
from torch import nn, Tensor

from .BottomUpTopDownEncoder import BottomUpTopDownEncoder


# TODO: Taken from some torch tutorial code: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class BottomUpTopDownTransformer(nn.Module):
    def __init__(
            self,
            model_dim: int,
            num_decoder_layers: int = 12,
            local_self_attn_window_size: int = 1024,
            feedforward_dim: int = 2048,
            num_local_self_attn: int = 8,
            num_segment_full_self_attn: int = 2,
            num_top_down_blocks: int = 4,
            num_attn_heads: int = 8,
            dropout: float = 0.1,
            avg_pool_kernel_size: int = 32,
            avg_pool_stride: int = 24,
            max_len: int = 10000,  # TODO: Examine this in particular
            device: str = None
    ):
        super().__init__()
        self._pos_encoding = PositionalEncoding(
            model_dim,
            dropout=dropout,
            max_len=max_len
        )
        self._encoder = BottomUpTopDownEncoder(
            model_dim,
            local_self_attn_window_size,
            feedforward_dim,
            num_local_self_attn,
            num_segment_full_self_attn,
            num_top_down_blocks,
            num_attn_heads,
            dropout,
            avg_pool_kernel_size,
            avg_pool_stride,
            device
        )
        self._model = nn.Transformer(
            d_model=model_dim,
            nhead=num_attn_heads,
            custom_encoder=self._encoder,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            device=device
        )

    def forward(
            self,
            src: Tensor,
            tgt: Tensor,
            tgt_mask: Tensor = None,
            src_key_padding_mask: Tensor = None,
            tgt_key_padding_mask: Tensor = None
    ) -> Tensor:
        return self.model.forward(self._pos_encoding(src), self._pos_encoding(tgt), tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

    @staticmethod
    def generate_square_subsequent_mask(sz, device='cpu'):
        return nn.Transformer.generate_square_subsequent_mask(sz, device=device)

    @property
    def model(self):
        return self._model
