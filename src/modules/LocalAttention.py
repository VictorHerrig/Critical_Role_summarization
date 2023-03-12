"""
Adapted from the Longformer code developed by the Allen Institute for AI under the Apache License:
https://github.com/allenai/longformer
"""
import math

import torch
from torch.nn import functional as F
from torch import nn

from .DiagonaledMM import diagonaled_mm


class LocalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, window_size: int = 1024, num_heads: int = 8, dropout: float = 0.1):
        """

        Parameters
        ----------
        hidden_size: int
            ...
        window_size: int
            ...
        num_heads: int
            ...
        dropout: float
            ...
        """
        super().__init__()

        assert window_size > 0
        if hidden_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_heads))

        self._window_size = window_size
        self._num_heads = num_heads
        self._head_dim = int(hidden_size / num_heads)
        self._embed_dim = hidden_size

        self._query = nn.Linear(hidden_size, self.embed_dim)
        self._key = nn.Linear(hidden_size, self.embed_dim)
        self._value = nn.Linear(hidden_size, self.embed_dim)

        self._dropout = dropout

    def forward(self, val: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        val: torch.Tensor
            Tensor of shape (sequence_length x batch_size x hidden_dim).

        Returns
        -------
        Tensor
            Attention output tensor of shape (sequence_length x batch_size x hidden_dim)
        """
        transposed_val = val.transpose(0, 1)
        seq_len, bsz, embed_dim = val.size()

        # Linear projection onto multiple heads
        q = self.query(transposed_val) / math.sqrt(self.head_dim)  # Scale Q by sqrt(head_dim) ahead of time
        k = self.key(transposed_val)
        v = self.value(transposed_val)
        multihead_q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).float().contiguous()
        multihead_k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).float().contiguous()
        multihead_v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).float().contiguous()

        # Q x K using diagonaled mat mul
        attn_weights = diagonaled_mm(multihead_q, multihead_k, self.attention_window, torch.ones(self.num_heads), False, 0, False)

        # TODO: Masking ... if needed
        # Apply mask(s)
        masked_attn_weights = attn_weights

        # Softmax of masked, scaled Q x K
        attn_weights_float = F.softmax(masked_attn_weights, dim=-1, dtype=torch.float32)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)

        # Matmul attention probs and value
        attn = diagonaled_mm(attn_probs, multihead_v, self.attention_window, torch.ones(self.num_heads), True, 0, False).type_as(val)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        output_attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()

        # TODO: Context layer

        return output_attn

    @property
    def window_size(self):
        return self._window_size

    @property
    def query(self):
        return self._query

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    @property
    def num_heads(self):
        return self._num_heads

    @property
    def head_dim(self):
        return self._head_dim

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def dropout(self):
        return self._dropout
