"""
Adapted from the Longformer code developed by the Allen Institute for AI under the Apache License:
https://github.com/allenai/longformer
"""
import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .DiagonaledMM import diagonaled_mm, mask_invalid_locations


class LocalSelfAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            window_size: int = 1024,
            num_heads: int = 8,
            dropout: float = 0.1,
            autoregressive: bool = True,
            device: str = 'cpu'
    ):
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
        device: str
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
        self._embed_dim = hidden_size

        self._query = nn.Linear(hidden_size, hidden_size, device=device)
        self._key = nn.Linear(hidden_size, hidden_size, device=device)
        self._value = nn.Linear(hidden_size, hidden_size, device=device)
        self._softmax = nn.Softmax(hidden_size)

        self._autoregressive = autoregressive
        assert self._window_size > 0

        self._dropout = dropout
        self.to(device)

    def forward(
            self,
            val: Tensor,
            key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """...

        Parameters
        ----------
        val: Tensor
            Tensor of shape (sequence_length, batch_size, hidden_dim).
        key_padding_mask: Tensor, optional
            Tensor of shape (sequence_length, batch_size) containing 1 where input sequence is padded and 0 everywhere
            else. If None, no masking will be used. (Default = None)

        Returns
        -------
        Tensor
            Attention output tensor of shape (sequence_length, batch_size, hidden_dim)
        """
        #transposed_val = val.transpose(0, 1)  # (batch_size, sequence_length, hidden_dim)
        seq_len, bsz, embed_dim = val.size()

        # Linear projection onto multiple heads
        q = self.query(val) / math.sqrt(self.head_dim)  # Scale Q by sqrt(head_dim) ahead of time
        k = self.key(val)
        v = self.value(val)
        # Diagonaled matmul wants (batch, seq_len, n_head, head_dim) so we transpose axes 0 and 1
        multihead_q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).float().contiguous()
        multihead_k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).float().contiguous()
        multihead_v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).float().contiguous()

        # Q x K using diagonaled mat mul
        attn_weights = diagonaled_mm(multihead_q, multihead_k, self.window_size, self.dilation, False, 0, False)

        # Apply attention mask
        mask_invalid_locations(attn_weights, self.window_size, self.dilation, self.autoregressive)
        if key_padding_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz, seq_len) to (bsz, seq_len, num_heads, hidden_size)
            key_padding_mask = key_padding_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # cast to float/half then replace 1s with -inf
            float_mask = key_padding_mask.type_as(q).masked_fill(key_padding_mask, -10000.0)
            float_mask = float_mask.repeat(1, 1, 1, 1)
            all_ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            d_mask = diagonaled_mm(all_ones, float_mask, self.window_size, self.dilation, False, 0, False)

            attn_weights += d_mask

        assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]
        assert attn_weights.size(dim=3) in [self.window_size * 2 + 1, self.window_size * 3]

        # Softmax of masked, scaled Q x K
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)

        # Matmul attention probs and value
        attn = diagonaled_mm(attn_probs, multihead_v, self.window_size, self.dilation, True, 0, False).type_as(val)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        output_attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()  # (seq_len, batch, embed_dim)

        return output_attn

    @property
    def window_size(self):
        return self._window_size

    @property
    def dilation(self):
        return 1

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
        return int(self.embed_dim / self.num_heads)

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def dropout(self):
        return self._dropout

    @property
    def autoregressive(self):
        return self._autoregressive
