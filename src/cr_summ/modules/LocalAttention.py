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
            device: str = 'cpu',
            use_tvm: bool = False
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
        use_tvm: bool, optional
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
        # self._softmax = nn.Softmax(-1)

        self._autoregressive = autoregressive
        assert self._window_size > 0

        self._dropout = dropout
        self._use_tvm = use_tvm
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
            Tensor of shape (batch_size, sequence_length) containing 1 where input sequence is padded and 0 everywhere
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

        if self.use_tvm:
            # Q x K using diagonaled mat mul
            attn_weights = diagonaled_mm(multihead_q, multihead_k, self.window_size, self.dilation, False, 0, False)
            mask_invalid_locations(attn_weights, self.window_size, self.dilation, self.autoregressive)
        else:
            # Q x K using sliding chunks mat mul
            attn_weights = self._sliding_chunks_matmul(multihead_q, multihead_k, self.window_size // 2)

        if key_padding_mask is not None:
            # # This implementation is fast and takes very little memory because num_heads x hidden_size = 1  <-- ?????
            # # from (bsz, seq_len) to (bsz, seq_len, num_heads, hidden_size)
            # # key_padding_mask = key_padding_mask.unsqueeze(dim=-1).unsqueeze(dim=-1).view(seq_len, bsz)
            # key_padding_mask = key_padding_mask.view(bsz, seq_len, 1, 1)#.transpose(0, 1)
            # # cast to float/half then replace 1s with -inf
            # float_mask = key_padding_mask.type_as(q).masked_fill(key_padding_mask, -10000.0)
            # # float_mask = float_mask.repeat(1, 1, 1, 1)
            # all_ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # # diagonal mask with zeros everywhere and -inf inplace of padding
            # d_mask = diagonaled_mm(all_ones, float_mask, self.window_size, self.dilation, False, 0, False)
            #
            # attn_weights += d_mask

            # Changed to the more slightly efficient Huggingface implementation
            # values to pad for attention probs
            remove_from_windowed_attention_mask = (key_padding_mask != 0)[:, :, None, None]

            # cast to fp32/fp16 then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(multihead_q).masked_fill(
                remove_from_windowed_attention_mask, torch.finfo(multihead_q.dtype).min
            )
            if self.use_tvm:
                diagonal_mask = diagonaled_mm(
                    float_mask.new_ones(size=float_mask.size()),
                    float_mask, self.window_size, self.dilation, False, 0, False
                )
            else:
                # diagonal mask with zeros everywhere and -inf inplace of padding
                diagonal_mask = self._sliding_chunks_matmul(
                    float_mask.new_ones(size=float_mask.size()), float_mask, self.window_size // 2
                )

            # pad local attention probs
            attn_weights += diagonal_mask

        assert list(attn_weights.size()) == [
            bsz,
            seq_len,
            self.num_heads,
            self.window_size + 1,
        ], (
            f"local_attn_probs should be of size ({bsz}, {seq_len}, {self.num_heads},"
            f" {self.window_size + 1}), but is of size {attn_weights.size()}"
        )

        # Softmax of masked, scaled Q x K
        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)

        # Matmul attention probs and value
        if self.use_tvm:
            attn = diagonaled_mm(attn_probs, multihead_v, self.window_size, self.dilation, True, 0, False).type_as(val)
        else:
            attn = self._sliding_chunks_matmul_pv(attn_probs, multihead_v, self.window_size // 2)

        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        output_attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()  # (seq_len, batch, embed_dim)

        return output_attn

    # In the case the TVM function cannot be compiled, or blows up your GPU memory...
    # These four functions are taken more or less straight-up from the Huggingface Transformers implementation
    @staticmethod
    def _mask_invalid_locations_sliding_chunks(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
            beginning_input, -float("inf")
        ).where(beginning_mask.bool(), beginning_input)
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
            ending_input, -float("inf")
        ).where(ending_mask.bool(), ending_input)

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = nn.functional.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
        )
        return hidden_states_padded

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # non-overlapping chunks of size = 2w
        hidden_states = hidden_states.view(
            hidden_states.size(0),
            torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
            window_overlap * 2,
            hidden_states.size(2),
        )
        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(hidden_states.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(hidden_states.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

    def _sliding_chunks_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
                seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()

        chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1

        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        query = self._chunk(query, window_overlap)
        key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multipl

        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = \
            diagonal_chunked_attention_scores[:, :, :window_overlap, : window_overlap + 1]
        diagonal_attention_scores[:, -1, :, window_overlap:] = \
            diagonal_chunked_attention_scores[:, -1, window_overlap:, : window_overlap + 1]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = \
            diagonal_chunked_attention_scores[:, :, -(window_overlap + 1): -1, window_overlap + 1:]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = \
            diagonal_chunked_attention_scores[:, 0, : window_overlap - 1, 1 - window_overlap:]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        # TODO: Check this should be here and not in the outer region
        self._mask_invalid_locations_sliding_chunks(diagonal_attention_scores, self.window_size // 2)

        return diagonal_attention_scores

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example:
        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```
                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        chunked_hidden_states = nn.functional.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    def _sliding_chunks_matmul_pv(
            self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        batch_size, seq_len, num_heads, head_dim = value.size()

        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads,
            torch.div(seq_len, window_overlap, rounding_mode="trunc"),
            window_overlap,
            2 * window_overlap + 1,
        )

        # group batch_size and num_heads dimensions into one
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

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

    @property
    def use_tvm(self):
        return self._use_tvm
