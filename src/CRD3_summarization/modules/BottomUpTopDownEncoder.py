"""This file defines the top-down bottom-up encoder block as detailed in https://arxiv.org/pdf/2203.07586v1.pdf."""
from collections import OrderedDict
from typing import Optional

import torch
from torch import nn, Tensor

from .LocalAttention import LocalSelfAttention


# ********************************************************* #
# ****************** Encoder sub-blocks ******************* #
# ********************************************************* #


class FullSelfAttentionBlock(nn.Module):
    def __init__(
            self,
            model_dim: int,
            num_attn_heads: int = 8,
            dropout: float = 0.1,
            device: str = 'cpu'
    ):
        super().__init__()
        self._full_self_attn = nn.MultiheadAttention(model_dim, num_heads=num_attn_heads, dropout=dropout, device=device)
        self._layer_norm = nn.LayerNorm([model_dim, ], device=device)
        self.to(device)

    def forward(
            self,
            val: Tensor,
            key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        attn_out = self._full_self_attn.forward(val, val, val, key_padding_mask=key_padding_mask)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        return self._layer_norm(val + attn_out)


class LocalSelfAttentionBlock(nn.Module):
    def __init__(
            self,
            model_dim: int,
            local_self_attn_window_size: int = 1024,
            num_attn_heads: int = 8,
            dropout: float = 0.1,
            device: str = 'cpu'
    ):
        super().__init__()
        self._local_self_attn = LocalSelfAttention(model_dim,
                                                   local_self_attn_window_size,
                                                   num_heads=num_attn_heads,
                                                   dropout=dropout,
                                                   device=device)

        self._layer_norm = nn.LayerNorm([model_dim, ], device=device)
        self.to(device)

    def forward(
            self,
            val: Tensor,
            key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        key_padding_mask = key_padding_mask.to(torch.bool) if key_padding_mask is not None else None
        attn_out = self._local_self_attn(val, key_padding_mask=key_padding_mask)
        return self._layer_norm(val + attn_out)


class TopDownEncoderSubBlock(nn.Module):
    def __init__(
            self,
            model_dim: int,
            feedforward_dim: int = 2048,
            local_self_attn_window_size: int = 1024,
            num_attn_heads: int = 8,
            dropout: float = 0.1,
            device: str = 'cpu'
    ):
        super().__init__()
        self._n_heads = num_attn_heads
        self._local_self_attn = LocalSelfAttention(model_dim, local_self_attn_window_size, num_attn_heads, dropout, device=device)
        self._full_cross_attn = nn.MultiheadAttention(model_dim, num_heads=num_attn_heads, dropout=dropout, device=device)
        self._layer_norm = nn.LayerNorm([model_dim, ], device=device)
        self._feedforward = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(model_dim, feedforward_dim, device=device)),
            ('relu', nn.ReLU()),  # TODO: GeLU?
            ('linear_2', nn.Linear(feedforward_dim, model_dim, device=device))
        ]))
        self._final_layer_norm = nn.LayerNorm([model_dim, ], device=device)
        self.to(device)

    def forward(
            self,
            val: Tensor,
            top_level_tokens: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            top_lvl_key_mask: Optional[Tensor] = None
    ) -> Tensor:
        key_padding_mask = key_padding_mask.to(torch.bool) if key_padding_mask is not None else None
        token_self_attn = self._local_self_attn(val, key_padding_mask=key_padding_mask)

        token_segment_cross_attn = self._full_cross_attn(token_self_attn,  # Query
                                                         top_level_tokens,  # Key
                                                         top_level_tokens,  # Value
                                                         key_padding_mask=top_lvl_key_mask)
        if isinstance(token_segment_cross_attn, tuple):
            token_segment_cross_attn = token_segment_cross_attn[0]
        # Note! The paper _specifically_ notes the residual connection is applied after the layer norm
        # so the cross attention acts strictly as an update.
        # Mind you, the paper also doesn't specify any other residual or layer norm components which does rather feel
        # silly, so I've added them in the other blocks for the time being anyway. Perhaps I will add a toggle parameter
        # in future.
        cross_attn_layer_norm = token_self_attn + self._layer_norm(token_segment_cross_attn)
        output = self._final_layer_norm(self._feedforward(cross_attn_layer_norm))

        return output


# ********************************************************* #
# ******************** Encoder blocks ********************* #
# ********************************************************* #


class BottomUpEncoderBlock(nn.Module):
    def __init__(
            self,
            model_dim: int,
            local_self_attn_window_size: int = 1024,
            num_attn_heads: int = 8,
            dropout: float = 0.1,
            avg_pool_kernel_size: int = 32,
            avg_pool_stride: int = 24,
            num_local_self_attn: int = 8,
            num_segment_full_self_attn: int = 2,
            device: str = 'cpu'
    ):
        super().__init__()

        assert num_local_self_attn > 0
        assert num_segment_full_self_attn > 0

        self._local_self_attn_stack = nn.ModuleDict({
            f'local_attention_{i}': LocalSelfAttentionBlock(model_dim, local_self_attn_window_size, num_attn_heads, dropout, device)
            for i in range(num_local_self_attn)
        })
        self._avg_pool = nn.AvgPool1d(kernel_size=avg_pool_kernel_size, stride=avg_pool_stride)
        self._mask_max_pool = nn.MaxPool1d(kernel_size=avg_pool_kernel_size, stride=avg_pool_stride)
        self._segment_full_self_attn_stack = nn.ModuleDict({
            f'full_attention_{i}': FullSelfAttentionBlock(model_dim, num_attn_heads, dropout, device)
            for i in range(num_segment_full_self_attn)
        })
        self.to(device)

    def forward(
            self,
            val: Tensor,
            key_padding_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Local self-attention on token-level 'bottom-level' representations
        local_attn_intermdeiate = val
        for block in self._local_self_attn_stack.values():
            local_attn_intermdeiate = block(local_attn_intermdeiate, key_padding_mask=key_padding_mask)
        bottom_lvl_repr = local_attn_intermdeiate

        # Pooling along seq_len to create initial 'top-level' representations
        top_lvl_initializations = self._avg_pool(bottom_lvl_repr.transpose(0, -1)).transpose(0, -1)
        if key_padding_mask is not None:
            # Ad-hoc min pooling (i.e. any pool with an unmasked value remains unmasked)
            top_lvl_key_mask = (1. - self._mask_max_pool((~key_padding_mask).to(torch.float))).to(torch.bool)
        else:
            top_lvl_key_mask = None

        # Full self-attention on segment-level 'top-level' representations
        for block in self._segment_full_self_attn_stack.values():
            top_lvl_initializations = block(top_lvl_initializations, key_padding_mask=top_lvl_key_mask)
        top_lvl_rep = top_lvl_initializations
        return bottom_lvl_repr, top_lvl_rep, top_lvl_key_mask


class TopDownEncoderBlock(nn.Module):
    def __init__(
            self,
            model_dim: int,
            feedforward_dim: int,
            local_self_attn_window_size: int = 1024,
            num_attn_heads: int = 8,
            dropout: float = 0.1,
            num_top_down_blocks: int = 4,
            device: str = 'cpu'
    ):
        super().__init__()

        # Can't be sequential since each sublayer output needs to be the argument to all 3 parameters for the next layer
        self._top_down_subblocks = nn.ModuleDict({
            f'top_down_encoder_block_{i}': TopDownEncoderSubBlock(model_dim, feedforward_dim, local_self_attn_window_size, num_attn_heads, dropout, device)
            for i in range(num_top_down_blocks)
        })
        self.to(device)

    def forward(
            self,
            val: Tensor,
            top_level_tokens: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            top_lvl_key_mask: Optional[Tensor] = None
    ) -> Tensor:
        top_down_intermediate = val
        for block in self._top_down_subblocks.values():
            top_down_intermediate = block(top_down_intermediate, top_level_tokens, key_padding_mask=key_padding_mask, top_lvl_key_mask=top_lvl_key_mask)
        final_tokens = top_down_intermediate

        return final_tokens


# ********************************************************* #
# ************************ Encoder ************************ #
# ********************************************************* #


class BottomUpTopDownEncoder(nn.Module):
    def __init__(
            self,
            model_dim: int,
            local_self_attn_window_size: int = 1024,
            feedforward_dim: int = 2048,
            num_local_self_attn: int = 8,  # 4
            num_segment_full_self_attn: int = 2,  # 2
            num_top_down_blocks: int = 4,  # 2
            num_attn_heads: int = 8,
            dropout: float = 0.1,
            avg_pool_kernel_size: int = 32,
            avg_pool_stride: int = 24,
            device: str = 'cpu'
    ):
        super().__init__()
        self._num_local_self_attn = num_local_self_attn
        self._num_top_down_blocks = num_top_down_blocks
        self._bottom_up_encoder = BottomUpEncoderBlock(
            model_dim,
            local_self_attn_window_size,
            num_attn_heads,
            dropout,
            avg_pool_kernel_size,
            avg_pool_stride,
            num_local_self_attn,
            num_segment_full_self_attn,
            device
        )
        self._top_down_encoder = TopDownEncoderBlock(
            model_dim,
            feedforward_dim,
            local_self_attn_window_size,
            num_attn_heads,
            dropout,
            num_top_down_blocks,
            device
        )
        self.to(device)

    def forward(
            self,
            val: Tensor,
            mask: Optional[Tensor] = None,  # TODO: Do something with this?
            src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        bottom_lvl_repr, top_lvl_repr, top_lvl_key_mask = self._bottom_up_encoder(val, key_padding_mask=src_key_padding_mask)
        final_token_repr = self._top_down_encoder(bottom_lvl_repr, top_lvl_repr, key_padding_mask=src_key_padding_mask, top_lvl_key_mask=top_lvl_key_mask)
        return final_token_repr

    def initialize_from_bart(self, bart_model: dict[str, Tensor]):
        encoder_base = 'encoder.layers'

        # Initialize bottom-up blocks
        for i in range(self._num_local_self_attn):
            # Initialize self-attention
            bart_layer = f'{encoder_base}.{i}.self_attn'
            encoder_layer = self._bottom_up_encoder._local_self_attn_stack[f'local_attention_{i}']

            encoder_layer._local_self_attn._query.weight = nn.Parameter(bart_model[f'{bart_layer}.q_proj.weight'])
            encoder_layer._local_self_attn._query.bias = nn.Parameter(bart_model[f'{bart_layer}.q_proj.bias'])
            encoder_layer._local_self_attn._value.weight = nn.Parameter(bart_model[f'{bart_layer}.v_proj.weight'])
            encoder_layer._local_self_attn._value.bias = nn.Parameter(bart_model[f'{bart_layer}.v_proj.bias'])
            encoder_layer._local_self_attn._key.weight = nn.Parameter(bart_model[f'{bart_layer}.k_proj.weight'])
            encoder_layer._local_self_attn._key.bias = nn.Parameter(bart_model[f'{bart_layer}.k_proj.bias'])
            encoder_layer._layer_norm.weight = nn.Parameter(bart_model[f'{bart_layer}_layer_norm.weight'])
            encoder_layer._layer_norm.bias = nn.Parameter(bart_model[f'{bart_layer}_layer_norm.bias'])

        # Initialize top-down blocks
        for i in range(self._num_local_self_attn, self._num_local_self_attn + self._num_top_down_blocks):
            # Initialize self-attention
            bart_layer = f'{encoder_base}.{i}.self_attn'
            encoder_layer = self._top_down_encoder._top_down_subblocks[f'top_down_encoder_block_{i - self._num_local_self_attn}']

            encoder_layer._local_self_attn._query.weight = nn.Parameter(bart_model[f'{bart_layer}.q_proj.weight'])
            encoder_layer._local_self_attn._query.bias = nn.Parameter(bart_model[f'{bart_layer}.q_proj.bias'])
            encoder_layer._local_self_attn._value.weight = nn.Parameter(bart_model[f'{bart_layer}.v_proj.weight'])
            encoder_layer._local_self_attn._value.bias = nn.Parameter(bart_model[f'{bart_layer}.v_proj.bias'])
            encoder_layer._local_self_attn._key.weight = nn.Parameter(bart_model[f'{bart_layer}.k_proj.weight'])
            encoder_layer._local_self_attn._key.bias = nn.Parameter(bart_model[f'{bart_layer}.k_proj.bias'])
            encoder_layer._layer_norm.weight = nn.Parameter(bart_model[f'{bart_layer}_layer_norm.weight'])
            encoder_layer._layer_norm.bias = nn.Parameter(bart_model[f'{bart_layer}_layer_norm.bias'])

            # Initialize feed-forward
            encoder_layer._feedforward[0].weight = nn.Parameter(bart_model[f'{encoder_base}.{i}.fc1.weight'])
            encoder_layer._feedforward[0].bias = nn.Parameter(bart_model[f'{encoder_base}.{i}.fc1.bias'])
            encoder_layer._feedforward[2].weight = nn.Parameter(bart_model[f'{encoder_base}.{i}.fc2.weight'])
            encoder_layer._feedforward[2].bias = nn.Parameter(bart_model[f'{encoder_base}.{i}.fc2.bias'])
            encoder_layer._final_layer_norm.weight = nn.Parameter(bart_model[f'{encoder_base}.{i}.final_layer_norm.weight'])
            encoder_layer._final_layer_norm.bias = nn.Parameter(bart_model[f'{encoder_base}.{i}.final_layer_norm.bias'])
