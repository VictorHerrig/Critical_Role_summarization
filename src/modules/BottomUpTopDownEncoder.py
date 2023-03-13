"""This file defines the top-down bottom-up encoder block as detailed in https://arxiv.org/pdf/2203.07586v1.pdf."""
from collections import OrderedDict

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
            device: str = None
    ):
        super().__init__()
        self._full_self_attn = nn.MultiheadAttention(model_dim, num_heads=num_attn_heads, dropout=dropout)
        self._layer_norm = nn.LayerNorm([model_dim, ])

    def forward(self, val: Tensor) -> Tensor:
        attn_out = self._full_self_attn(val, val, val)
        return self._layer_norm(val + attn_out)


class LocalSelfAttentionBlock(nn.Module):
    def __init__(
            self,
            model_dim: int,
            local_self_attn_window_size: int = 1024,
            num_attn_heads: int = 8,
            dropout: float = 0.1,
            device: str = None
    ):
        super().__init__()
        self._local_self_attn = LocalSelfAttention(model_dim, local_self_attn_window_size, num_heads=num_attn_heads, dropout=dropout)
        self._layer_norm = nn.LayerNorm([model_dim, ])

    def forward(self, val: Tensor) -> Tensor:
        attn_out = self._local_self_attn(val)
        return self._layer_norm(val + attn_out)


class TopDownEncoderSubBlock(nn.Module):
    def __init__(
            self,
            model_dim: int,
            feedforward_dim: int = 2048,
            local_self_attn_window_size: int = 1024,
            num_attn_heads: int = 8,
            dropout: float = 0.1,
            device: str = None
    ):
        super().__init__()
        self._local_self_attn = LocalSelfAttention(model_dim, local_self_attn_window_size, num_attn_heads, dropout)
        self._full_cross_attn = nn.MultiheadAttention(model_dim, num_heads=num_attn_heads, dropout=dropout)
        self._layer_norm = nn.LayerNorm([model_dim, ])
        self._feedforward = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(model_dim, feedforward_dim)),
            ('relu', nn.ReLU()),
            ('linear_2', nn.Linear(feedforward_dim, model_dim))
        ]))

    def forward(self, val: Tensor, top_level_tokens: Tensor) -> Tensor:
        token_self_attn = self._local_self_attn(val, val, val)
        token_segment_cross_attn = self._full_cross_attn.forward(token_self_attn, top_level_tokens, top_level_tokens)
        # Note! The paper _specifically_ notes the residual connection is applied after the layer norm
        # so the cross attention acts strictly as an update.
        # Mind you, the paper also doesn't specify any other residual or layer norm components which does rather feel
        # silly, so I've added them in the other blocks for the time being anyway. Perhaps I will add a toggle parameter
        # in future.
        cross_attn_layer_norm = token_self_attn + self._layer_norm(token_segment_cross_attn)
        output = self._feedforward(cross_attn_layer_norm)

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
            device: str = None
    ):
        super().__init__()

        self._local_self_attn_stack = nn.Sequential(
            OrderedDict([
                (
                    f'local_self_attn_block_{i}',
                    LocalSelfAttentionBlock(model_dim, local_self_attn_window_size, num_attn_heads, dropout)
                )
                for i in range(num_local_self_attn)
            ])
        )
        self._avg_pool = nn.AvgPool1d(kernel_size=avg_pool_kernel_size, stride=avg_pool_stride)
        self._segment_full_self_attn_stack = nn.Sequential(
            OrderedDict([
                (
                    f'full_self_attn_block_{i}',
                    FullSelfAttentionBlock(model_dim, num_attn_heads, dropout)
                )
                for i in range(num_segment_full_self_attn)
            ])
        )

    def forward(self, val: Tensor) -> tuple[Tensor, Tensor]:
        # Local self-attention on token-level 'bottom-level' representations
        bottom_lvl_repr = self._local_self_attn_stack(val)
        # Pooling to create initial 'top-level' representations
        top_level_initializations = self._avg_pool(bottom_lvl_repr)
        # Full self-attention on segment-level 'top-level' representations
        return bottom_lvl_repr, self._segment_full_self_attn_stack(top_level_initializations)


class TopDownEncoderBlock(nn.Module):
    def __init__(
            self,
            model_dim: int,
            feedforward_dim: int,
            local_self_attn_window_size: int = 1024,
            num_attn_heads: int = 8,
            dropout: float = 0.1,
            num_top_down_blocks: int = 4,
            device: str = None
    ):
        super().__init__()

        # Can't be sequential since each sublayer output needs to be the argument to all 3 parameters for the next layer
        self._segment_full_self_attn_stack = nn.ModuleList([
            TopDownEncoderSubBlock(model_dim, feedforward_dim, local_self_attn_window_size, num_attn_heads, dropout, device)
            for _ in range(num_top_down_blocks)
        ])

    def forward(self, val: Tensor, top_level_tokens: Tensor) -> Tensor:
        top_down_intermediate = val
        for block in self._segment_full_self_attn_stack:
            top_down_intermediate = block(top_down_intermediate, top_level_tokens)
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
            num_local_self_attn: int = 8,
            num_segment_full_self_attn: int = 2,
            num_top_down_blocks: int = 4,
            num_attn_heads: int = 8,
            dropout: float = 0.1,
            avg_pool_kernel_size: int = 32,
            avg_pool_stride: int = 24,
            device: str = None
    ):
        super().__init__()
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

    def forward(self, val: Tensor) -> Tensor:
        bottom_lvl_repr, top_lvl_repr = self._bottom_up_encoder(val)
        final_token_repr = self._top_down_encoder(bottom_lvl_repr, top_lvl_repr)
        return final_token_repr
