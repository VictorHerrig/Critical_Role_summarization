"""Model that wraps a BottomUpTopDownTransformer for use with CRD3 data."""
import torch
from numpy import unravel_index
from torch import nn, Tensor
from torch.nn import functional as F

from ..modules.BottomUpTopDownTransformer import BottomUpTopDownTransformer


class CRD3SummarizationModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            speaker_size: int,
            model_dim: int,
            pad_token_idx: int,
            bos_token_idx: int,
            eos_token_idx: int,
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
            max_len: int = 4000,
            max_tgt_seq_len: int = 150,
            device: str = None,
            initialize_from_bart: str = None
    ):
        super().__init__()

        assert model_dim > speaker_size

        self._vocab_size = vocab_size
        self._speaker_size = speaker_size
        self._max_tgt_seq_len = max_tgt_seq_len

        self._pad_token_idx = pad_token_idx
        self._bos_token_idx = bos_token_idx
        self._eos_token_idx = eos_token_idx

        self._embedding_layer = nn.Linear(vocab_size, model_dim, device=device)
        # The model will get the concatenation of word embeddings and speaker vectors as source representations
        # Speakers will be concatenated to word embeddings and reduced to the model dim via linear layer
        self._speaker_linear = nn.Linear(model_dim + speaker_size, model_dim, device=device)
        self._model = BottomUpTopDownTransformer(
            model_dim,
            num_decoder_layers,
            local_self_attn_window_size,
            feedforward_dim,
            num_local_self_attn,
            num_segment_full_self_attn,
            num_top_down_blocks,
            num_attn_heads,
            dropout,
            avg_pool_kernel_size,
            avg_pool_stride,
            max_len,
            device,
            initialize_from_bart
        )
        self.num_decoder_layers = num_decoder_layers
        self.num_local_self_attn = num_local_self_attn
        self.num_segment_full_self_attn = num_segment_full_self_attn
        self.num_top_down_blocks = num_top_down_blocks
        self._decoder_linear = nn.Linear(model_dim, vocab_size, device=device)
        self._decoder_smax = nn.Softmax(-1)
        self._device = device

    def forward(
            self,
            src: Tensor,
            speakers: Tensor,
            tgt: Tensor,
            src_key_padding_mask: Tensor = None,
            tgt_key_padding_mask: Tensor = None
    ) -> Tensor:
        tgt_mask = self._model.generate_square_subsequent_mask(tgt.shape[0]).to(self.device)

        src_embeddings = self._embedding_layer(src)
        tgt_embeddings = self._embedding_layer(tgt)
        concat_src = torch.concat((speakers, src_embeddings), dim=-1)

        src_input = self._speaker_linear(concat_src)
        out_seq = self._model(src_input,
                              tgt_embeddings,
                              tgt_mask=tgt_mask,
                              src_key_padding_mask=src_key_padding_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self._decoder_linear(out_seq)
        return self._decoder_smax(logits)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def speaker_size(self) -> int:
        return self._speaker_size

    @property
    def max_tgt_seq_len(self) -> int:
        return self._max_tgt_seq_len

    @property
    def device(self) -> str:
        return self._device
