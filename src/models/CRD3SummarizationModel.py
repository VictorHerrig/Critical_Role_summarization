# TODO: Write
import torch
from torch import nn, Tensor

from ..modules.BottomUpTopDownTransformer import BottomUpTopDownTransformer


class CRD3SummarizationModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            speaker_size: int,
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
            max_len: int = 10000,
            max_tgt_seq_len: int = 500,  # TODO: Tune
            device: str = None
    ):
        super().__init__()

        assert model_dim > speaker_size  # TODO: Message

        self._vocab_size = vocab_size
        self._speaker_size = speaker_size
        self._max_tgt_seq_len = max_tgt_seq_len

        self._embedding_layer = nn.Embedding(vocab_size, model_dim)  # TODO: Sparse?
        # The model will get the concatenation of word embeddings and speaker vectors as source representations
        # Speakers will be concatenated to word embeddings and reduced to the model dim via linear layer
        self._encoder_linear = nn.Linear(model_dim + speaker_size, model_dim)
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
            device
        )
        self._decoder_linear = nn.Linear(model_dim, vocab_size)

    def forward(
            self,
            src: Tensor,
            speakers: Tensor,
            tgt: Tensor,
            src_key_padding_mask: Tensor = None,
            tgt_key_padding_mask: Tensor = None
    ) -> Tensor:
        tgt_mask = self._model.generate_square_subsequent_mask(tgt.shape[0])

        src_embeddings = self._embedding_layer(src)
        tgt_embeddings = self._embedding_layer(tgt)
        concat_src = torch.concat((speakers, src_embeddings), dim=-1)

        src_input = self._encoder_linear(concat_src)
        out_seq = self._model(src_input,
                              tgt_embeddings,
                              tgt_mask=tgt_mask,
                              src_key_padding_mask=src_key_padding_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)
        return self._decoder_linear(out_seq)

    def generate(
            self,
            src: Tensor,
            tgt_token: Tensor,
            src_key_padding_mask: Tensor = None,
            max_tgt_seq_len: int = 500
    ) -> Tensor:
        tgt = tgt_token.clone()
        eos = False
        src_encoding = self._model.model.encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Generate tokens one-by-one
        while not eos and tgt.shape[0] < max_tgt_seq_len:
            tgt_token = self._decoder_linear(self._model.model.decoder.forward(tgt_token, src_encoding))
            tgt = torch.concat((tgt, tgt_token.clone()), dim=0)

        return tgt

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def speaker_size(self):
        return self._speaker_size

    @property
    def max_tgt_seq_len(self):
        return self._max_tgt_seq_len
