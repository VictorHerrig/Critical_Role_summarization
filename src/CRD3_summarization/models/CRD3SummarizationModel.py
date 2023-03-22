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
            device: str = None
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
        self._encoder_linear = nn.Linear(model_dim + speaker_size, model_dim, device=device)
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

        src_input = self._encoder_linear(concat_src)
        out_seq = self._model(src_input,
                              tgt_embeddings,
                              tgt_mask=tgt_mask,
                              src_key_padding_mask=src_key_padding_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self._decoder_linear(out_seq)
        return self._decoder_smax(logits)

    def generate(
            self,
            src: Tensor,
            speakers: Tensor,
            src_key_padding_mask: Tensor = None,
            max_tgt_seq_len: int = 500,
            beam_size: int = 8,
            top_k: int = 32,
            top_p: float = 0.94
    ) -> Tensor:
        with torch.no_grad():
            # TODO: Add scaled LM score into the beam search.
            # TODO: Add sequence length regularizer into beam search.
            assert beam_size < self._vocab_size

            # Find encoded source representation
            src_embeddings = self._embedding_layer(src)
            concat_src = torch.concat((speakers, src_embeddings), dim=-1)
            src_input = self._encoder_linear(concat_src)
            src_encoding = self._model.model.encoder(src_input, src_key_padding_mask=src_key_padding_mask)

            # Initialize target as <bos> token
            tgt = torch.zeros((1, 1, self._vocab_size), dtype=torch.float32, device=self.device)  # (1, 1, vocab_size)
            tgt[..., self._bos_token_idx] = 1

            # Generate tokens one-by-one
            eos = False

            beams = torch.zeros((0, beam_size, self.vocab_size))  # (seq_len, beam_size, model_size)
            beam_probs = torch.ones(1)  # (beam_size, )
            is_retired = torch.zeros(beam_size, dtype=torch.bool)

            while not eos and tgt.size(0) < max_tgt_seq_len:
                # Get output sequences for all beams
                out_seq = self._model.model.decoder.forward(tgt, src_encoding)  # (seq_len, beam_size, model_size)

                # No need to calculate the whole sequence. We will find the marginal prob later
                last_token = out_seq[-1, ...]  # (beam_size, model_size)
                probs = self._decoder_smax(self._decoder_linear(last_token))  # (beam_size, vocab_size)

                # Use top-k and top-p methods to consolidate prob
                prob_sort_idx = torch.argsort(probs, dim=1, descending=True)  # (beam_size, vocab_size)
                # Zero all non-top-k choices
                not_top_k_idx = probs[prob_sort_idx][:, top_k:]  # (beam_size, top_k)
                probs[not_top_k_idx] = 0.
                # Zero all non-top-p choices
                top_p_idx = torch.cumsum(probs[prob_sort_idx], dim=1) < top_p  # (beam_size, vocab_size)
                probs[~top_p_idx] = 0.
                # Scale the remainder
                probs = probs / torch.sum(probs, dim=1)  # (beam_size, vocab_size)

                # Find full sequence probs
                if beam_probs.size() == 1:
                    # First run or a simple greedy search
                    marginal_probs = probs
                else:
                    marginal_probs = beam_probs[~is_retired].unsqueeze(-1) * probs

                # Take the top k sequences
                flat_topk_idx = marginal_probs.view(-1).argsort()[-beam_size:]
                topk_idx = unravel_index(flat_topk_idx, marginal_probs.shape)  # (beam_size, 2) tuple
                topk_beams_idx = torch.tensor([i[0] for i in topk_idx])  # (beam_size, )
                topk_vocab_idx = torch.tensor([i[1] for i in topk_idx])  # (beam_size, )
                topk_vals = probs[topk_beams_idx, topk_vocab_idx]  # (beam_size, )

                # The following can lead to more than beam_size beams added if there are retired beams - the lowest prob
                # beams including retired beams will be culled

                # One-hot new tok-k sequences to add to previous beams, and pad retired beams
                n_added = topk_vals.shape(0) + is_retired.sum()
                retired_idx: Tensor = torch.argwhere(is_retired)
                not_retired_idx = torch.concat((torch.argwhere(~is_retired), torch.arange(len(is_retired), n_added)))
                new_tokens = torch.zeros(n_added, self.vocab_size, dtype=torch.float32)  # (n_added, vocab_size)
                new_tokens[not_retired_idx, topk_vocab_idx] = 1.
                new_tokens[retired_idx, self._pad_token_idx] = 1.

                # Add new tokens to beam sequences and cull low prob beams
                new_base_beam_idx = torch.arange(n_added)
                new_base_beam_idx[not_retired_idx] = topk_beams_idx  # (n_added,)
                new_beams = torch.concat((beams[:, new_base_beam_idx, :], new_tokens))  # (seq_len + 1, n_added, vocab_size)

                # Collect beam probs and pick the top k
                new_beam_probs = torch.zeros(n_added)
                new_beam_probs[retired_idx] = beam_probs[is_retired]
                new_beam_probs[not_retired_idx] = topk_vals
                topk_new_beams = torch.argsort(new_beam_probs)[-beam_size:]

                # TODO: This can 'fail' if the model predicts <pad> ... but that in itself is a type of failure...
                # Assign the new topk beams and retire any that end in <eos> or <pad>
                beams = new_beams[topk_new_beams]  # (seq_len + 1, beam_size, vocab_size)
                beam_probs = new_beam_probs[topk_new_beams]  # (beam_size, )
                is_retired = beams[-1].argmax(-1) == self._pad_token_idx or beams[-1].argmax(-1) == self._eos_token_idx

                tgt = beams[~is_retired]

                # If all beams have finished, we are done
                eos = is_retired.all()

            max_prob_idx = beam_probs.argmax()
            output = beams[max_prob_idx]

            return output

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
