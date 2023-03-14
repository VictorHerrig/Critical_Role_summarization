# TODO: Write
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

        # TODO: pad token idx
        self._pad_token_idx = 0
        # TODO: unk token idx for tgt key mask?

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
        self._decoder_smax = nn.Softmax(vocab_size)
        self._device = device

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
        logits = self._decoder_linear(out_seq)
        return self._decoder_smax(logits)

    def generate(
            self,
            src: Tensor,
            speakers: Tensor,
            src_key_padding_mask: Tensor = None,  # TODO: Why would this be necessary at inference time?
            max_tgt_seq_len: int = 500,
            beam_size: int = 5
    ) -> Tensor:
        with torch.no_grad():
            assert beam_size < self._vocab_size

            # Find encoded source representation
            src_embeddings = self._embedding_layer(src)
            concat_src = torch.concat((speakers, src_embeddings), dim=-1)
            src_input = self._encoder_linear(concat_src)
            src_encoding = self._model.model.encoder(src_input, src_key_padding_mask=src_key_padding_mask)

            # Initialize target as <pad> token
            tgt = torch.zeros((1, 1, self._vocab_size), dtype=torch.float32, device=self.device)  # (1, 1, vocab_size)
            tgt[..., self._pad_token_idx] = 1

            # TODO: Proper beam search
            # Generate tokens one-by-one
            eos = False

            beams = torch.zeros((0, beam_size, self.vocab_size))  # (seq_len, beam_size, model_size)
            beam_probs = torch.ones(1)  # (beam_size)
            retired_beams = torch.zeros(beam_size, dtype=torch.bool)

            current_beam_size = beam_size  # TODO: Maybe not needed

            while not eos and tgt.shape[0] < max_tgt_seq_len:
                # Get output sequences for all beams
                out_seq = self._model.model.decoder.forward(tgt, src_encoding)  # (seq_len, beam_size, model_size)
                # TODO: Maybe this ... check later

                # No need to calculate the whole sequence. We will find the marginal prob later
                last_token = out_seq[-1, ...]  # (beam_size, model_size)
                probs = self._decoder_smax(self._decoder_linear(last_token))  # (beam_size, vocab_size)
                # seq_probs = self._decoder_smax(self._decoder_linear(out_seq))  # (seq_len, beam_size, vocab_size)
                # probs = torch.prod(seq_probs, dim=0)  # (beam_size, vocab_size)

                # Find full sequence probs
                if beam_probs.size() == 1:
                    # First run or a simple greedy search
                    marginal_probs = probs
                else:
                    marginal_probs = beam_probs[~retired_beams] * probs
                flat_topk_idx = marginal_probs.view(-1).argsort()[-beam_size:]  # TODO: If beams cannot be brough out of retirement, this will be current_beam_size

                # Take the top k sequences
                topk_idx = unravel_index(flat_topk_idx, marginal_probs.shape)  # (beam_size, 2) tuple
                topk_beams_idx = torch.tensor([i[0] for i in topk_idx])  # (beam_size, )
                topk_vocab_idx = torch.tensor([i[0] for i in topk_idx])  # (beam_size, )
                topk_vals = probs[topk_beams_idx, topk_vocab_idx]  # (beam_size, )

                # TODO: Check for and store completed sequences (i.e. sequences with <pad> prediction)
                #  Will have to compete with new beams, i.e. if beams is 5, and one is finished, pass the other 4 beams
                #  to the next loop and see if the held back one is beaten.

                # TODO: Check for new beams that are better than retired beams
                # TODO: It occurs to me that the above should be impossible as current beam prob can only decrease
                # TODO: Mind you, a previously higher=prob beam could spawn several higher prob beams and therefore
                #  replace a retired beam

                # One-hot new tok-k sequences to add to previous beams, and pad retired beams
                new_tokens = torch.zeros(beam_size, self.vocab_size, dtype=torch.float32)
                new_tokens[~retired_beams, topk_vocab_idx] = 1.
                new_tokens[retired_beams, self._pad_token_idx] = 1.

                # Add new tokens to beam sequences
                beams[~retired_beams] = beams[~retired_beams][topk_beams_idx]  # If beams were superseded
                beams = torch.concat((beams, new_tokens), dim=0)
                beam_probs[~retired_beams] = topk_vals

                new_retired_beams = new_tokens.argmax(-1) == self._pad_token_idx
                retired_beams = beams[-1].argmax(-1) == self._pad_token_idx

                # TODO: Check
                tgt = torch.concat((tgt[0:1, ~new_retired_beams], beams[~retired_beams]), dim=0)  # Make sure to retain the leading <pad>

                # If all beams have finished, we are done
                eos = retired_beams.all()

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
