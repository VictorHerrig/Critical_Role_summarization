"""Defines a Dataset for parsing and iterating over CRD3 data."""
import json
from os import listdir, path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from numpy import ceil
from torch.nn.functional import one_hot
from torch.utils.data import IterableDataset, get_worker_info

from tokenizers import Tokenizer


class CRD3Dataset(IterableDataset):
    def __init__(
            self,
            cfg_file: str
    ) -> None:
        """Dataset that loads summary, speaker and utterance information from CRD3 json files.

        Parameters
        ----------
        cfg_file: str
            Path to the configuration YAML file. See CRD3Dataset_train.yaml for configuration documentation.
        """
        super().__init__()
        # Load config
        with open(cfg_file, 'r') as f:
            self._cfg: dict = yaml.safe_load(f)
        self._indir = path.abspath(self._cfg['CRD3_path'])
        self._buffer_size = self._cfg['buffer_size']

        # Load filenames from the input directory present in the index file
        if 'idx_file' in self._cfg:
            idx_file = path.abspath(self._cfg['idx_file'])
            file_subset = pd.read_csv(idx_file).values.squeeze().tolist()
            self._files = [path.join(self.indir, fn) for fn in listdir(self.indir)
                           if 'json' in fn and fn.split('_')[0] in file_subset]
        else:
            self._files = [path.join(self.indir, fn) for fn in listdir(self.indir)
                           if 'json' in fn]
        self._files = np.array(self._files)

        # Prepare text processing objects
        self._max_src_seq_len = self._cfg['max_src_seq_len']
        self._max_tgt_seq_len = self._cfg['max_tgt_seq_len']

        self._tokenizer = Tokenizer.from_file(self._cfg['tokenizer_path']) if 'tokenizer_path' in self._cfg else None
        self._speaker_tokenizer = Tokenizer.from_file(self._cfg['spkr_tokenizer_path']) if 'spkr_tokenizer_path' in self._cfg else None

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # Summary-turn pairs are grouped in JSON files
        # To avoid repeating samples in the same order, use a buffer to pull samples across multiple JSON files
        # Why did they remove the buffered dataset class? ... ¯\_(ツ)_/¯
        buffer = []
        for speaker_strings, turn_strings, summary_string in self.iter_chunk(shuffle=True):
            buffer.append((speaker_strings, turn_strings, summary_string))
            if len(buffer) < self._buffer_size:
                yield self._prepare_data(*buffer.pop(np.random.randint(0, len(buffer), 1).item()))

        # Finish out the buffer after all samples have been pulled
        for speaker_samp, turn_samp, summary_samp in np.random.permutation(buffer):
            yield self._prepare_data(speaker_samp, turn_samp, summary_samp)

        # Clear out the buffer
        buffer.clear()
        # TODO: Above perhaps not needed

    def construct_string(self, token_idxs: torch.Tensor) -> str:
        """Builds a string from a list of token indices.

        Parameters
        ----------
        token_idxs

        Returns
        -------

        """
        if self.tokenizer is None:
            raise ValueError('No tokenizer passed!')
        return self.tokenizer.decode(token_idxs.tolist())

    def _prepare_data(
            self,
            speaker_strings: list[str],
            turn_strings: list[str],
            summary_string: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Takes sample strings and converts them to encoded vectors. The targets consist of a tensor of stacked one-hot
        encodings for each summary token. The data consist of a tensor of stacked vecors, each of which is a
        concatenation of a multi-hot speaker encoding with a one-hot token encoding.

        Parameters
        ----------
        speaker_strings: list[str]
            List of space-separated strings containing speaker name(s).
        turn_strings: list[str]
            List of strings containing turn string.
        summary_string: str
            String containing turn summary.

        Returns
        -------
        source: torch.Tensor
            One-hot token encoding of turn_strings of dimension (src_seq_len, vocab_size).
        speakers: torch.Tensor
            Multi-hot speaker encoding of dimension (src_seq_len, speaker_vocab_size + vocab_size).
        targets: torch.Tensor
            Stacked one-hot token encodings of summary_string of dimension (tgt_seq_len, vocab_size).
        src_key_padding_mask: torch.Tensor
            Padding mask tensor of same shape dimension 0 of source.
        tgt_key_padding_mask: torch.Tensor
            Padding mask tensor of same shape dimension 0 of target.
        """
        # Tokenize and one-hot summary strings
        target_idxs = self.tokenizer.encode(summary_string).ids[:self.max_tgt_seq_len - 2]  # -2 for <EOS> and <BOS>
        target: torch.Tensor = one_hot(target_idxs, num_classes=self.vocab_size).to(torch.float32)
        # targets.requires_grad = True
        # TODO: Decide where to set requires_grad

        # Create multi-hot speaker encoding
        src_chunk_data = []
        speaker_chunk_data = []
        chunk_len = 0
        for turn_speaker_str, turn_str in zip(speaker_strings, turn_strings):

            # Create one-hot encoding for each token in the turn
            turn_idxs = self.tokenizer.encode(turn_str).ids
            if len(turn_idxs) + chunk_len > self.max_src_seq_len:
                turn_idxs = turn_idxs[:self.max_src_seq_len - chunk_len]
            chunk_len += len(turn_idxs)
            turn_data: torch.Tensor = one_hot(turn_idxs, num_classes=self.vocab_size).to(torch.float32)
            # turn_data.requires_grad = True
            src_chunk_data.append(turn_data)

            # Stack speaker so there is an identical speaker tensor for each utterance tensor
            speaker_idxs = self.tokenizer.encode(turn_speaker_str).ids
            speaker_data: torch.Tensor = torch.zeros(self.speaker_vocab_size, dtype=torch.float32)
            speaker_data[speaker_idxs] = 1.
            speaker_data = speaker_data.repeat(turn_data.size(0))
            # speaker_data.requires_grad = True
            speaker_chunk_data.append(speaker_data)

            # Only read up to the max chunk length
            if chunk_len == self.max_src_seq_len:
                break

        # Concat all chunk data to make source and speaker tensors
        source = torch.stack(src_chunk_data, dim=0)
        speaker = torch.stack(speaker_chunk_data, dim=0)
        #
        # # Calculate padding masks
        # src_key_padding_mask = torch.arange(self.max_src_seq_len) > source.size(0)  # (max_src_seq_len, vocab_size)
        # tgt_key_padding_mask = torch.arange(self.max_tgt_seq_len) > target.size(0)  # (max_tgt_seq_len, vocab_size)
        #
        # # Add padding to the source and target. Add <bos> and <eos> to target. Add zeros to the speaker.
        # src_pad_amt = self.max_src_seq_len - source.size(0)
        # tgt_pad_amt = self.max_tgt_seq_len - target.size(0) - 2  # -s for <EOS> and <BOS>
        # src_pad_data = torch.zeros((src_pad_amt, self.vocab_size), dtype=torch.float32)  # (src_pad_amt, vocab_size)
        # src_pad_data[:, self.pad_token] = 1.
        # speaker_pad_data = torch.zeros((src_pad_amt, self.speaker_vocab_size), dtype=torch.float32)  # (src_pad_amt, speaker_vocab_size)
        # tgt_bos_data = torch.zeros((1, self.vocab_size), dtype=torch.float32)  # (1, vocab_size)
        # tgt_bos_data[:, self.bos_token] = 1.
        # tgt_eos_data = torch.zeros((1, self.vocab_size), dtype=torch.float32)  # (1, vocab_size)
        # tgt_eos_data[:, self.eos_token] = 1.
        # tgt_pad_data = torch.zeros((tgt_pad_amt, self.vocab_size), dtype=torch.float32)  # (tgt_pad_amt, vocab_size)
        # tgt_pad_data[:, self.pad_token] = 1.
        #
        # source = torch.concat((source, src_pad_data), dim=0)  # (max_src_seq_len, vocab_size)
        # speaker = torch.concat((speaker, speaker_pad_data), dim=0)  # (max_src_seq_len, speaker_vocab_size)
        # target = torch.concat((tgt_bos_data, target, tgt_eos_data, tgt_pad_data), dim=0)  # (max_tgt_seq_len, vocab_size)

        # return source, speaker, target, src_key_padding_mask, tgt_key_padding_mask
        return source, speaker, target

    def iter_chunk(
            self,
            shuffle: bool = False
    ) -> Iterator[tuple[list[str], list[str], str]]:
        """Iterates aver json files and returns parsed chunks.

        Parameters
        ----------
        shuffle: bool, optional
            Whether to shuffle files. (default = False)

        Yields
        ------
        speaker_strings: list[str]
            List of strings representing speakers for each turn in the chunk.
        utt_strings: list[str]
            List of strings representing utterances for each turn in the chunk.
        summary_string: str
            String representing the summary of the chunk.
        """
        for filename in self._iter_files(shuffle):
            yield from CRD3Dataset.parse_json(filename)

    def iter_chunk_w_filename(
            self,
            shuffle: bool = False
    ) -> Iterator[tuple[list[str], list[str], str]]:
        """Iterates aver json files and returns parsed chunks with the name of the origin file.

        Parameters
        ----------
        shuffle: bool, optional
            Whether to shuffle files. (default = False)

        Yields
        ------
        filename: str
            Origin filename.
        speaker_strings: list[str]
            List of strings representing speakers for each turn in the chunk.
        turn_strings: list[str]
            List of strings representing utterances for each turn in the chunk.
        summary_string: str
            String representing the summary of the chunk.
        """
        for filename in self._iter_files(shuffle):
            for speaker_strings, turn_strings, summary_string in CRD3Dataset.parse_json(filename):
                yield filename, speaker_strings, turn_strings, summary_string

    def _iter_files(
            self,
            shuffle: bool = False
    ) -> Iterator[str]:
        """Yields JSON files from the input directory. If run by a DataLoader, will check for worker parallelization.

        Parameters
        ----------
        shuffle: bool, optional
            Whether to shuffle files. (default = False)

        Yields
        ------
        filename: str
            Filepath.
        """
        worker_info = get_worker_info()
        # Single-process data loading, return the full iterator
        if worker_info is None:
            start_idx = 0
            end_idx = len(self)
        # Multiprocess data loading, return an iterator or a segment of the dataset
        else:
            n_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(ceil(len(self) / float(n_workers)))
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self))

        idxs = np.random.permutation(np.arange(start_idx, end_idx)) if shuffle else np.arange(start_idx, end_idx)
        yield from self._files[idxs]

    @staticmethod
    def parse_json(
            filepath
    ) -> Iterator[tuple[list[str], list[str], str]]:
        """Parses a CRD3 JSON file and yields values for every chunk.

        Parameters
        ----------
        filepath: str
            Path to the file to parse.

        Yields
        ------
        speaker_strings: list[str]
            List of strings representing speakers for each turn in the chunk.
        turn_strings: list[str]
            List of strings representing utterances for each turn in the chunk.
        summary_string: str
            String representing the summary of the chunk.
        """
        with open(filepath, 'r') as f:
            json_data = json.load(f)
            for chunk in json_data:
                # Load summary, speaker and utterance strings
                summary_string: str = chunk['CHUNK']
                speaker_strings, turn_strings = zip(*[(' '.join(c['NAMES']), ' '.join(c['UTTERANCES']))
                                                     for c in chunk['TURNS']])
                yield speaker_strings, turn_strings, summary_string

    def __len__(self):
        return len(self._files)

    @property
    def indir(self):
        return self._indir

    @property
    def max_src_seq_len(self):
        return self._max_src_seq_len

    @property
    def max_tgt_seq_len(self):
        return self._max_tgt_seq_len

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def speaker_vocab_size(self):
        return self._speaker_tokenizer.get_vocab_size()

    @property
    def pad_token(self):
        return self.tokenizer.token_to_id('<PAD>')

    @property
    def unk_token(self):
        return self.tokenizer.token_to_id('<UNK>')

    @property
    def bos_token(self):
        return self.tokenizer.token_to_id('<BOS>')

    @property
    def eos_token(self):
        return self.tokenizer.token_to_id('<EOS>')


class CRD3BatchCollator:
    def __init__(self, pad_token_idx: str):
        self.pad_token_idx = pad_token_idx

    def __call__(
            self,
            samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source, speaker, target = zip(*samples)
        source, src_key_padding_mask = self.add_padding(source)
        speaker = CRD3BatchCollator.add_speaker_padding(speaker)
        target, tgt_key_padding_mask = self.add_padding(target)

        return source, speaker, target, src_key_padding_mask, tgt_key_padding_mask

    def add_padding(self, inputs: list[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Shapes
        max_len = max([t.size(0) for t in inputs])
        vocab_size = inputs[0].size(-1)

        if any([max_len - t.size(0) > 0 for t in inputs]):
            # Create padding masks
            padding_mask = torch.stack([(torch.arange(max_len) > t.size(0)).to(torch.float32) for t in inputs], dim=1)
        else:
            padding_mask = None

        # Append padding one-hot vectors
        for i, t in enumerate(inputs):

            pad_amt = max_len - t.size(0)
            if pad_amt > 0:
                padding = torch.zeros((pad_amt, vocab_size), dtype=torch.float32)
                padding[:, self.pad_token_idx] = 1.
                inputs[i] = torch.concat((t, padding), dim=0)

        # Create a batch axis at axis 1
        return torch.stack(inputs, dim=1), padding_mask

    @staticmethod
    def add_speaker_padding(inputs: list[torch.Tensor]) -> torch.Tensor:
        # Shapes
        max_len = max([t.size(0) for t in inputs])
        vocab_size = inputs[0].size(-1)

        # Fill with all zeros
        for i, t in enumerate(inputs):
            pad_amt = max_len - t.size(0)
            if pad_amt > 0:
                padding = torch.zeros((pad_amt, vocab_size), dtype=torch.float32)
                inputs[i] = torch.concat((t, padding), dim=0)

        # Create a batch axis at axis 1
        return torch.stack(inputs, dim=1)
