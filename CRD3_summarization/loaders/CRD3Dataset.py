"""Defines a Dataset for parsing and iterating over CRD3 data."""
import json
from os import listdir, path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import yaml
from numpy import ceil
from torch.nn.functional import one_hot
from torch.utils.data import IterableDataset, get_worker_info
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


class CRD3Dataset(IterableDataset):
    def __init__(
            self,
            cfg_file: str
    ) -> None:
        """Dataset that loads summary, speaker and utterance information from CRD3 json files.

        Parameters
        ----------
        cfg_file: str
            Path to the configuration YAML file. See CRD3.yaml for configuration documentation.
        """
        super().__init__()

        # Load config
        with open(cfg_file, 'r') as f:
            self._cfg: dict = yaml.safe_load(f)
        self._indir = self._cfg['CRD3_path']
        self._buffer_size = self._cfg['buffer_size']
        idx_file = self._cfg['idx_file']

        # Load filenames from the input directory present in the index file
        file_subset = pd.read_csv(idx_file).values.squeeze().tolist()
        self._files = [path.join(self.indir, fn) for fn in listdir(self.indir)
                       if 'json' in fn and fn.split('_')[0] in file_subset]

        # Prepare text processing objects
        self._tokenizer = get_tokenizer('basic_english')
        self._vocab: Vocab = ...  # TODO: Define this
        self._speaker_vocab: Vocab = ...  # TODO: Define this
        # TODO: Use a standard vocab generated from all inputs

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # Summary-turn pairs are grouped in JSON files
        # To avoid repeating samples in the same order, use a buffer to pull samples across multiple JSON files
        # Why did they remove the buffered dataset class? ... ¯\_(ツ)_/¯
        buffer = []
        for speaker_strings, utt_strings, summary_string in self._iter_chunk(shuffle=True):
            buffer.append((speaker_strings, utt_strings, summary_string))
            if len(buffer) < self._buffer_size:
                yield self._prepare_data(*buffer.pop(np.random.randint(0, len(buffer), 1).item()))

        # Finish out the buffer after all samples have been pulled
        for speaker_samp, utt_samp, summary_samp in np.random.permutation(buffer):
            yield self._prepare_data(speaker_samp, utt_samp, summary_samp)

        # Clear out the buffer
        buffer.clear()
        # TODO: Above perhaps not needed

        # TODO: Batch size?

    def _prepare_data(
            self,
            speaker_strings: list[str],
            utt_strings: list[str],
            summary_string: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Takes sample strings and converts them to encoded vectors. The targets consist of a tensor of stacked one-hot
        encodings for each summary token. The data consist of a tensor of stacked vecors, each of which is a
        concatenation of a multi-hot speaker encoding with a one-hot token encoding.

        Parameters
        ----------
        speaker_strings: list[str]
            List of space-separated strings containing speaker name(s).
        utt_strings: list[str]
            List of strings containing turn utterances.
        summary_string: str
            String containing turn summary.

        Returns
        -------
        data: torch.Tensor
            Stacked concatenation of the multi-hot speaker encoding with each one-hot token encoding of utt_strings of
            dimension (n_utt_tokens_over_all_turns, speaker_vocab_size + vocab_size).
        targets: torch.Tensor
            Stacked one-hot token encodings of summary_string of dimension (n_summary_tokens, vocab_size).
        """
        # Tokenize and one-hot summary strings
        target_idxs = self._vocab.lookup_indices(self._tokenizer(summary_string))
        targets: torch.Tensor = one_hot(target_idxs)
        # targets.requires_grad = True
        # TODO: Decide where to set requires_grad

        # Create multi-hot speaker encoding
        turn_data: list[torch.Tensor] = []
        for turn_speaker_str, turn_utt_str in zip(speaker_strings, utt_strings):
            speaker_idxs = np.array(self._speaker_vocab.lookup_indices(self._tokenizer(turn_speaker_str)))
            speaker_data: torch.Tensor = torch.zeros(len(self._speaker_vocab))
            speaker_data[speaker_idxs] = torch.ones(len(speaker_idxs))

            # Create one-hot encoding for each token in the utterance
            utt_idxs = self._vocab.lookup_indices(self._tokenizer(turn_utt_str))
            utt_data: torch.Tensor = one_hot(utt_idxs)
            # utt_data.requires_grad = True

            # Stack speaker so there is an identical speaker tensor for each utterance tensor
            speaker_data = speaker_data.repeat(len(utt_data))
            # speaker_data.requires_grad = True

            # Concatenate speaker and utterance data
            turn_data.append(torch.concat((speaker_data, utt_data)))

        # Concatenate data for all turns in the chunk
        data = torch.concat(turn_data)

        return data, targets

    def _iter_chunk(
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
        utt_strings: list[str]
            List of strings representing utterances for each turn in the chunk.
        summary_string: str
            String representing the summary of the chunk.
        """
        with open(filepath, 'r') as f:
            json_data = json.load(f)
            for chunk in json_data:
                # Load summary, speaker and utterance strings
                summary_string: str = chunk['CHUNK']
                speaker_strings, utt_strings = zip(*[(' '.join(c['NAMES']), ' '.join(c['UTTERANCES']))
                                                     for c in chunk['TURNS']])
                yield speaker_strings, utt_strings, summary_string

    def __len__(self):
        return len(self._files)

    @property
    def indir(self):
        return self._indir
