"""Defines a Dataset for parsing and iterating over CRD3 data."""
import json
from os import listdir, path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import yaml
from numpy import ceil
from spacy.lang.en import English
from spacy.util import compile_suffix_regex, compile_infix_regex, compile_prefix_regex
from torch.nn.functional import one_hot
from torch.utils.data import IterableDataset, get_worker_info
from torchtext.data.utils import get_tokenizer
from spacy.vocab import Vocab


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
        self._tokenizer = self.get_tokenizer()
        self._vocab: Vocab = Vocab().from_disk(self._cfg['vocab_path'])
        self._speaker_vocab: Vocab = Vocab().from_disk(self._cfg['spkr_vocab_path'])
        self._vocab_hash2idx: np.ndarray = np.load(self._cfg['vocab_hash2idx_path'])
        self._speaker_vocab_hash2idx: np.ndarray = np.load(self._cfg['spkr_vocab_hash2idx_path'])

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

        # TODO: Batch size?

    def lookup_token(
            self,
            val: str | int
    ) -> int | str:
        """Lookup either token or indices in the object's vocabulary.

        Parameters
        ----------
        val

        Returns
        -------

        """
        # TODO: Implement str -> hash -> idx and idx -> hash -> str
        if isinstance(val, str):
            try:
                # str -> hash -> idx
                ret = self._vocab_hash2idx[self._vocab_hash2idx == self._vocab.strings[val]]
            except (ValueError, IndexError):
                raise ValueError('Item not in vocabulary')
        elif isinstance(val, int):
            try:
                # idx -> hash -> str
                ret = self._vocab.strings[self._vocab_hash2idx[val]]
            except (ValueError, IndexError):
                raise ValueError('Item not in vocabulary')
        else:
            raise ValueError()

        if len(ret) == 1:  # TODO: Length issues - hash collisions?
            return ret[0]
        else:
            raise Exception('Unknown exception')  # TODO: Improve

    def lookup_speaker_token(
            self,
            val: str | int
    ) -> int | str:
        """Lookup either token or indices in the object's speaker vocabulary.

        Parameters
        ----------
        val

        Returns
        -------

        """
        # TODO: Implement str -> hash -> idx and idx -> hash -> str
        if isinstance(val, str):
            try:
                # str -> hash -> idx
                ret = self._speaker_vocab_hash2idx[self._speaker_vocab_hash2idx == self._speaker_vocab.strings[val]]
            except (ValueError, IndexError):
                raise ValueError('Item not in speaker vocabulary')
        elif isinstance(val, int):
            try:
                # idx -> hash -> str
                ret = self._speaker_vocab.strings[self._speaker_vocab_hash2idx[val]]
            except (ValueError, IndexError):
                raise ValueError('Item not in speaker vocabulary')
        else:
            raise ValueError()

        if len(ret) == 1:  # TODO: Length issues - hash collisions?
            return ret[0]
        else:
            raise Exception('Unknown exception')  # TODO: Improve

    def construct_string(self, token_idxs: torch.Tensor) -> str:
        """Builds a string from a list of token indices.

        Parameters
        ----------
        token_idxs

        Returns
        -------

        """
        # TODO: Simple string join is going to put in spaces where there should be none. Maybe use a spacy function?
        return ' '.join([self.lookup_token(idx) for idx in token_idxs])

    @staticmethod
    def get_tokenizer():
        """Returns an English tokenizer with some extra rules to deal with some peculiarities in the data."""
        nlp = English()
        suffixes = nlp.Defaults.suffixes + [r'''--$''', r'''\)$''', r''':$''', r'''\]$''', r'''\-$''']
        prefixes = nlp.Defaults.prefixes + [r'''^--''', r'''^\(''', r'''^\[''', r'''^\-''']
        infixes = nlp.Defaults.infixes + [
            r'''\(''', r'''\)''', r'''--''', r'''"''', r'''\[''', r'''\]''',
            r'''(?<=[dmsu123])x(?=[0-9]{1,3})''']  # Last one is for episode numbers, e.g. 1x24, sx65, e3x01...
        suffix_regex = compile_suffix_regex(suffixes)
        prefix_regex = compile_prefix_regex(prefixes)
        infix_regex = compile_infix_regex(infixes)
        nlp.tokenizer.suffix_search = suffix_regex.search
        nlp.tokenizer.prefix_search = prefix_regex.search
        nlp.tokenizer.infix_finditer = infix_regex.finditer
        return nlp.tokenizer

    def _prepare_data(
            self,
            speaker_strings: list[str],
            turn_strings: list[str],
            summary_string: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        data: torch.Tensor
            Stacked concatenation of the multi-hot speaker encoding with each one-hot token encoding of turn_strings of
            dimension (n_turn_tokens_over_all_turns, speaker_vocab_size + vocab_size).
        targets: torch.Tensor
            Stacked one-hot token encodings of summary_string of dimension (n_summary_tokens, vocab_size).
        """
        # Tokenize and one-hot summary strings
        target_idxs = [self.lookup_token(t) for t in self._tokenizer(summary_string)]
        targets: torch.Tensor = one_hot(target_idxs)
        # targets.requires_grad = True
        # TODO: Decide where to set requires_grad

        # Create multi-hot speaker encoding
        chunk_data: list[torch.Tensor] = []
        for turn_speaker_str, turn_str in zip(speaker_strings, turn_strings):
            speaker_idxs = np.array([self.lookup_token(t) for t in self._tokenizer(turn_speaker_str)])
            speaker_data: torch.Tensor = torch.zeros(len(self._speaker_vocab))
            speaker_data[speaker_idxs] = torch.ones(len(speaker_idxs))

            # Create one-hot encoding for each token in the turn
            turn_idxs = [self.lookup_token(t) for t in self._tokenizer(turn_str)]
            turn_data: torch.Tensor = one_hot(turn_idxs)
            # turn_data.requires_grad = True

            # Stack speaker so there is an identical speaker tensor for each utterance tensor
            speaker_data = speaker_data.repeat(len(turn_data))
            # speaker_data.requires_grad = True

            # Concatenate speaker and utterance data
            chunk_data.append(torch.concat((speaker_data, turn_data)))
            # TODO: Use sparse because this vocab is going to be huge?

        # Concatenate data for all turns in the chunk
        data = torch.concat(chunk_data)

        return data, targets

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
