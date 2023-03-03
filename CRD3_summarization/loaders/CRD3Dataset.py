import json
import os
from itertools import chain

import numpy as np
import pandas as pd
import torch
import yaml
from os import listdir, path
from typing import Iterator

from numpy import ceil
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import IterableDataset, get_worker_info

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab


class CRD3Dataset(IterableDataset):
    def __init__(
            self,
            cfg_file: str
    ) -> None:
        super().__init__()

        with open(cfg_file, 'r') as f:
            self._cfg: dict = yaml.safe_load(f)

        self._indir = self._cfg['CRD3_path']
        self._buffer_size = self._cfg['buffer_size']
        idx_file = self._cfg['idx_file']
        file_subset = pd.read_csv(idx_file).values.squeeze().tolist()
        self._files = [path.join(self.indir, fn) for fn in listdir(self.indir) if 'json' in fn and fn.split('_')[0] in file_subset]
        self.__len = self._cfg.get('n_summary', self._calc_len())

        self._tokenizer = get_tokenizer('basic_english')
        self._vocab: Vocab = build_vocab_from_iterator(self._calc_len())
        self._speaker_vocab: Vocab = ...  # TODO:
        # TODO: Use a standard vocab generated from all inputs

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for speaker_strings, utt_strings, summary_string in self._iter_utt(True):
            # TODO: Use buffer and take random utterances
            targets = self._str2tensor(summary_string)
            speaker_data = self._str2tensor(speaker_strings, self._speaker_vocab)
            utt_data = self._str2tensor(utt_strings)
            data = torch.concat((speaker_data, utt_data))
            return data, targets

    def _str2tensor(self, val: str | list[str], vocab_to_use: Vocab = self._vocab):
        if isinstance(str, list):
            return list(chain(*[self._str2tensor(v) for v in val]))
        idxs = vocab_to_use.lookup_indices(self._tokenizer(val))
        # TODO: Sparse one-hot bc this vocab will be huge
        return one_hot(idxs)

    def _iter_utt(self, shuffle: bool = False):
        for filename in self._iter_files(shuffle):
            yield from CRD3Dataset.parse_json(filename)

    def _iter_files(self, shuffle: bool = False) -> Iterator[str]:
        worker_info = get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            start_idx = 0
            end_idx = len(self)
        else:
            n_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(ceil(len(self) / float(n_workers)))
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self))

        idxs = np.random.permutation(np.arange(start_idx, end_idx)) if shuffle else np.arange(start_idx, end_idx)
        yield from self._files[idxs]

    @staticmethod
    def parse_json(filepath) -> Iterator[tuple[str, str, str]]:
        with open(filepath, 'r') as f:
            json_data = json.load(f)
            for chunk in json_data:
                # Load summary, speaker and utterance strings
                summary_string: str = chunk['CHUNK']
                speaker_strings, utt_strings = zip(*[(c['NAMES'], ' '.join(c['UTTERANCES'])) for c in chunk['TURNS']])
                yield speaker_strings, utt_strings, summary_string

    def __len__(self):
        if self.__len is not None:
            return self.__len
        else:
            return self._calc_len()

    def _calc_len(self) ->  Iterator[tuple[str, str, str]]:
        for speaker_strings, utt_strings, summary_string in self._iter_utt():
            self.__len += 1
            yield self._tokenizer(speaker_strings) + self._tokenizer(utt_strings) + self._tokenizer(summary_string)

    @property
    def indir(self):
        return self._indir
