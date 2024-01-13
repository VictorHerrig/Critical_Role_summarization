"""Defines a Dataset for parsing and iterating over CRD3 data."""
import abc
import json
from os import listdir, path
from typing import Iterator, Optional

import numpy as np
import pandas as pd  # TODO: Replace with own function to remove requirement
import torch
import yaml
from numpy import ceil
from transformers import PreTrainedTokenizer, AutoTokenizer
from torch.nn.functional import one_hot
from torch.utils.data import IterableDataset, get_worker_info


# TODO: Experiment with adding summaries from previous chunks to prompt prefix


class CRD3Dataset(IterableDataset, abc.ABC):
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
        self._max_seq_len = self._cfg['max_src_seq_len']

        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self._cfg['tokenizer_path'])

        # TODO: Langchain or similar eventually
        # Prompt prefix and suffix strings
        self._prompt_prefix = self._cfg.get('prompt_prefix', '')
        self._prompt_suffix = self._cfg.get('prompt_suffix', '')

        # Pseudo lazy evaluation of prompt prefix and suffix tokens and tensors
        self._prompt_prefix_tokens = None
        self._prompt_suffix_tokens = None
        self._prompt_prefix_tensor = None
        self._prompt_suffix_tensor = None

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # Summary-turn pairs are grouped in JSON files
        # To avoid repeating samples in the same order, use a buffer to pull samples across multiple JSON files
        # Why did they remove the buffered dataset class? ... ¯\_(ツ)_/¯
        buffer = []
        for speaker_strings, turn_strings, summary_string in self.iter_chunk(shuffle=True):
            # Make sure there are no empty strings
            # TODO: WARN logging
            if len(summary_string) <= 0: continue
            speaker_strings, turn_strings = zip(*[(s, t) for s, t in zip(speaker_strings, turn_strings)
                                                  if len(s) > 0 and len(t) > 0])
            if len(speaker_strings) <= 0 or len(turn_strings) <= 0: continue

            buffer.append((speaker_strings, turn_strings, summary_string))
            if len(buffer) > self.buffer_size:
                yield self._prepare_data(*buffer.pop(np.random.randint(0, len(buffer), 1).item()))

        # Finish out the buffer after all samples have been pulled
        for i in np.random.permutation(np.arange(len(buffer))):
            speaker_samp, turn_samp, summary_samp = buffer[i]
            yield self._prepare_data(speaker_samp, turn_samp, summary_samp)
        buffer.clear()

    def __getitem__(self, idx):
        if idx >= len(self._files):
            raise IndexError('Index out of range of files')

        # Load file of this index and take the first chunk
        json_iter = CRD3Dataset.parse_json(self._files[idx])
        speaker_strings, turn_strings, summary_string = next(json_iter)

        # Retry until a non-empty summary appears
        try:
            while len(summary_string) <= 0:
                speaker_strings, turn_strings, summary_string = next(json_iter)
        except StopIteration as e:
            raise ValueError(f'No valid summary in file index {idx} {self._files[idx]}')

        # If the turns and/or speakers and all empty, raise error
        speaker_strings, turn_strings = zip(*[(s, t) for s, t in zip(speaker_strings, turn_strings)
                                              if len(s) > 0 and len(t) > 0])
        if len(speaker_strings) <= 0 or len(turn_strings) <= 0:
            raise ValueError(f'No valid turn strings in file index {idx} {self._files[idx]}')

        return self._prepare_data(speaker_strings, turn_strings, summary_string)

    def construct_string(self, token_idxs: torch.Tensor) -> str:
        """Builds a string from a list of token indices.

        Parameters
        ----------
        token_idxs

        Returns
        -------

        """
        return str.replace(self.tokenizer.decode(token_idxs.tolist(), skip_special_tokens=False), 'Ġ', ' ')
    
    @abc.abstractmethod
    def _build_inputs(
            self,
            source_turn_tokens: list[list[int]],
            summary_tokens: list[int]
    ) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError()

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
        source: torch.Tensor
            One-hot token encoding of turn_strings of dimension (src_seq_len, vocab_size).
        speaker: torch.Tensor
            Multi-hot speaker encoding of dimension (src_seq_len, speaker_vocab_size).
        target: torch.Tensor
            Stacked one-hot token encodings of summary_string of dimension (tgt_seq_len, vocab_size).
        """
        def tokenize_turn(turn_speaker_str, turn_str):
            return self.tokenizer.encode(turn_speaker_str + ':\n' + turn_str + '\n')

        # Tokenize script and summary
        src_turn_tokens = [tokenize_turn(*turn_pair) for turn_pair in zip(speaker_strings, turn_strings)]
        summary_tokens = self.tokenizer.encode(summary_string)

        # Build tensors according to subclass
        return self._build_inputs(src_turn_tokens, summary_tokens)

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
                speaker_strings, turn_strings = zip(*[(', '.join(t['NAMES']), ' '.join(t['UTTERANCES']))
                                                      for t in chunk['TURNS']])
                yield speaker_strings, turn_strings, summary_string

    def __len__(self):
        return len(self._files)

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def indir(self):
        return self._indir

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def pad_token(self) -> int:
        return self.tokenizer.encode(self.tokenizer.pad_token)[0]

    @property
    def unk_token(self) -> int:
        return self.tokenizer.encode(self.tokenizer.unk_token)[0]

    @property
    def bos_token(self) -> int:
        return self.tokenizer.encode(self.tokenizer.bos_token)[0]

    @property
    def bos_token_tensor(self):
        return one_hot(torch.tensor([self.bos_token]), num_classes=self.vocab_size).to(torch.bfloat16)

    @property
    def eos_token(self) -> int:
        return self.tokenizer.encode(self.tokenizer.eos_token)[0]

    @property
    def eos_token_tensor(self):
        return one_hot(torch.tensor([self.eos_token]), num_classes=self.vocab_size).to(torch.bfloat16)

    @property
    def prompt_prefix(self) -> str:
        return self._prompt_prefix
    
    @property
    def prompt_prefix_tokens(self) -> list[int]:
        if self._prompt_prefix_tokens is None:
            self._prompt_prefix_tokens = self.tokenizer.encode(self.prompt_prefix)
        return self._prompt_prefix_tokens

    @property
    def prompt_prefix_tensor(self) -> torch.Tensor:
        if self._prompt_prefix_tensor is None:
            self._prompt_prefix_tensor = one_hot(
                torch.tensor(self.prompt_prefix_tokens), num_classes=self.vocab_size
            ).to(torch.bfloat16)
        return self._prompt_prefix_tensor

    @property
    def prompt_suffix(self) -> str:
        return self._prompt_suffix
    
    @property
    def prompt_suffix_tokens(self) -> list[int]:
        if self._prompt_suffix_tokens is None:
            self._prompt_suffix_tokens = self.tokenizer.encode(self.prompt_suffix)
        return self._prompt_suffix_tokens

    @property
    def prompt_suffix_tensor(self) -> torch.Tensor:
        if self._prompt_suffix_tensor is None:
            self._prompt_suffix_tensor = one_hot(
                torch.tensor(self.prompt_suffix_tokens), num_classes=self.vocab_size
            ).to(torch.bfloat16)
        return self._prompt_suffix_tensor


class CRD3EncoderDecoderDataset(CRD3Dataset):
    def __init__(
            self,
            cfg_file: str
    ) -> None:
        super().__init__(cfg_file)
        self._max_tgt_seq_len = self._cfg['max_tgt_seq_len']


    def _build_inputs(
            self,
            source_turn_tokens: list[list[int]],
            summary_tokens: list[int]
    ) -> (torch.Tensor, torch.Tensor):
        """
        
        Parameters
        ----------
        source_turn_tokens
        summary_tokens

        Returns
        -------

        """
        # Check if the script fits within the max sequence length - 2 (for BOs and EOS)
        num_source_tokens = [len(s) for s in source_turn_tokens]
        total_source_tokens = sum(num_source_tokens)
        max_len = self.max_seq_len - 2 - len(self.prompt_prefix_tokens) - len(self.prompt_suffix_tokens)
        if total_source_tokens > max_len:
            # Truncate turns from the beginning of the script until it fits in the max sequence length if needed
            cum_seq_len = np.cumsum(reversed(num_source_tokens))
            first_idx = len(num_source_tokens) - np.searchsorted(max_len, cum_seq_len)
            source_turn_tokens = source_turn_tokens[first_idx:]

        # One-hot the combined turns with BOS and EOS
        source_turn_tokens = np.concatenate(source_turn_tokens)
        source = one_hot(torch.tensor(source_turn_tokens), num_classes=self.vocab_size).to(torch.bfloat16)

        # Add the prompt prefix and suffix
        source = torch.concat((
            self.bos_token_tensor,
            self.prompt_prefix_tensor,
            source,
            self.prompt_suffix_tensor,
            self.eos_token_tensor
        ))

        # One-hot the target with BOS and EOS
        summary_tokens = [self.bos_token] + summary_tokens + [self.eos_token]
        target = one_hot(torch.tensor(summary_tokens), num_classes=self.vocab_size).to(torch.bfloat16)

        return source, target

    @property
    def max_tgt_seq_len(self):
        return self._max_tgt_seq_len


class CRD3DecoderOnlyDataset(CRD3Dataset):
    def _build_inputs(
            self,
            source_turn_tokens: list[list[int]],
            summary_tokens: list[int]
    ) -> (torch.Tensor, torch.Tensor):
        """

        Parameters
        ----------
        source_turn_tokens
        summary_tokens

        Returns
        -------

        """
        # Check if the script fits within the max sequence length - 2 (for BOs and EOS)
        num_source_tokens = [len(s) for s in source_turn_tokens]
        total_source_tokens = sum(num_source_tokens)
        max_len = self.max_seq_len - 2 - len(self.prompt_prefix_tokens) - len(self.prompt_suffix_tokens) - len(summary_tokens)
        if total_source_tokens > max_len:
            # Truncate turns from the beginning of the script until it fits in the max sequence length if needed
            cum_seq_len = np.cumsum(reversed(num_source_tokens))
            first_idx = len(num_source_tokens) - np.searchsorted(max_len, cum_seq_len)
            source_turn_tokens = source_turn_tokens[first_idx:]  # TODO: If empty

        # One-hot the combined turns with BOS and EOS
        source_turn_tokens = np.concatenate(source_turn_tokens)
        source = one_hot(torch.tensor(source_turn_tokens), num_classes=self.vocab_size).to(torch.bfloat16)
        target = one_hot(torch.tensor(summary_tokens), num_classes=self.vocab_size).to(torch.bfloat16)

        # Add the prompt prefix and suffix and BOS token
        generation_tensor = torch.concat((
            self.bos_token_tensor,
            self.prompt_prefix_tensor,
            source,
            self.prompt_suffix_tensor,
            target,
            self.eos_token_tensor
        ))

        # Source and target are the same for decoder-only transformers
        return generation_tensor, generation_tensor


class CRD3BatchCollator:
    def __init__(self, pad_token_idx: str, window_size: int):
        self.pad_token_idx = pad_token_idx
        self.window_size = window_size

    def __call__(
            self,
            samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source, speaker, target = zip(*samples)
        source, src_key_padding_mask = self.add_padding(source)
        speaker = self.add_speaker_padding(speaker)
        target, tgt_key_padding_mask = self.add_padding(target)

        # Tokens/speakers: (seq_len, bsz, model_dim)
        # Padding: (bsz, seq_len, model_dim)
        return source, speaker, target, src_key_padding_mask, tgt_key_padding_mask

    def add_padding(self, inputs: list[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Seq len must be a multiple of window_size for sliding chunk MM in local attention
        max_len = max([t.size(0) for t in inputs])
        max_len = ((max_len // self.window_size) + int(max_len % self.window_size != 0)) * self.window_size
        #max_len += self.window_size - (max_len % self.window_size) if max_len % self.window_size != 0 else 0
        vocab_size = inputs[0].size(-1)

        if any([max_len - t.size(0) > 0 for t in inputs]):
            # Create padding masks
            padding_mask = torch.stack([(torch.arange(max_len) >= t.size(0)).to(torch.bool) for t in inputs], dim=0)
            # print(padding_mask.shape)
        else:
            padding_mask = None

        output = list(inputs)

        # Append padding one-hot vectors
        for i, t in enumerate(output):
            pad_amt = max_len - t.size(0)
            if pad_amt > 0:
                padding = torch.zeros((pad_amt, vocab_size), dtype=torch.bfloat16)
                padding[:, self.pad_token_idx] = 1.
                output[i] = torch.concat((t, padding), dim=0)

        # Create a batch axis at axis 1
        return torch.stack(output, dim=1), padding_mask

    def add_speaker_padding(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        # Shapes
        max_len = max([t.size(0) for t in inputs])
        max_len = ((max_len // self.window_size) + int(max_len % self.window_size != 0)) * self.window_size
        vocab_size = inputs[0].size(-1)

        output = list(inputs)

        # Fill with all zeros
        for i, t in enumerate(output):
            pad_amt = max_len - t.size(0)
            if pad_amt > 0:
                padding = torch.zeros((pad_amt, vocab_size), dtype=torch.bfloat16)
                output[i] = torch.concat((t, padding), dim=0)

        # Create a batch axis at axis 1
        return torch.stack(output, dim=1)
