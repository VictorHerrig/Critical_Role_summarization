"""Defines a Dataset for parsing and iterating over CRD3 data."""
import abc
import re
from typing import Iterator

import numpy as np
import torch
import yaml
from datasets import load_dataset, Dataset as HFDataset
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer


# TODO: Experiment with adding summaries from previous chunks to prompt prefix
# Possibly using something like <s> [SYS] <summary of last n chunks> [/SYS][INST] <prompt> [/INST] <response> </s>


""" ============================= """
"""            Base Class         """
""" ============================= """


class BaseSummarizationDataset(Dataset, abc.ABC):
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
        self._dataset = self._load_dataset()

        required_keys = ['tokenizer_path', 'max_seq_len']
        for k in required_keys:
            assert k in self._cfg, f'Config must contain key {k}'

        # Prepare text processing objects
        self._max_seq_len = self._cfg['max_seq_len']
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self._cfg['tokenizer_path'])

        # Prompt prefix and suffix strings
        self._prompt_prefix = self._cfg.get('prompt_prefix', '')
        self._prompt_suffix = self._cfg.get('prompt_suffix', '')

        # Pseudo lazy evaluation of prompt prefix and suffix tokens
        self._prompt_prefix_ids = None
        self._prompt_suffix_ids = None
        self._inst_open_ids = None
        self._inst_close_ids = None

    # -------------------- #
    #   Concrete Methods   #
    # -------------------- #

    def __iter__(self) -> Iterator[dict]:
        for turn_strings, summary_string in self._iter_strings():
            try:
                yield self._build_dict(turn_strings, summary_string)
            except ValueError:
                continue

    # TODO: Streaming
    def __getitem__(self, idx: int) -> dict:
        return self._build_dict(*self._get_strings(idx))

    def __len__(self):
        return len(self._dataset)

    def _iter_strings(self) -> Iterator[tuple[list[str], str]]:
        # Iterate through shuffles dataset
        for datum in self._dataset.shuffle():
            try:
                turn_strings, summary_string = self._parse_data(datum)
                yield turn_strings, summary_string
            except ValueError:
                continue

    def _get_strings(self, idx: int) -> tuple[list[str], str]:
        if idx >= len(self._dataset):
            raise IndexError('Index out of range of files')

        datum = self._dataset[idx]

        return self._parse_data(datum)

    # -------------------- #
    #   Abstract Methods   #
    # -------------------- #

    @abc.abstractmethod
    def _load_dataset(self) -> HFDataset:
        raise NotImplementedError()

    @abc.abstractmethod
    def _parse_data(self, dataset_dict: dict) -> tuple[list[str], str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _build_dict(self, turn_strings: list[str], summary_str: str) -> dict:
        raise NotImplementedError()

    # -------------------- #
    #      Properties      #
    # -------------------- #

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def unk_token_id(self) -> int:
        return self.tokenizer.unk_token_id

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_string(self) -> str:
        return self.tokenizer.pad_token

    @property
    def unk_token_string(self) -> str:
        return self.tokenizer.unk_token

    @property
    def bos_token_string(self) -> str:
        return self.tokenizer.bos_token

    @property
    def eos_token_string(self) -> str:
        return self.tokenizer.eos_token

    @property
    def bos_token_tensor(self):
        return one_hot(torch.tensor([self.bos_token_id]), num_classes=self.vocab_size).to(torch.int)

    @property
    def eos_token_tensor(self):
        return one_hot(torch.tensor([self.eos_token_id]), num_classes=self.vocab_size).to(torch.int)

    @property
    def prompt_prefix(self) -> str:
        return self._prompt_prefix

    @property
    def prompt_prefix_ids(self) -> list[int]:
        if self._prompt_prefix_ids is None:
            self._prompt_prefix_ids = self.tokenizer.encode(self.prompt_prefix, add_special_tokens=False)
        return self._prompt_prefix_ids

    @property
    def prompt_suffix(self) -> str:
        return self._prompt_suffix

    @property
    def prompt_suffix_ids(self) -> list[int]:
        if self._prompt_suffix_ids is None:
            self._prompt_suffix_ids = self.tokenizer.encode(self.prompt_suffix, add_special_tokens=False)
        return self._prompt_suffix_ids

    @property
    @abc.abstractmethod
    def inst_open(self):
        raise NotImplementedError()

    @property
    def inst_open_ids(self):
        if self._inst_open_ids is None:
            self._inst_open_ids = self.tokenizer.encode(self.inst_open, add_special_tokens=False)
        return self._inst_open_ids

    @property
    def inst_open_len(self):
        return len(self.inst_open_ids)

    @property
    @abc.abstractmethod
    def inst_close(self):
        raise NotImplementedError()

    @property
    def inst_close_ids(self):
        if self._inst_close_ids is None:
            self._inst_close_ids = self.tokenizer.encode(self.inst_close, add_special_tokens=False)
        return self._inst_close_ids

    @property
    def inst_close_len(self):
        return len(self.inst_close_ids)


""" ============================= """
""" Partially Implemented Classes """
""" ============================= """


class EncoderDecoderDataset(BaseSummarizationDataset, abc.ABC):
    def __init__(
            self,
            cfg_file: str
    ) -> None:
        super().__init__(cfg_file)
        assert 'max_tgt_seq_len' in self._cfg, 'Config file must contain key max_tgt_seq_len'
        self._max_tgt_seq_len = self._cfg['max_tgt_seq_len']

    def _build_dict(
            self,
            turn_strings: list[str],
            summary_string: str
    ) -> dict:
        turn_ids = [self.tokenizer.encode(turn_string, add_special_tokens=False) for turn_string in turn_strings]
        summ_ids = self.tokenizer.encode(summary_string, add_special_tokens=False)

        # Check if the script fits within the max sequence length - 2 (for BOs and EOS)
        turn_lens = [len(s) for s in turn_ids]
        total_source_len = sum(turn_lens)
        max_len = self.max_seq_len - 2 - len(self.prompt_prefix_ids) - len(self.prompt_suffix_ids)
        first_idx = 0
        if total_source_len > max_len:
            # Truncate turns from the beginning of the script until it fits in the max sequence length if needed
            cum_seq_len = np.cumsum(turn_lens[::-1])
            first_idx = len(turn_lens) - np.searchsorted(cum_seq_len, max_len)
            turn_ids = turn_ids[first_idx:]

        if len(turn_ids) <= 0:
            raise ValueError('No valid source string')

        # Build source tensor
        turn_ids = np.concatenate(turn_ids)
        source = torch.concat((
            torch.tensor([self.bos_token_id]),
            torch.tensor(self.prompt_prefix_ids),
            torch.tensor(turn_ids),
            torch.tensor(self.prompt_suffix_ids),
            torch.tensor([self.eos_token_id])
        )).to(torch.int)

        # Build target tensor
        summ_ids = [self.bos_token_id] + summ_ids[-(self.max_tgt_seq_len - 2):] + [self.eos_token_id]
        target = torch.tensor(summ_ids)

        source_string = self.prompt_prefix + ''.join(turn_strings[first_idx:]) + self.prompt_suffix

        return dict(
            text=source_string,
            summary=summary_string,
            input_ids=source,
            labels=target,
        )

    @property
    def max_tgt_seq_len(self):
        return self._max_tgt_seq_len

    @property
    def inst_open(self):
        return ''

    @property
    def inst_close(self):
        return ''


class DecoderOnlyDataset(BaseSummarizationDataset, abc.ABC):
    def _build_dict(
            self,
            turn_strings: list[str],
            summary_string: str
    ) -> dict:
        turn_ids = [self.tokenizer.encode(turn_string, add_special_tokens=False) for turn_string in turn_strings]
        summ_ids = self.tokenizer.encode(summary_string, add_special_tokens=False)

        # Check if the script fits within the max sequence length - 2 (for BOs and EOS)
        turn_lens = [len(s) for s in turn_ids]
        total_source_len = sum(turn_lens)
        max_len = self.max_seq_len - 2 - len(self.prompt_prefix_ids) - len(self.prompt_suffix_ids)
        first_idx = 0
        if total_source_len > max_len:
            # Truncate turns from the beginning of the script until it fits in the max sequence length if needed
            cum_seq_len = np.cumsum(turn_lens[::-1])
            first_idx = len(turn_lens) - np.searchsorted(cum_seq_len, max_len)
            turn_ids = turn_ids[first_idx:]

        if len(turn_ids) <= 0:
            raise ValueError('No valid source string')

        # Build sequence tensor
        turn_ids = np.concatenate(turn_ids)
        sequence = torch.concat((
            torch.tensor([self.bos_token_id]),
            torch.tensor(self.inst_open_ids),
            torch.tensor(self.prompt_prefix_ids),
            torch.tensor(turn_ids),
            torch.tensor(self.prompt_suffix_ids),
            torch.tensor(self.inst_close_ids),
            torch.tensor(summ_ids),
            torch.tensor([self.eos_token_id])
        )).to(torch.int)
        labels = sequence.clone()

        prompt_string = self.bos_token_string + \
                        self.inst_open + \
                        self.prompt_prefix + \
                        ''.join(turn_strings[first_idx:]) + \
                        self.prompt_suffix + \
                        self.inst_close

        sequence_string = prompt_string + summary_string + self.bos_token_string

        return dict(
            text=sequence_string,
            prompt=prompt_string,
            summary=summary_string,
            input_ids=sequence,
            labels=labels,
        )


class CRD3Dataset(BaseSummarizationDataset, abc.ABC):
    def __init__(
            self,
            cfg_file: str
    ) -> None:
        """Dataset that loads summary, speaker and utterance information from the CRD3 Huggingface dataset.

        Parameters
        ----------
        cfg_file: str
            Path to the configuration YAML file. See CRD3Dataset_train.yaml for configuration documentation.
        """
        super().__init__(cfg_file=cfg_file)

    def _load_dataset(self) -> HFDataset:
        return load_dataset('crd3', split=self._cfg.get('split', 'train')).filter(lambda x: len(x['chunk']) > 1)

    def _parse_data(
            self,
            dataset_dict: dict
    ) -> tuple[list[str], str]:
        summary_string: str = dataset_dict['chunk']
        turns: list[dict] = dataset_dict['turns']

        # Pass if there are empty strings and construct turn strings
        if len(summary_string) <= 0:
            raise ValueError('Empty summary string')
        turn_strings = [f'{" and ".join(turn["names"])}: {" ".join(turn["utterances"])}\n' for turn in turns
                        if len(turn["names"]) > 0 and len(turn["utterances"]) > 0]
        if len(turn_strings) <= 0:
            raise ValueError('Empty turn string')

        return turn_strings, summary_string


class DialogsumDataset(BaseSummarizationDataset, abc.ABC):
    def __init__(
            self,
            cfg_file: str
    ) -> None:
        """Dataset that loads summary, speaker and utterance information from the dialogsum Huggingface dataset.

        Parameters
        ----------
        cfg_file: str
            Path to the configuration YAML file. See DialogsumDataset_train.yaml for configuration documentation.
        """
        super().__init__(cfg_file=cfg_file)

    def _load_dataset(self) -> HFDataset:
        return load_dataset('knkarthick/dialogsum', split=self._cfg.get('split', 'train')).filter(lambda x: len(x['chunk']) > 1)

    def _parse_data(
            self,
            dataset_dict: dict
    ) -> tuple[list[str], str]:
        summary_string: str = dataset_dict['summary']
        if len(summary_string) <= 0:
            raise ValueError('Empty summary string')
        dialogue: str = dataset_dict['dialogue']
        if len(dialogue) <= 0:
            raise ValueError('Empty dialogue string')
        turns = re.split(r'\W*(#Person_\d+#)\W*:\W*', dialogue)[1:]

        # Pass if there are empty strings or an odd number of turn string, else construct turn strings
        if len(turns) % 2 != 0:
            raise ValueError('Invalid number of turns')
        turn_strings = [f'{turns[i]}: {turns[i + 1]}\n' for i in range(0, len(turns), 2)]
        if len(turn_strings) <= 0:
            raise ValueError('Empty turn string')

        return turn_strings, summary_string


class MistralDataset(DecoderOnlyDataset, abc.ABC):
    @property
    def inst_open(self):
        return '[INST]'

    @property
    def inst_close(self):
        return '[/INST]'


class MistralliteDataset(DecoderOnlyDataset, abc.ABC):
    @property
    def inst_open(self):
        return '<|prompter|>'

    @property
    def inst_close(self):
        return '<|assistant|>'


""" ============================= """
"""        Concrete Classes       """
""" ============================= """


class MistralCRD3Dataset(MistralDataset, CRD3Dataset):
    ...


class MistralliteCRD3Dataset(MistralliteDataset, CRD3Dataset):
    ...


class EncoderDecoderCRD3Dataset(EncoderDecoderDataset, CRD3Dataset):
    ...


class MistralDialogsumDataset(MistralDataset, DialogsumDataset):
    ...


class MistralliteDialogsumDataset(MistralliteDataset, DialogsumDataset):
    ...


class EncoderDecoderDialogsumDataset(EncoderDecoderDataset, DialogsumDataset):
    ...
