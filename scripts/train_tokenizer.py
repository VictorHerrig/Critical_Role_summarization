"""Script that generates tokenizers for the data."""
import sys
from argparse import ArgumentParser
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer

sys.path.append('../src/CRD3_summarization')
try:
    from loaders.CRD3Dataset import CRD3Dataset
except ImportError as e:
    print('Must be run from the scripts/ directory!')
    raise e


def split_gen(dataset: CRD3Dataset) -> Iterator[str]:
    # Read CRD3 data
    for _, turn_strs, summ_str in dataset.iter_chunk(False):
        yield summ_str
        yield from turn_strs
    # Read data from campaign 3 episode 1
    with open('../data/C3E001.txt', 'r') as f:
        for line in f:
            yield ':'.join(line.split(':')[1:])


def split_gen_spkr(dataset: CRD3Dataset) -> Iterator[str]:
    # Read CRD3 data
    for spkr, _, _ in dataset.iter_chunk(False):
        yield spkr
    # Read data from campaign 3 episode 1
    with open('../data/C3E001.txt', 'r') as f:
        for line in f:
            yield line.split(':')[0]


def main(
        cfg_file: dict,
        outfile: str,
        spkr_outfile: str,
        vocab_size: int = 6000,
        spkr_vocab_size: int = 100
) -> None:
    # Train tokenizer on all text data
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.BertNormalizer()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.BPEDecoder()

    dataset = CRD3Dataset(cfg_file)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    )
    tokenizer.train_from_iterator(split_gen(dataset), trainer, length=1400000)
    tokenizer.save(outfile)

    # Train tokenizer on all speaker data
    speaker_tokenizer = Tokenizer(WordLevel())
    speaker_tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.BertNormalizer()
    ])
    speaker_tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    speaker_tokenizer.decoder = decoders.CTC()

    speaker_trainer = WordLevelTrainer(
        vocab_size=spkr_vocab_size,
        show_progress=True,
        special_tokens=['<UNK>']
    )
    speaker_tokenizer.train_from_iterator(split_gen_spkr(dataset), speaker_trainer, length=1400000)
    speaker_tokenizer.save(spkr_outfile)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg-file', required=True, type=str, help='Path to the CRD3Dataset yaml configuration file.')
    parser.add_argument('--outfile', required=True, type=str, help='Path of the output file (directories will not be made).')
    parser.add_argument('--spkr-outfile', required=True, type=str, help='Path of the output speaker file (directories will not be made).')
    parser.add_argument('--vocab-size', default=6000, type=int, help='Target vocabulary size. (Default=6000)')
    parser.add_argument('--spkr-vocab-size', default=100, type=int, help='Target speaker vocabulary size. (Default=100)')
    cfg_dict = vars(parser.parse_args())
    main(**cfg_dict)
