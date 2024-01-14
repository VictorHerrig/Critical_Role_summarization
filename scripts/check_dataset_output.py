# Simple script to manually check the dataset output. Run from the root directory.


import torch

from CRD3_summarization.loaders import CRD3EncoderDecoderDataset, CRD3DecoderOnlyDataset


def main():
    encoder_decoder_ds = CRD3EncoderDecoderDataset('conf/check_dataset_output.yaml')
    decoder_only_ds = CRD3DecoderOnlyDataset('conf/check_dataset_output.yaml')

    print('===================================')
    print('encoder-decoder string output:')
    print('===================================\n\n')
    turn_list, summ_str = encoder_decoder_ds.get_strings(0)
    turn_str = ''.join(turn_list)
    print('\n-----------')
    print('   Turn')
    print('-----------\n')
    print(turn_str)
    print()
    print('\n-----------')
    print('  Summary')
    print('-----------\n')
    print(summ_str)
    print()
    print(encoder_decoder_ds.prompt_prefix)
    print(encoder_decoder_ds.prompt_suffix)
    input()

    print('\n\n===================================')
    print('encoder-decoder decoded output:')
    print('===================================\n\n')
    turn_onehot, summ_onehot = encoder_decoder_ds[0]
    turn_tokens = torch.argmax(turn_onehot, 1)
    summ_tokens = torch.argmax(summ_onehot, 1)
    turn_str = encoder_decoder_ds.construct_string(turn_tokens)
    summ_str = encoder_decoder_ds.construct_string(summ_tokens)
    print('\n-----------')
    print('   Turn')
    print('-----------\n')
    print(turn_str)
    print()
    print('\n-----------')
    print('  Summary')
    print('-----------\n')
    print(summ_str)
    input()

    print('\n\n===================================')
    print('decoder-only string output:')
    print('===================================\n\n')
    turn_list, summ_str = decoder_only_ds.get_strings(0)
    turn_str = ''.join(turn_list)
    print('\n-----------')
    print('   Turn')
    print('-----------\n')
    print(turn_str)
    print()
    print('\n-----------')
    print('  Summary')
    print('-----------\n')
    print(summ_str)
    input()

    print('\n\n===================================')
    print('decoder-only decoded output:')
    print('===================================\n\n')
    turn_onehot, summ_onehot = decoder_only_ds[0]
    turn_tokens = torch.argmax(turn_onehot, 1)
    summ_tokens = torch.argmax(summ_onehot, 1)
    turn_str = encoder_decoder_ds.construct_string(turn_tokens)
    summ_str = encoder_decoder_ds.construct_string(summ_tokens)
    print('\n-----------')
    print('   Turn')
    print('-----------\n')
    print(turn_str)
    print()
    print('\n-----------')
    print('  Summary')
    print('-----------\n')
    print(summ_str)
    input()


if __name__ == '__main__':
    main()
