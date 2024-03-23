# Simple script to manually check the dataset output. Run from the root directory.


from numpy.random import randint

from cr_summ.Datasets import EncoderDecoderCRD3Dataset, MistralCRD3Dataset,\
    EncoderDecoderDialogsumDataset, MistralDialogsumDataset


def main():
    # encoder_decoder_ds = EncoderDecoderCRD3Dataset('conf/dataset_test.yaml')
    # mistral_ds = MistralCRD3Dataset('conf/dataset_test.yaml')
    encoder_decoder_ds = EncoderDecoderDialogsumDataset('conf/dataset_val.yaml')
    mistral_ds = MistralDialogsumDataset('conf/dataset_val.yaml')

    idx = randint(0, len(mistral_ds) - 1)

    print('===================================')
    print('encoder-decoder string output:')
    print('===================================\n\n')
    d = encoder_decoder_ds[idx]
    source_str = d['text']
    summ_str = d['summary']
    source_ids = d['input_ids']
    summ_ids = d['labels']
    decoded_source_str = encoder_decoder_ds.tokenizer.decode(source_ids, skip_special_tokens=False)
    decoded_summ_str = encoder_decoder_ds.tokenizer.decode(summ_ids, skip_special_tokens=False)
    print('\n-------------------')
    print('      Source')
    print('-------------------\n')
    print(source_str)
    print()
    print('\n-------------------')
    print('      Summary')
    print('-------------------\n')
    print(summ_str)
    print()
    print('\n-------------------')
    print('  Decoded Source')
    print('-------------------\n')
    print(decoded_source_str)
    print()
    print('\n-------------------')
    print('  Decoded Summary  ')
    print('-------------------\n')
    print(decoded_summ_str)
    input()

    print('\n\n===================================')
    print('decoder-only string output:')
    print('===================================\n\n')
    d = mistral_ds[idx]
    sequence_str = d['text']
    prompt_str = d['prompt']
    summ_str = d['summary']
    sequence_ids = d['input_ids']
    decoded_sequence = mistral_ds.tokenizer.decode(sequence_ids, skip_special_tokens=False)
    print('\n-------------------')
    print('      Prompt')
    print('-------------------\n')
    print(prompt_str)
    print()
    print('\n-------------------')
    print('      Summary')
    print('-------------------\n')
    print(summ_str)
    print()
    print('\n-------------------')
    print('      Sequence')
    print('-------------------\n')
    print(sequence_str)
    print()
    print('\n-------------------')
    print(' Decoded Sequence')
    print('-------------------\n')
    print(decoded_sequence)


if __name__ == '__main__':
    main()
