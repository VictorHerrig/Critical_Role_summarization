# This script simply tests that you can load the model into GPU memory


import torch
from transformers import GenerationConfig


from CRD3_summarization.models.HuggingfaceModels import QuantModelFactory
from CRD3_summarization.loaders import CRD3DecoderOnlyDataset


def main():
    model, tokenizer = QuantModelFactory.mistral_7b()
    # model, tokenizer = QuantModelFactory.mistral_7b_flash()

    print('\n------------')
    print('Model params')
    print('------------\n\n')
    for param_name, param in model.named_parameters():
        print(f'{param_name}: {tuple(param.shape)}')
    print(f'\n\nTokenizer vocab size: {tokenizer.vocab_size}')

    decoder_only_ds = CRD3DecoderOnlyDataset('conf/check_dataset_output.yaml')
    prompt, _ = decoder_only_ds[0]
    prompt_tokens = torch.argmax(prompt, 1)
    prompt_str = decoder_only_ds.construct_string(prompt_tokens)

    print('\n-----------')
    print('   Prompt')
    print('-----------\n\n')
    print(prompt_str)
    print('\n\n')

    print(prompt.size())

    generation_config = GenerationConfig(
        max_new_tokens=512,
        num_beams=1,  #3,
        early_stopping=True,
        eos_token_id=decoder_only_ds.eos_token_id,
        bos_token_id=decoder_only_ds.bos_token_id,
        pad_token_id=decoder_only_ds.pad_token_id
    )

    output = model.generate(
        prompt,
        generation_config=generation_config
    )

    for i in range(output.sequences.size()[0]):
        print('\n-----------')
        print(f'Sequence {i}')
        print('-----------\n\n')
        print(f'Score for sequence {i}: {output.sequences_scores[i]}\n')
        print(f'Output:\n\n')
        print(decoder_only_ds.construct_string(output.sequences[i, :].squeeze()))
        print('\n\n')


if __name__ == '__main__':
    main()