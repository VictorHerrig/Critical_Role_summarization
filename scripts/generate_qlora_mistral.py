# This script simply tests that you can load the model into GPU memory
import torch
from transformers import GenerationConfig

from cr_summ.HuggingfaceModels import QuantModelFactory
from cr_summ.SummarizationDatasets import MistralCRD3Dataset, MistralDialogsumDataset


def main():
    test_dataset = MistralCRD3Dataset('conf/dataset_test.yaml')
    # test_dataset = MistralDialogsumDataset('conf/dataset_test.yaml')

    model, tokenizer = QuantModelFactory.mistral_7b_unsloth_4bit()
    model.load_adapter('dialogsum_unsloth_mistral_lora/checkpoint-192', device_map='auto', adapter_name='dialogsum')
    model.load_adapter('crd3_unsloth_mistral_lora/checkpoint-128', device_map='auto', adapter_name='crd3')
    model.set_adapter(['dialogsum'])
    # model.set_adapter(['crd3'])
    # model.set_adapter(['dialogsum', 'crd3'])

    idx = 2677
    # idx = 25
    data_dict = test_dataset[idx]
    prompt_str = data_dict['prompt']
    summary_str = data_dict['summary']

    # print('\n------------')
    # print('Model params')
    # print('------------\n\n')
    for param_name, param in model.named_parameters():
        param.requires_grad = False
    #     print(f'{param_name}: {tuple(param.shape)}, requires grad? {param.requires_grad}')
    # print(f'\n\nTokenizer vocab size: {tokenizer.vocab_size}')

    print('\n-----------')
    print('   Prompt')
    print('-----------\n\n')
    print(prompt_str)
    print('\n\n')

    print('Actual Summary')

    print('\n----------------')
    print(' Actual Summary')
    print('----------------\n\n')
    print(summary_str)
    print('\n\n')

    prompt = data_dict['generation_ids'].reshape(1, -1)
    max_new_tokens = min(512, prompt.size()[-1] // 2)

    print(f'Prompt size: {prompt.size()}')
    print(f'Max output size: {max_new_tokens}')

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    prompt = prompt.to(model.device)
    output = model.generate(
        prompt,
        generation_config=generation_config
    )

    print(f'Output size: {output.size()}')

    # For single output
    output_str = tokenizer.decode(output.squeeze(), skip_special_tokens=False).split('[/INST]')[-1]
    print('\n----------')
    print(f'Sequence')
    print('----------\n\n')
    print(f'Output:\n\n')
    print(output_str)
    print('\n\n')


if __name__ == '__main__':
    main()
