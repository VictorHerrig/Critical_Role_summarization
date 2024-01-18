# This script simply tests that you can load the model into GPU memory
import torch
from transformers import GenerationConfig

from unsloth import FastLanguageModel

from CRD3_summarization.CRD3Datasets import MistralCRD3Dataset
from CRD3_summarization.HuggingfaceModels import QuantModelFactory


def main():
    test_dataset = MistralCRD3Dataset('conf/CRD3Dataset_test.yaml')

    model, tokenizer = QuantModelFactory.mistral_7b_unsloth_4bit()
    model.load_adapter('unsloth_mistral_train/checkpoint-192')

    idx = 2677
    data_dict = test_dataset[idx]
    prompt_str = data_dict['prompt']
    summary_str = data_dict['summary']

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

    prompt = torch.tensor(tokenizer.encode(prompt_str, add_special_tokens=False)).to(torch.int).reshape(1, -1)

    print(f'Prompt size: {prompt.size()}')

    generation_config = GenerationConfig(
        max_new_tokens=min(512, prompt.size()[1] // 3),
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
    print('\n----------')
    print(f'Sequence')
    print('----------\n\n')
    print(f'Output:\n\n')
    print(tokenizer.decode(output.squeeze(), skip_special_tokens=False).split('[/INST]')[-1])
    print('\n\n')


if __name__ == '__main__':
    main()
