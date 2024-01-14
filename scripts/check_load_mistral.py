# This script simply tests that you can load the model into GPU memory


from transformers import GenerationConfig


from CRD3_summarization.models.HuggingfaceModels import QuantModelFactory
from CRD3_summarization.loaders import CRD3DecoderOnlyDataset, CRD3MistralLiteDataset


def main():
    # dataset = CRD3DecoderOnlyDataset('conf/check_dataset_output.yaml')
    dataset = CRD3MistralLiteDataset('conf/check_dataset_output.yaml')

    model, tokenizer = QuantModelFactory.mistrallite()
    # model, tokenizer = QuantModelFactory.mistrallite_flash()
    # model, tokenizer = QuantModelFactory.mistral_7b_flash()

    print('\n------------')
    print('Model params')
    print('------------\n\n')
    for param_name, param in model.named_parameters():
        print(f'{param_name}: {tuple(param.shape)}')
    print(f'\n\nTokenizer vocab size: {tokenizer.vocab_size}')

    prompt, _ = dataset[33 * 4 - 1]
    prompt_str = dataset.construct_string(prompt.squeeze())

    print('\n-----------')
    print('   Prompt')
    print('-----------\n\n')
    print(prompt_str)
    print('\n\n')

    print(prompt.size())

    generation_config = GenerationConfig(
        max_new_tokens=512,
        num_beams=1,
        eos_token_id=dataset.eos_token_id,
        bos_token_id=dataset.bos_token_id,
        pad_token_id=dataset.pad_token_id
    )

    prompt = prompt.to(model.device)
    output = model.generate(
        prompt,
        generation_config=generation_config
    )

    # For beam search
    # for i in range(output.sequences.size()[0]):
    #     print('\n-----------')
    #     print(f'Sequence {i}')
    #     print('-----------\n\n')
    #     print(f'Score for sequence {i}: {output.sequences_scores[i]}\n')
    #     print(f'Output:\n\n')
    #     print(decoder_only_ds.construct_string(output.sequences[i, :].squeeze()))
    #     print('\n\n')

    # For greedy search
    print('\n----------')
    print(f'Sequence')
    print('----------\n\n')
    print(f'Output:\n\n')
    print(dataset.construct_string(output.squeeze()))
    print('\n\n')


if __name__ == '__main__':
    main()
