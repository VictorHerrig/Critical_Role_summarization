# This script simply tests that you can load the model into GPU memory
import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer

from CRD3_summarization.models.HuggingfaceModels import QuantModelFactory
from CRD3_summarization.loaders import CRD3DecoderOnlyDataset, CRD3MistralLiteDataset

from unsloth import FastLanguageModel


def main():
    dataset = CRD3DecoderOnlyDataset('conf/CRD3Dataset_mistral_unsloth_test.yaml')
    # dataset = CRD3MistralLiteDataset('conf/check_dataset_output.yaml')

    # model, tokenizer = QuantModelFactory.mistrallite()
    model, tokenizer = FastLanguageModel.from_pretrained('unsloth/mistral-7b-bnb-4bit', load_in_4bit=True, device_map='auto')
    model.load_adapter('unsloth_mistral_train/checkpoint-256')
    # model, tokenizer = QuantModelFactory.mistrallite_flash()
    # model, tokenizer = QuantModelFactory.mistral_7b_flash()

    print('\n------------')
    print('Model params')
    print('------------\n\n')
    for param_name, param in model.named_parameters():
        print(f'{param_name}: {tuple(param.shape)}')
    print(model.device)
    print(f'\n\nTokenizer vocab size: {tokenizer.vocab_size}')

    prompt_str = """
        <s> [INST] The following is an excerpt from a show in which multiple actors are playing Dungeons and Dragons:
    
         MATT: 24. So in setting up preparation, you take each one of the heads, and to fill each one of their-- I want to say concave, but it's an interior pyramid space. Convex is outward. It will be about 100 platinum pieces or so to fill each face void. So you take the platinum, you melt it down into the smelter, get it to where it's in the gripped, tong-held metallic reservoir, until it eventually melts down. You then pour it into each of the heads, so 300 platinum utilized to do this. It fills to the very edge. Some of it spills over, but you can easily snip that off. As you wait for it to cool, which you can probably help with, actually, as there is no basin for quenching the metal at the moment, so the steam (whoosh) rises up all around, and you turn the heads over, and each one (sliding sound) leaves these heavy metallic pyramid pieces that are smooth and perfect on three of the sides, and one of them appears to be almost like a broken crystal on the inside. It's a shattered, messed-up section.
         LAURA: Puzzle piece! Got to fit them together!
         TALIESIN: Okay, so if we fit these together--
         MATT: They seem to fit together very well; you figured it out very quickly, they fit and hold together.
         LAURA: With the jaggedy ends, or with the smooth ends?
         MATT: The jaggedy ends all fit together.
         TALIESIN: Does this now make another concave piece?
         MATT: All together, it makes a pyramid that comes to a central point.
         
        That is the end of the script.
        Write a summary of the script, making sure to include all the major events contained within.
         [/INST]
        """

    print('\n-----------')
    print('   Prompt')
    print('-----------\n\n')
    print(prompt_str)
    print('\n\n')


    # prompt = dataset[3]['input_ids'].reshape(1, -1)
    # prompt_str = dataset.construct_string(prompt.squeeze())

    prompt = torch.tensor(dataset.tokenizer.encode(prompt_str, add_special_tokens=False)).to(torch.int).reshape(1, -1)

    print(prompt.size())

    generation_config = GenerationConfig(
        max_new_tokens=512,
        num_beams=3,
        eos_token_id=dataset.eos_token_id,
        bos_token_id=dataset.bos_token_id,
        pad_token_id=dataset.pad_token_id
    )

    prompt = prompt.to(model.device)
    output = model.generate(
        prompt,
        generation_config=generation_config
    )

    print(type(output))

    # For multiple output beam search
    # for i in range(output.sequences.size()[0]):
    #     print('\n-----------')
    #     print(f'Sequence {i}')
    #     print('-----------\n\n')
    #     print(f'Score for sequence {i}: {output.sequences_scores[i]}\n')
    #     print(f'Output:\n\n')
    #     print(dataset.construct_string(output.sequences[i, :].squeeze()))
    #     print('\n\n')

    # For single output
    print('\n----------')
    print(f'Sequence')
    print('----------\n\n')
    print(f'Output:\n\n')
    print(dataset.construct_string(output.squeeze()))
    print('\n\n')


if __name__ == '__main__':
    main()
