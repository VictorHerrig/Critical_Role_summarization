from CRD3_summarization.CRD3Datasets import MistralCRD3Dataset
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel

from CRD3_summarization.HuggingfaceModels import QuantModelFactory

from datetime import datetime
import yaml
from argparse import ArgumentParser


def main(args: dict):
    train_dataset = MistralCRD3Dataset('conf/CRD3Dataset_train.yaml')
    val_dataset = MistralCRD3Dataset('conf/CRD3Dataset_val.yaml')

    def val_generator():
        """Function that generates a random subset of cardinality 64"""
        i = 0
        subset_card = 64
        for val in val_dataset:
            yield val
            i += 1
            if i >= subset_card:
                break

    val_subset = Dataset.from_generator(val_generator)

    # Load model and adapter
    model, tokenizer = QuantModelFactory.mistral_7b_unsloth_4bit()
    # model, tokenizer = FastLanguageModel.from_pretrained('unsloth/mistral-7b-bnb-4bit', load_in_4bit=True, device_map='auto')

    # Update default LoRA args with values in the config file and load LoRA
    lora_args = dict(r=4, lora_alpha=4, bias='none')
    lora_args = dict(lora_args, **args['lora_args'])
    model = FastLanguageModel.get_peft_model(model, **lora_args)

    # Update default train args with values in the config file
    train_args = dict(
        output_dir='unsloth_mistral_train',
        do_train=True,
        do_eval=True,
        max_steps=8192,
        evaluation_strategy='steps',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        optim='adamw_8bit',
        logging_steps=16,
        fp16=True,
        save_steps=128,
        logging_dir=f'mistral_qlora_train_{datetime.today()}',
        logging_strategy='steps',
        eval_steps=64
    )
    train_args = dict(train_args, **args['train_args'])
    train_args = TrainingArguments(**train_args)

    # Train
    trainer = Trainer(model, args=train_args, train_dataset=train_dataset, eval_dataset=val_subset, tokenizer=tokenizer)
    trainer.train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--conf', required=True, type=str, help='Path to the config file, e.g. conf/train.yaml')
    with open(parser.parse_args().conf, 'r') as f:
        args = yaml.safe_load(f)
    main(args)
