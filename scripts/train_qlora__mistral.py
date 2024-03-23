from argparse import ArgumentParser

import yaml
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel

from cr_summ.HuggingfaceModels import QuantModelFactory
from cr_summ import Datasets


def main(passed_args: dict):
    # Instantiate datasets
    dataset_type = args['dataset_type']
    train_dataset = getattr(Datasets, dataset_type)('conf/dataset_train.yaml')
    val_dataset = getattr(Datasets, dataset_type)('conf/dataset_val.yaml')

    def val_generator():
        """Generator that yields a random subset of the val dataset with cardinality 64."""
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

    # Update default LoRA args with values in the config file and load LoRA
    lora_args = dict(r=4, lora_alpha=4, bias='none')
    lora_args = dict(lora_args, **passed_args['lora_args'])
    model = FastLanguageModel.get_peft_model(model, **lora_args)

    # Update default train args with values in the config file
    train_args = dict(
        output_dir='crd3_unsloth_mistral_lora',
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
        logging_strategy='steps',
        eval_steps=64
    )
    train_args = dict(train_args, **passed_args['train_args'])
    train_args = TrainingArguments(**train_args)

    # Train
    trainer = Trainer(
        model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_subset,
        tokenizer=tokenizer
    )
    trainer.train(resume_from_checkpoint=passed_args.get('resume_from_checkpoint', None))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--conf', required=True, type=str, help='Path to the config file, e.g. conf/train.yaml')
    with open(parser.parse_args().conf, 'r') as f:
        args = yaml.safe_load(f)
    main(args)
