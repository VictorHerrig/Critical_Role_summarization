from argparse import ArgumentParser

import yaml
from transformers import Trainer, TrainingArguments

from cr_summ.SAASR import SAASRModel
from cr_summ.Datasets import SpeakerRecognitionDataset


def main(passed_args: dict):
    model = SAASRModel('small.en', speaker_model_kwargs=dict(num_speakers=2, input_size=768, speaker_names=['MATT', 'SAM'], nhead=2, dim_feedforward=64))
    model.requires_grad_(True)
    model.train()

    train_dataset = SpeakerRecognitionDataset(cfg_file=passed_args["train_dataset_cfg_path"])
    eval_dataset = SpeakerRecognitionDataset(cfg_file=passed_args["val_dataset_cfg_path"])
    test_dataset = SpeakerRecognitionDataset(cfg_file=passed_args["test_dataset_cfg_path"])

    train_args = dict(
        output_dir='speaker_recognition',
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
        save_steps=128,
        logging_strategy='steps',
        eval_steps=64
    )
    train_args = dict(train_args, **passed_args['train_args'])
    train_args = TrainingArguments(**train_args)

    trainer = Trainer(
        model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.hf_tokenizer
    )
    trainer.train(resume_from_checkpoint=passed_args.get('resume_from_checkpoint', None))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--conf', required=True, type=str, help='Path to the config file, e.g. conf/train.yaml')
    with open(parser.parse_args().conf, 'r') as f:
        args = yaml.safe_load(f)
    main(args)
