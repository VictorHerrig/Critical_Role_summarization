# This script simply tests that you can load the model into GPU memory
import accelerate
import torch
from accelerate import Accelerator
from accelerate.test_utils.scripts.external_deps.test_peak_memory_usage import TorchTracemalloc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, LoftQConfig, get_peft_model

from torch.utils.data import DataLoader


from CRD3_summarization.models.HuggingfaceModels import QuantModelFactory
from CRD3_summarization.datasets import CRD3MistralLiteDataset, CRD3DecoderOnlyDataset

from transformers import Trainer, TrainingArguments

from unsloth import FastLanguageModel

from datasets import Dataset


def main():

    # print('Record PID then press enter...')
    # input()

    #accelerator = Accelerator()
    # train_dataset = CRD3MistralLiteDataset('conf/CRD3Dataset_mistrallite_train.yaml')
    # val_dataset = CRD3MistralLiteDataset('conf/CRD3Dataset_mistrallite_val.yaml')
    # test_dataset = CRD3MistralLiteDataset('conf/CRD3Dataset_mistrallite_val.yaml')
    train_dataset = CRD3DecoderOnlyDataset('conf/CRD3Dataset_mistrallite_train.yaml')
    val_dataset = CRD3DecoderOnlyDataset('conf/CRD3Dataset_mistrallite_val.yaml')
    # test_dataset = CRD3DecoderOnlyDataset('conf/CRD3Dataset_mistrallite_val.yaml')

    # train_loader = DataLoader(train_dataset, batch_size=1, num_workers=2)
    # val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2)
    # test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2)

    def eval_generator():
        i = 0
        subset_card = 64
        for val in val_dataset:
            yield val
            i += 1
            if i >= subset_card:
                break

    eval_subset = Dataset.from_generator(eval_generator)

    # model, tokenizer = QuantModelFactory.mistral_7b_unsloth_4bit()
    model, tokenizer = FastLanguageModel.from_pretrained('unsloth/mistral-7b-bnb-4bit', load_in_4bit=True, device_map='auto')
    # model, tokenizer = QuantModelFactory.mistrallite()
    # default_quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,  # bfloat16,  <- My GPU is Turing...
    #     bnb_4bit_use_double_quant=True
    # )
    # model = AutoModelForCausalLM.from_pretrained('amazon/MistralLite', quantization_config=default_quant_config)  # < -- Not enough RAM to load in CPU
    #loftq_config = LoftQConfig(loftq_bits=4)
    # lora_config = LoraConfig(
    #     r=4,
    #     lora_alpha=4,
    #     lora_dropout=0.1,
    #     #init_lora_weights='loftq',
    #     #loftq_config=loftq_config,
    #     task_type='CAUSAL_LM',
    #     inference_mode=False
    # )
    # model = get_peft_model(model, lora_config)
    model = FastLanguageModel.get_peft_model(
        model,
        r=4,
        lora_alpha=4,
        # lora_dropout=0.1,
        # init_lora_weights='loftq',
        # loftq_config=loftq_config,
        # task_type='CAUSAL_LM',
        # inference_mode=False,
        bias='none'
    )
    # accelerator.prepare(
    #     model,
    #     train_loader,
    #     val_loader,
    #     test_loader
    # )

    # I guess this value is used by accelerate
    # The docs just say 'use this'
    # is_ds_zero_3 = False
    # if getattr(accelerator.state, "deepspeed_plugin", None):
    #    is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    # print('Done loading...')
    # input()

    train_args = TrainingArguments(output_dir='unsloth_mistral_train', do_train=True, do_eval=True, max_steps=8192,
                                   evaluation_strategy='steps', per_device_train_batch_size=1,
                                   gradient_accumulation_steps=8, per_device_eval_batch_size=1,
                                   eval_accumulation_steps=1, optim='adamw_8bit', logging_steps=16, fp16=True,
                                   save_steps=128, logging_strategy='steps', eval_steps=64)
    trainer = Trainer(model, args=train_args, train_dataset=train_dataset, eval_dataset=eval_subset, tokenizer=tokenizer)

    trainer.train()


if __name__ == '__main__':
    main()
