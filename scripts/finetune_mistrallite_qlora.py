# This script simply tests that you can load the model into GPU memory


from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, LoftQConfig, get_peft_model

from torch.utils.data import DataLoader


from CRD3_summarization.models.HuggingfaceModels import QuantModelFactory
from CRD3_summarization.loaders import CRD3MistralLiteDataset


def main():
    accelerator = Accelerator()
    train_dataset = CRD3MistralLiteDataset('conf/CRD3Dataset_mistrallite_train.yaml')
    val_dataset = CRD3MistralLiteDataset('conf/CRD3Dataset_mistrallite_val.yaml')
    test_dataset = CRD3MistralLiteDataset('conf/CRD3Dataset_mistrallite_val.yaml')

    model, tokenizer = QuantModelFactory.mistrallite(device_map='cuda:0')
    # model = AutoModelForCausalLM.from_pretrained('amazon/MistralLite')  < -- Not enough RAM to load in CPU
    #loftq_config = LoftQConfig(loftq_bits=4)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        #init_lora_weights='loftq',
        #loftq_config=loftq_config,
        task_type='CAUSAL_LM',
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=True)

    accelerator.prepare(
        model,
        train_loader,
        val_loader,
        test_loader
    )

    # I guess this value is used by accelerate
    # The docs just say 'use this'
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3





if __name__ == '__main__':
    main()
