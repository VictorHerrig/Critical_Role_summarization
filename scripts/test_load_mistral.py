import torch

from transformers import AutoModel
from transformers import BitsAndBytesConfig


def main():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModel.from_pretrained(
        'mistralai/Mistral-7B-v0.1',
        quantization_config=quant_config
    )
    input()  # Wait and check VRAM


if __name__ == '__main__':
    main()
