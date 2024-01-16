import torch
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from unsloth import FastLanguageModel


class QuantModelFactory:
    @staticmethod
    def load(
            model_path: str,
            tokenizer_path: str = None,
            quant_config: dict = None,
            **kwargs
    ) -> (transformers.PreTrainedModel, transformers.PreTrainedTokenizer):
        """

        Parameters
        ----------
        model_path: str
            String passed to the `from_pretrained` function. A path, either local or on the Huggingface hub.
        tokenizer_path: str, optional
             String passed to the `from_pretrained` function when loading a tokenizer. If left unspecified, will be the
             same as `model_path`. (default = None)
        quant_config: dict, optional
            Dictionary of quantization configurations. By default, uses double 4-bit quantization with bfloat16 compute
            type. If left None, will use the default. (default = None)

        Returns
        -------
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        """
        # Override quant defaults if provided
        default_quant_config = dict(
            load_in_4bit=True,
            #bnb_4bit_compute_dtype=torch.float16,  # bfloat16,  <- My GPU is Turing...
            #bnb_4bit_use_double_quant=True,
            device_map='auto'  # TODO: Check
        )
        quant_config = quant_config if quant_config is not None else default_quant_config
        # quant_config = BitsAndBytesConfig(**quant_config)

        # Load model and tokenizer
        # tokenizer = AutoTokenizer.from_pretrained(
        #     model_path if tokenizer_path is None else tokenizer_path
        # )
        # Override default kwargs with passed kwargs
        # arg_dict = dict(dict(quantization_config=quant_config, device_map='auto'), **kwargs)
        arg_dict = dict(quant_config, **kwargs)
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     **arg_dict
        # )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            **arg_dict
        )

        return model, tokenizer

    @staticmethod
    def mistral_7b(**kwargs):
        return QuantModelFactory.load(
            model_path='mistralai/Mistral-7B-v0.1',
            **kwargs
        )

    @staticmethod
    def mistral_7b_flash(**kwargs):
        return QuantModelFactory.load(
            model_path='mistralai/Mistral-7B-v0.1',
            attn_implementation='flash_attention_2',
            **kwargs
        )

    @staticmethod
    def mistrallite(**kwargs):
        return QuantModelFactory.load(
            model_path='amazon/MistralLite',
            **kwargs
        )

    @staticmethod
    def mistrallite_flash(**kwargs):
        return QuantModelFactory.load(
            model_path='amazon/MistralLite',
            attn_implementation='flash_attention_2',
            **kwargs
        )

    @staticmethod
    def mistral_7b_unsloth_4bit(**kwargs):
        return QuantModelFactory.load(
            model_path='unsloth/mistral-7b-bnb-4bit',
            **kwargs
        )
