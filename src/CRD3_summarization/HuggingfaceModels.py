import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from unsloth import FastLanguageModel


class QuantModelFactory:
    @staticmethod
    def load(
            model_path: str,
            use_unsloth: bool = False,
            quant_config: dict = None,
            **kwargs
    ) -> (transformers.PreTrainedModel, transformers.PreTrainedTokenizer):
        """

        Parameters
        ----------
        model_path: str
            String passed to the `from_pretrained` function. A path, either local or on the Huggingface hub.
        use_unsloth: bool, optional
             Whether to use unsloth model and tokenizer. (default = False)
        quant_config: dict, optional
            Dictionary of quantization configurations. By default, uses double 4-bit quantization with bfloat16 compute
            type. If left None, will use the default. (default = None)

        Returns
        -------
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        """
        # Unsloth uses a different class to initialize models and tokens
        if use_unsloth:
            return FastLanguageModel.from_pretrained(
                model_path,
                load_in_4bit=True,
                device_map='auto',
                **kwargs
            )

        # Override quant defaults if provided
        default_quant_config = dict(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            device_map='auto'
        )
        quant_config = quant_config if quant_config is not None else default_quant_config
        quant_config = BitsAndBytesConfig(**quant_config)

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Override default kwargs with passed kwargs
        arg_dict = dict(dict(quantization_config=quant_config, device_map='auto'), **kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_path, **arg_dict)

        return model, tokenizer

    @staticmethod
    def mistral_7b(**kwargs):
        """Constructs a Mistral object."""
        return QuantModelFactory.load(
            model_path='mistralai/Mistral-7B-v0.1',
            **kwargs
        )

    @staticmethod
    def mistral_7b_flash(**kwargs):
        """Constructs a Mistral object with flash attention."""
        return QuantModelFactory.load(
            model_path='mistralai/Mistral-7B-v0.1',
            attn_implementation='flash_attention_2',
            **kwargs
        )

    @staticmethod
    def mistrallite(**kwargs):
        """Constructs a Mistrallite object."""
        return QuantModelFactory.load(
            model_path='amazon/MistralLite',
            **kwargs
        )

    @staticmethod
    def mistrallite_flash(**kwargs):
        """Constructs a Mistrallite object with flash attention."""
        return QuantModelFactory.load(
            model_path='amazon/MistralLite',
            attn_implementation='flash_attention_2',
            **kwargs
        )

    @staticmethod
    def mistral_7b_unsloth_4bit(**kwargs):
        """Constructs a Mistral object using unsloth."""
        return QuantModelFactory.load(
            model_path='unsloth/mistral-7b-bnb-4bit',
            use_unsloth=True,
            **kwargs
        )
