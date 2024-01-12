import torch
import transformers

from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig


class QuantModelFactory:
    @staticmethod
    def load(
            model_path: str,
            tokenizer_path: str = None,
            quant_config: dict = None
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
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        quant_config = quant_config if quant_config is not None else default_quant_config
        quant_config = BitsAndBytesConfig(**quant_config)

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path if tokenizer_path is None else tokenizer_path
        )
        model = AutoModel.from_pretrained(
            model_path,
            quantization_config=quant_config
        )

        return model, tokenizer

    @staticmethod
    def mistral_7b():
        return QuantModelFactory.load(
            model_path='mistralai/Mistral-7B-v0.1'
        )
