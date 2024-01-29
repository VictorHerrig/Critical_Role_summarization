import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import AutoModel


MODEL_SIZE_ENCODE_DIMS = {
    'small.en': 768
}


class SpeakerRecognitionDecoder(torch.nn.Module):
    def __init__(
            self,
            num_speakers: int,
            architecture_type: str,
            input_size: int,
            nhead: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            device: str = None,
            num_layers: int = 2
    ):
        super().__init__()
        assert architecture_type in ['decoder', 'encoder'],\
            ValueError('Expected architecture_type to be either "decoder" or "encoder"')

        # Transformer block
        if architecture_type == 'decoder':
            # Unidirectional
            decoder_layer = torch.nn.TransformerDecoderLayer(
                d_model=input_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                device=device
            )
            self._decoder = torch.nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_layers
            )
        else:
            # Bidirectional
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                device=device
            )
            self._decoder = torch.nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers
            )

        # Linear + smax for classification
        self._linear = torch.nn.Linear(
            in_features=input_size,
            out_features=num_speakers
        )
        self._smax = torch.nn.Softmax(
            dim=num_speakers
        )

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        decoder_out = self._decoder(tgt, memory, tgt_is_causal=False, memory_is_causal=False)
        logits = self._linear(decoder_out)
        return self._smax(logits)

    @classmethod
    def encoder_model(
            cls,
            num_speakers: int,
            input_size: int,
            **kwargs
    ):
        return cls(
            num_speakers=num_speakers,
            architecture_type='encoder',
            input_size=input_size,
            **kwargs
        )

    @classmethod
    def decoder_model(
            cls,
            num_speakers: int,
            input_size: int,
            **kwargs
    ):
        return cls(
            num_speakers=num_speakers,
            architecture_type='decoder',
            input_size=input_size,
            **kwargs
        )


class SAASRModel(torch.nn.Module):
    def __init__(
            self,
            asr_model_size_or_path: str,
            speaker_model_path: str = None,
            use_gpu_for_asr: bool = False,
            use_gpu_for_speaker: bool = False,
            compute_type: str = 'default',
            cpu_threads: int = 0,
            num_workers: int = 1,
            speaker_model_kwargs: dict = None
    ) -> None:
        super().__init__()
        self._whipser_model = WhisperModel(
            model_size_or_path=asr_model_size_or_path,
            device='auto' if use_gpu_for_asr else 'cpu',  # TODO: Make this better
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers
        )
        if speaker_model_path is not None:
            # Load a pretrained speaker recognition model
            self._speaker_recognition_model: SpeakerRecognitionDecoder = AutoModel.from_pretrained(
                speaker_model_path,
                device_map='auto' if use_gpu_for_speaker else 'cpu'  # TODO: Make this better
            )
        else:
            # Create a new speaker recognition model for training
            assert speaker_model_kwargs is not None,\
                ValueError('One of speaker_model_kwargs or speaker_model_path must be passed')
            self._speaker_recognition_model = SpeakerRecognitionDecoder(**speaker_model_kwargs)

    def forward(
            self,
            waveform: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        features = self._whipser_model.feature_extractor(waveform)
        encoding = self._whipser_model.encode(features)

        speaker_output = self._speaker_recognition_model(encoding)

        # TODO: Look inside the faster_whisper codebase
        ...

    def generate(
            self,
            *args,
            **kwargs
    ) -> ...:
        raise NotImplementedError()
