import logging
from typing import Optional, Union, BinaryIO, Iterable, Tuple, List, NamedTuple

import ctranslate2
import numpy as np
import torch
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import WhisperModel, TranscriptionOptions, TranscriptionInfo, Word
from faster_whisper.utils import format_timestamp
from transformers import AutoModel

MODEL_SIZE_ENCODE_DIMS = {
    'small.en': 768
}


class Speaker(NamedTuple):
    start: float
    end: float
    speaker: str
    probability: float


class SAWord(NamedTuple):
    start: float
    end: float
    word: str
    probability: float
    speaker: str
    speaker_probability: float


# Pretty big bummer we can't subclass NamedTuple types
class SASegment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: List[SAWord]
    speakers: List[Speaker]


class SequencePool(torch.nn.Module):
    def __init__(
            self,
            batch_first: bool = False
    ):
        super().__init__()
        self.batch_first = batch_first

    def forward(
            self,
            val
    ):
        if self.batch_first:
            return torch.mean(val, 1)
        else:
            return torch.mean(val, 0)


class SpeakerRecognitionDecoder(torch.nn.Module):
    def __init__(
            self,
            num_speakers: int,
            architecture_type: str,
            input_size: int,
            speaker_names: list[str],
            nhead: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            device: str = None,
            num_layers: int = 2,
            dtype: torch.dtype = torch.float32
    ):
        assert architecture_type in ['decoder', 'encoder', 'mlp'],\
            ValueError('Expected architecture_type to be either "decoder" or "encoder"')
        super().__init__()

        # No one should be able to change this - make it private
        # Yes I know python people hate this, but it's justified in this instance, I would say
        self.__architecture_type = architecture_type

        self.speaker_names = speaker_names
        self.device = device

        if architecture_type == 'decoder':
            # Decoder transformer block
            decoder_layer = torch.nn.TransformerDecoderLayer(
                d_model=input_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                device=device,
                dtype=dtype,
                batch_first=True
            )
            decoder_fn = torch.nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_layers
            )

            def decoder_only_decoder_fn(tgt: torch.Tensor, **kwargs):
                return decoder_fn(tgt, torch.clone(tgt), **kwargs)

            self._decoder = decoder_only_decoder_fn
        elif architecture_type == 'encoder':
            # TODO: Fix
            # Encoder transformer block
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                device=device,
                dtype=dtype,
                batch_first=True
            )
            decoder_fn = torch.nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers
            )

            def encoder_only_decoder_fn(tgt: torch.Tensor, **kwargs):
                return decoder_fn(tgt, torch.clone(tgt), **kwargs)

            self._decoder = encoder_only_decoder_fn
        elif architecture_type == 'mlp':
            # A very lightweight MLP to return simply one speaker prediction per sequence
            # Start with global pooling over sequence length
            pool_layer = SequencePool()
            first_layer = torch.nn.Linear(
                in_features=input_size,
                out_features=dim_feedforward,
                device=device,
                dtype=dtype
            )
            in_between_layers = [
                torch.nn.Linear(
                    in_features=dim_feedforward,
                    out_features=dim_feedforward,
                    device=device,
                    dtype=dtype
                )
                for _ in range(num_layers - 2)
            ]
            last_layer = torch.nn.Linear(
                in_features=dim_feedforward,
                out_features=input_size,
                device=device,
                dtype=dtype
            )
            decoder_fn = torch.nn.Sequential(
                pool_layer,
                first_layer,
                *in_between_layers,
                last_layer
            )

            def mlp_decoder_fn(tgt: torch.Tensor, **kwargs):
                return decoder_fn(tgt)

            self._decoder = mlp_decoder_fn

        # Linear + smax for classification
        self._linear = torch.nn.Linear(
            in_features=input_size,
            out_features=num_speakers,
            device=device,
            dtype=dtype
        )
        self._smax = torch.nn.Softmax(dim=-1)

    @property
    def architecture_type(self):
        return self.__architecture_type

    def forward(
            self,
            tgt: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        print(tgt.shape)
        decoder_out = self._decoder(tgt, **kwargs)
        print(decoder_out.shape)
        logits = self._linear(decoder_out)
        print(logits.shape)
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


class SAASRModel(WhisperModel):
    def __init__(
            self,
            whisper_model_size_or_path: str,
            speaker_model_path: str = None,
            whisper_model_device: str = "auto",
            speaker_model_device: str = "auto",
            whisper_device_index: Union[int, List[int]] = 0,
            compute_type: str = "default",
            cpu_threads: int = 0,
            num_workers: int = 1,
            download_root: Optional[str] = None,
            local_files_only: bool = False,
            speaker_model_kwargs: dict = None
    ) -> None:
        super().__init__(
            model_size_or_path=whisper_model_size_or_path,
            device=whisper_model_device,
            device_index=whisper_device_index,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            download_root=download_root,
            local_files_only=local_files_only
        )
        if speaker_model_path is not None:
            # Load a pretrained speaker recognition model
            self._speaker_recognition_model: SpeakerRecognitionDecoder = AutoModel.from_pretrained(
                speaker_model_path,
                device_map=speaker_model_device
            )
        else:
            # Create a new speaker recognition model for training
            assert speaker_model_kwargs is not None,\
                ValueError('One of speaker_model_kwargs or speaker_model_path must be passed')
            self._speaker_recognition_model = SpeakerRecognitionDecoder(**speaker_model_kwargs)

    def forward(
            self,
            waveform: Union[str, BinaryIO, np.ndarray, torch.Tensor]
    ):
        """Performs a forward pass ONLY returning the speaker labels. This is for training a speaker recognition model
        with a pre-trained whisper encoder.

        Parameters
        ----------
        waveform

        Returns
        -------

        """
        features = self.feature_extractor(waveform)
        encoder_output = self.encode(features)
        speaker_output = self._speaker_recognition_model(encoder_output)
        return speaker_output

    def transcribe(self, *args, **kwargs) -> Tuple[Iterable[SASegment], TranscriptionInfo]:
        # I don't want to rewrite the whole method simply to fix the type hints
        # So I'm just calling super and superficially overriding the return type
        # The IDE will whine because it can't figure out the
        # transcribe return type changes with the generate_segments return type
        return super().transcribe(*args, **kwargs)  # wah wah me no read code too gud

    def generate_segments(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        encoder_output: Optional[ctranslate2.StorageView] = None,
    ) -> Iterable[SASegment]:
        content_frames = features.shape[-1] - self.feature_extractor.nb_max_frames
        idx = 0
        seek = 0
        all_tokens = []
        prompt_reset_since = 0

        if options.initial_prompt is not None:
            if isinstance(options.initial_prompt, str):
                initial_prompt = " " + options.initial_prompt.strip()
                initial_prompt_tokens = tokenizer.encode(initial_prompt)
                all_tokens.extend(initial_prompt_tokens)
            else:
                all_tokens.extend(options.initial_prompt)

        last_speech_timestamp = 0.0
        while seek < content_frames:
            time_offset = seek * self.feature_extractor.time_per_frame
            segment = features[:, seek: seek + self.feature_extractor.nb_max_frames]
            segment_size = min(self.feature_extractor.nb_max_frames, content_frames - seek)
            segment_duration = segment_size * self.feature_extractor.time_per_frame

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Processing segment at %s", format_timestamp(time_offset))

            previous_tokens = all_tokens[prompt_reset_since:]
            prompt = self.get_prompt(
                tokenizer,
                previous_tokens,
                without_timestamps=options.without_timestamps,
                prefix=options.prefix if seek == 0 else None,
            )

            if seek > 0 or encoder_output is None:
                encoder_output = self.encode(segment)

            # TODO: This is not very elegant
            # Extract the CTranslate2 StorageView into a Tensor
            n_segment_frames = encoder_output.shape[1]
            print(f'time_offset: {type(time_offset)}')
            print(f'n_segment_frames: {type(n_segment_frames)}')
            segment_duration = self.feature_extractor.time_per_frame * n_segment_frames
            print(f'segment_duration: {type(segment_duration)}')
            print(encoder_output.shape, n_segment_frames, self.feature_extractor.time_per_frame, segment_duration)
            speaker_recog_input = torch.clone(torch.tensor(encoder_output))\
                .to(torch.float32)\
                .to(self._speaker_recognition_model.device)
            print(speaker_recog_input.shape, speaker_recog_input.dtype)

            # Run speaker attribution for this segment
            # Segments are quite short
            # TODO: Check iteration over batch sizes
            if self._speaker_recognition_model.architecture_type in ['mlp']:
                segment_speaker_logits = self._speaker_recognition_model(speaker_recog_input)  # [n_speaker]
                segment_speaker_id = segment_speaker_logits.argmax(-1)  # scalar
                speakers = [
                    Speaker(
                        start=time_offset,
                        end=time_offset + segment_duration,
                        speaker=segment_speaker_id,
                        probability=segment_speaker_logits[segment_speaker_id]
                    ),
                ]
            else:
                speaker_probs = self._speaker_recognition_model(speaker_recog_input)  # [batch, seq_len, n_speaker]
                speaker_ids = speaker_probs.squeeze().argmax(-1)  # [seq_len]  TODO: batch size?
                speaker_changes = torch.nonzero(speaker_ids[1:] - speaker_ids[:-1])  # [seq_len]
                if len(speaker_changes) == 0:
                    # No speaker changes, use mean prob of only speaker
                    speaker_id = speaker_ids[0]
                    speaker_name = self._speaker_recognition_model.speaker_names[speaker_id]
                    speakers = [
                        Speaker(
                            start=time_offset,
                            end=time_offset + segment_duration,
                            speaker=speaker_name,
                            probability=speaker_probs[..., speaker_id].mean()
                        ),
                    ]
                else:
                    speakers = [
                        Speaker(
                            start=time_offset + start_idx.item() * self.feature_extractor.time_per_frame,
                            end=time_offset + stop_idx.item() * self.feature_extractor.time_per_frame,
                            speaker=self._speaker_recognition_model.speaker_names[speaker_ids[start_idx]],
                            probability=speaker_probs[0, start_idx: stop_idx, speaker_ids[start_idx]].mean().item()  # TODO: Kinda hacky, could be on word level
                        )
                        for start_idx, stop_idx in zip(speaker_changes[:-1], speaker_changes[1:])
                    ]
            # TODO: Consider encoder-decoder with whisper encoding and generated tokens

            (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            ) = self.generate_with_fallback(encoder_output, prompt, tokenizer, options)

            if options.no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > options.no_speech_threshold

                if options.log_prob_threshold is not None and avg_logprob > options.log_prob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    self.logger.debug(
                        "No speech threshold is met (%f > %f)",
                        result.no_speech_prob,
                        options.no_speech_threshold,
                    )

                    # fast-forward to the next segment boundary
                    seek += segment_size
                    continue

            # TODO: Speaker recognition that also takes in generation logits/sequences
            tokens = result.sequences_ids[0]

            previous_seek = seek
            current_segments = []

            single_timestamp_ending = len(tokens) >= 2 and tokens[-2] < tokenizer.timestamp_begin <= tokens[-1]

            consecutive_timestamps = [
                i
                for i in range(len(tokens))
                if i > 0 and tokens[i] >= tokenizer.timestamp_begin and tokens[i - 1] >= tokenizer.timestamp_begin
            ]

            if len(consecutive_timestamps) > 0:
                slices = list(consecutive_timestamps)
                if single_timestamp_ending:
                    slices.append(len(tokens))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_position = sliced_tokens[0] - tokenizer.timestamp_begin
                    end_timestamp_position = sliced_tokens[-1] - tokenizer.timestamp_begin
                    start_time = time_offset + start_timestamp_position * self.time_precision
                    end_time = time_offset + end_timestamp_position * self.time_precision

                    current_segments.append(
                        dict(
                            seek=seek,
                            start=start_time,
                            end=end_time,
                            tokens=sliced_tokens,
                        )
                    )
                    last_slice = current_slice

                if single_timestamp_ending:
                    # single timestamp at the end means no speech after the last timestamp.
                    seek += segment_size
                else:
                    # otherwise, ignore the unfinished segment and seek to the last timestamp
                    last_timestamp_position = tokens[last_slice - 1] - tokenizer.timestamp_begin
                    seek += last_timestamp_position * self.input_stride

            else:
                duration = segment_duration
                timestamps = [token for token in tokens if token >= tokenizer.timestamp_begin]
                if len(timestamps) > 0 and timestamps[-1] != tokenizer.timestamp_begin:
                    last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
                    duration = last_timestamp_position * self.time_precision

                current_segments.append(
                    dict(
                        seek=seek,
                        start=time_offset,
                        end=time_offset + duration,
                        tokens=tokens,
                    )
                )

                seek += segment_size

            # Timestamps are necessary to align speaker labels
            self.add_word_timestamps(
                current_segments,
                tokenizer,
                encoder_output,
                segment_size,
                options.prepend_punctuations,
                options.append_punctuations,
                last_speech_timestamp=last_speech_timestamp,
            )

            word_end_timestamps = [w["end"] for s in current_segments for w in s["words"]]
            if len(word_end_timestamps) > 0:
                last_speech_timestamp = word_end_timestamps[-1]
            if not single_timestamp_ending and len(word_end_timestamps) > 0:
                seek_shift = round((word_end_timestamps[-1] - time_offset) * self.frames_per_second)

                if seek_shift > 0:
                    seek = previous_seek + seek_shift

            for segment in current_segments:
                tokens = segment["tokens"]
                text = tokenizer.decode(tokens)

                # Find corresponding speaker for each word
                word_idx = 0
                holdover_word = False
                for spkr in speakers:
                    # In the case the previous speaker changed without covering 50% of the word duration
                    if holdover_word:
                        segment['words'][word_idx]['speaker'] = spkr.speaker
                        segment['words'][word_idx]['speaker_probability'] = spkr.probability
                        word_idx += 1

                    # Break condition
                    if word_idx >= len(segment['words']):
                        break

                    # Initial calculation
                    current_word = segment['words'][word_idx]
                    word_duration = current_word['end'] - current_word['start']
                    intersect_duration = min(current_word['end'], spkr.end) - max(current_word['start'], spkr.start)
                    timestamp_overlap = intersect_duration / word_duration

                    # If the speaker timestamps overlap at least 50% of the word timestamp
                    while timestamp_overlap >= 0.5:
                        segment['words'][word_idx]['speaker'] = spkr.speaker
                        segment['words'][word_idx]['speaker_probability'] = spkr.probability

                        word_idx += 1
                        if word_idx >= len(segment['words']) - 1:
                            break
                        current_word = segment['words'][word_idx]
                        word_duration = current_word['end'] - current_word['start']
                        intersect_duration = min(current_word['end'], spkr.end) - max(current_word['start'], spkr.start)
                        timestamp_overlap = intersect_duration / word_duration
                    else:
                        holdover_word = True
                # If there are somehow words after the speaker labels, label them as 'UNKNOWN'
                if word_idx < len(segment['words']):
                    segment['words'][word_idx]['speaker'] = 'UNKNOWN'
                    segment['words'][word_idx]['speaker_probability'] = 0.5

                if segment["start"] == segment["end"] or not text.strip():
                    continue

                all_tokens.extend(tokens)
                idx += 1

                yield SASegment(
                    id=idx,
                    seek=seek,
                    start=segment["start"],
                    end=segment["end"],
                    text=text,
                    tokens=tokens,
                    temperature=temperature,
                    avg_logprob=avg_logprob,
                    compression_ratio=compression_ratio,
                    no_speech_prob=result.no_speech_prob,
                    words=(
                        [SAWord(**word) for word in segment["words"]]
                        #if options.word_timestamps
                        #else None
                    ),
                    speakers=speakers
                )

            if (
                    not options.condition_on_previous_text
                    or temperature > options.prompt_reset_on_temperature
            ):
                if options.condition_on_previous_text:
                    self.logger.debug(
                        "Reset prompt. prompt_reset_on_temperature threshold is met %f > %f",
                        temperature,
                        options.prompt_reset_on_temperature,
                    )

                prompt_reset_since = len(all_tokens)
