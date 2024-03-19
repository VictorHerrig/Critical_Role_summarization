import itertools
import logging
from typing import Optional, Union, BinaryIO, Iterable, Tuple, List, NamedTuple

import ctranslate2
import numpy as np
import torch
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import WhisperModel, TranscriptionOptions, TranscriptionInfo, merge_punctuations
from faster_whisper.utils import format_timestamp
from transformers import AutoModel

MODEL_SIZE_ENCODE_DIMS = {
    'small.en': 768
}


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


class SpeakerRecognitionDecoder(torch.nn.Module):
    def __init__(
            self,
            num_speakers: int,
            input_size: int,
            speaker_names: list[str],
            nhead: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            device: str = None,
            num_layers: int = 2,
            vocab_size:int = 50363,
            dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        self.speaker_names = speaker_names
        self.device = device

        # Decoder transformer block
        self.embed_layer = torch.nn.Embedding(vocab_size, embedding_dim=input_size)
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=input_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device=device,
            dtype=dtype,
            batch_first=True
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

        # Linear + smax for classification
        self._linear = torch.nn.Linear(
            in_features=input_size,
            out_features=num_speakers,
            device=device,
            dtype=dtype
        )
        self._smax = torch.nn.Softmax(dim=-1)

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        print(tgt.shape, memory.shape)
        print(tgt[:, :20])
        embeds = self.embed_layer(tgt)
        print(embeds.shape)
        decoder_out = self.transformer_decoder(embeds, memory, **kwargs)

        logits = self._linear(decoder_out)
        return self._smax(logits)


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
            waveform: Union[str, BinaryIO, np.ndarray, torch.Tensor],
            tokenizer: Tokenizer,
            options: TranscriptionOptions
    ):
        """Performs a forward pass ONLY returning the speaker labels. This is for training a speaker recognition model
        with a pre-trained whisper encoder.

        Parameters
        ----------
        waveform: numpy.ndarray
            Waveform input as a numpy array
        tokenizer: Tokenizer
            Faster whisper tokenizer
        options: TranscriptionOptions
            Faster whisper transcription options. Use what you will use during generation.

        Returns
        -------
        speaker_probs: torch.Tensor
            Tensor of shape [seq_len, n_speakers]
        """
        # Audio encoder
        features = self.feature_extractor(waveform)
        encoder_output = self.encode(features)

        # Text decoder
        prompt = self.get_prompt(
            tokenizer,
            [],
            without_timestamps=False,
            prefix=None,
        )
        (
            result,
            avg_logprob,
            temperature,
            compression_ratio,
        ) = self.generate_with_fallback(encoder_output, prompt, tokenizer, options)

        speaker_recog_input = torch.clone(torch.tensor(encoder_output)) \
            .to(torch.float32) \
            .to(self._speaker_recognition_model.device)

        tokens = result.sequences_ids[0]
        single_timestamp_ending = len(tokens) >= 2 and tokens[-2] < tokenizer.timestamp_begin <= tokens[-1]
        consecutive_timestamps = [
            i
            for i in range(len(tokens))
            if i > 0 and tokens[i] >= tokenizer.timestamp_begin and tokens[i - 1] >= tokenizer.timestamp_begin
        ]

        # Shortened version to return only speaker probs
        if len(consecutive_timestamps) > 0:
            slices = list(consecutive_timestamps)
            if single_timestamp_ending:
                slices.append(len(tokens))

            last_slice = 0
            speaker_probs = []
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                slice_speaker_probs = self._speaker_recognition_model(
                    tgt=torch.tensor([sliced_tokens], dtype=torch.int32),  # [1, seq_len]
                    memory=speaker_recog_input  # [1, n_slice_frames, model_dim]
                )[0]  # [slice_seq_len, n_speaker]
                speaker_probs.append(slice_speaker_probs)
                last_slice = current_slice
            torch.cat(speaker_probs, 0)  # [seq_len, n_speaker]

        else:
            text_tokens = [token for token in tokens if token < tokenizer.timestamp_begin]
            speaker_probs = self._speaker_recognition_model(
                tgt=torch.tensor([text_tokens], dtype=torch.int32),  # [1, seq_len]
                memory=speaker_recog_input  # [1, n_frames, model_dim]
            )[0]  # [seq_len, n_speaker]

        return speaker_probs

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
        seek = 0
        all_tokens = []

        if options.initial_prompt is not None:
            if isinstance(options.initial_prompt, str):
                initial_prompt = " " + options.initial_prompt.strip()
                initial_prompt_tokens = tokenizer.encode(initial_prompt)
                all_tokens.extend(initial_prompt_tokens)
            else:
                all_tokens.extend(options.initial_prompt)

        content_frames = features.shape[-1] - self.feature_extractor.nb_max_frames
        last_speech_timestamp = 0.0
        prompt_reset_since = 0
        while seek < content_frames:
            time_offset = seek * self.feature_extractor.time_per_frame
            segment = features[:, seek: seek + self.feature_extractor.nb_max_frames]
            segment_size = min(self.feature_extractor.nb_max_frames, content_frames - seek)

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

            (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            ) = self.generate_with_fallback(encoder_output, prompt, tokenizer, options)

            # TODO: This is not very elegant
            # Extract the CTranslate2 StorageView into a Tensor
            n_segment_frames = encoder_output.shape[1]
            segment_duration = self.feature_extractor.time_per_frame * n_segment_frames
            speaker_recog_input = torch.clone(torch.tensor(encoder_output)) \
                .to(torch.float32) \
                .to(self._speaker_recognition_model.device)

            # Check voice activity
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

            tokens = result.sequences_ids[0]
            for token, score in zip(result.sequences_ids, result.scores):
                print(f'{token}: {score}')

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

                    speaker_probs = self._speaker_recognition_model(
                        tgt=torch.tensor([sliced_tokens], dtype=torch.int32),  # [1, seq_len]
                        memory=speaker_recog_input  # [1, n_slice_frames, model_dim]
                    )[0]  # [slice_seq_len, n_speaker]
                    speaker_ids = speaker_probs.squeeze().argmax(-1)  # [seq_len]

                    current_segments.append(
                        dict(
                            seek=seek,
                            start=start_time,
                            end=end_time,
                            tokens=sliced_tokens,
                            speakers=speaker_ids.tolist(),
                            speaker_probs=speaker_probs
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

                text_tokens = [token for token in tokens if token < tokenizer.timestamp_begin]
                speaker_probs = self._speaker_recognition_model(
                    tgt=torch.tensor([text_tokens], dtype=torch.int32),  # [1, seq_len]
                    memory=speaker_recog_input  # [1, n_frames, model_dim]
                )[0]  # [seq_len, n_speaker]
                speaker_ids = speaker_probs.squeeze().argmax(-1)  # [seq_len]

                current_segments.append(
                    dict(
                        seek=seek,
                        start=time_offset,
                        end=time_offset + duration,
                        tokens=tokens,
                        speakers=speaker_ids.tolist(),
                        speaker_probs=speaker_probs
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

            idx = 0
            for segment in current_segments:
                tokens = segment["tokens"]
                text = tokenizer.decode(tokens)

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
                    words=([SAWord(**word) for word in segment["words"]])
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

    def add_word_timestamps(
            self,
            segments: List[dict],
            tokenizer: Tokenizer,
            encoder_output: ctranslate2.StorageView,
            num_frames: int,
            prepend_punctuations: str,
            append_punctuations: str,
            last_speech_timestamp: float,
    ) -> None:
        if len(segments) == 0:
            return

        text_tokens_per_segment = [
            [token for token in segment["tokens"] if token < tokenizer.eot]
            for segment in segments
        ]

        text_tokens = list(itertools.chain.from_iterable(text_tokens_per_segment))
        alignment = self.find_alignment(
            tokenizer, text_tokens, encoder_output, num_frames
        )
        word_durations = np.array([word["end"] - word["start"] for word in alignment])
        word_durations = word_durations[word_durations.nonzero()]
        median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
        median_duration = min(0.7, float(median_duration))
        max_duration = median_duration * 2

        # hack: truncate long words at sentence boundaries.
        # a better segmentation algorithm based on VAD should be able to replace this.
        if len(word_durations) > 0:
            sentence_end_marks = ".。!！?？"
            # ensure words at sentence boundaries
            # are not longer than twice the median word duration.
            for i in range(1, len(alignment)):
                if alignment[i]["end"] - alignment[i]["start"] > max_duration:
                    if alignment[i]["word"] in sentence_end_marks:
                        alignment[i]["end"] = alignment[i]["start"] + max_duration
                    elif alignment[i - 1]["word"] in sentence_end_marks:
                        alignment[i]["start"] = alignment[i]["end"] - max_duration

        merge_punctuations(alignment, prepend_punctuations, append_punctuations)

        time_offset = (
                segments[0]["seek"]
                * self.feature_extractor.hop_length
                / self.feature_extractor.sampling_rate
        )

        word_index = 0
        token_idx = 0

        for segment, text_tokens in zip(segments, text_tokens_per_segment):
            saved_tokens = 0
            words = []

            while word_index < len(alignment) and saved_tokens < len(text_tokens):
                timing = alignment[word_index]

                if timing["word"]:

                    word_tokens = timing["tokens"]
                    matching_subsets = [text_tokens[i: i + len(word_tokens)] == word_tokens for i in range(token_idx, len(text_tokens))]
                    if not any(matching_subsets):
                        raise ValueError()
                    start_token_idx = np.argmax(matching_subsets)
                    end_token_idx = start_token_idx + len(word_tokens)
                    token_idx = end_token_idx

                    # start_token_idx, end_token_idx = word_token_idxs[word_index + 1]
                    speaker_probs = segment['speaker_probs'][start_token_idx: end_token_idx].mean(0)  # [n_spkr]
                    word_speaker_id = speaker_probs.argmax()
                    word_speaker_name = self.speaker_names[word_speaker_id]
                    word_speaker_prob = speaker_probs[word_speaker_id].item()

                    words.append(
                        dict(
                            word=timing["word"],
                            start=round(time_offset + timing["start"], 2),
                            end=round(time_offset + timing["end"], 2),
                            probability=timing["probability"],
                            speaker=word_speaker_name,
                            speaker_probability=word_speaker_prob,
                        )
                    )

                saved_tokens += len(timing["tokens"])
                word_index += 1

            # hack: truncate long words at segment boundaries.
            # a better segmentation algorithm based on VAD should be able to replace this.
            if len(words) > 0:
                # ensure the first and second word after a pause is not longer than
                # twice the median word duration.
                if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
                        words[0]["end"] - words[0]["start"] > max_duration
                        or (
                                len(words) > 1
                                and words[1]["end"] - words[0]["start"] > max_duration * 2
                        )
                ):
                    if (
                            len(words) > 1
                            and words[1]["end"] - words[1]["start"] > max_duration
                    ):
                        boundary = max(
                            words[1]["end"] / 2, words[1]["end"] - max_duration
                        )
                        words[0]["end"] = words[1]["start"] = boundary
                    words[0]["start"] = max(0, words[0]["end"] - max_duration)

                # prefer the segment-level start timestamp if the first word is too long.
                if (
                        segment["start"] < words[0]["end"]
                        and segment["start"] - 0.5 > words[0]["start"]
                ):
                    words[0]["start"] = max(
                        0, min(words[0]["end"] - median_duration, segment["start"])
                    )
                else:
                    segment["start"] = words[0]["start"]

                # prefer the segment-level end timestamp if the last word is too long.
                if (
                        segment["end"] > words[-1]["start"]
                        and segment["end"] + 0.5 < words[-1]["end"]
                ):
                    words[-1]["end"] = max(
                        words[-1]["start"] + median_duration, segment["end"]
                    )
                else:
                    segment["end"] = words[-1]["end"]

                last_speech_timestamp = segment["end"]

            segment["words"] = words

    @property
    def speaker_names(self):
        return self._speaker_recognition_model.speaker_names
