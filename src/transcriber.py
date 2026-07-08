from collections import namedtuple
from typing import Iterator
import numpy as np

from faster_whisper import WhisperModel

from audio_source import AudioSource

WordToken = namedtuple("WordToken", ["word", "start_sec", "end_sec"])

SILENCE_RMS_THRESHOLD = 0.01  # chunks below this RMS energy are treated as silence


class WhisperTranscriber:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_file(self, path: str) -> Iterator[WordToken]:
        """One-shot transcription of a full audio file. Best accuracy."""
        segments, _ = self._model.transcribe(path, word_timestamps=True, vad_filter=True)
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    yield WordToken(
                        word=word.word.strip(),
                        start_sec=word.start,
                        end_sec=word.end,
                    )

    def transcribe_chunks(
        self,
        source: AudioSource,
        overlap_seconds: float = 1.0,
    ) -> Iterator[WordToken]:
        """
        Chunked transcription for real-time use.
        Yields WordTokens with absolute timestamps as each chunk is processed.
        Overlap between consecutive chunks prevents boundary word drops.
        """
        sample_rate = source.sample_rate
        overlap_samples = int(overlap_seconds * sample_rate)

        chunk_offset_sec = 0.0
        prev_overlap: np.ndarray = np.array([], dtype=np.float32)
        last_yielded_end = 0.0

        for chunk in source.chunks():
            chunk_duration = len(chunk) / sample_rate

            # Skip silent chunks — energy check before sending to Whisper
            rms = np.sqrt(np.mean(chunk ** 2))
            if rms < SILENCE_RMS_THRESHOLD:
                chunk_offset_sec += chunk_duration
                prev_overlap = np.array([], dtype=np.float32)
                continue

            if len(prev_overlap) > 0:
                audio = np.concatenate([prev_overlap, chunk])
                overlap_duration = len(prev_overlap) / sample_rate
            else:
                audio = chunk
                overlap_duration = 0.0

            segments, _ = self._model.transcribe(audio, word_timestamps=True)

            for segment in segments:
                if not segment.words:
                    continue
                for word in segment.words:
                    abs_start = chunk_offset_sec - overlap_duration + word.start
                    abs_end = chunk_offset_sec - overlap_duration + word.end
                    # Skip words already yielded from previous chunk's overlap
                    if abs_start < last_yielded_end - 0.05:
                        continue
                    token = WordToken(
                        word=word.word.strip(),
                        start_sec=round(abs_start, 3),
                        end_sec=round(abs_end, 3),
                    )
                    yield token
                    last_yielded_end = abs_end

            # Keep last `overlap_seconds` of this chunk as overlap for next
            prev_overlap = chunk[-overlap_samples:] if len(chunk) > overlap_samples else chunk
            chunk_offset_sec += chunk_duration
