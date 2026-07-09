import subprocess
import threading
from typing import Iterator, Protocol
import numpy as np


class AudioSource(Protocol):
    @property
    def sample_rate(self) -> int: ...
    def chunks(self) -> Iterator[np.ndarray]: ...


class FileAudioSource:
    def __init__(self, path: str, chunk_seconds: float = 5.0, sample_rate: int = 16000):
        self.path = path
        self.chunk_seconds = chunk_seconds
        self._sample_rate = sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def chunks(self) -> Iterator[np.ndarray]:
        chunk_samples = int(self.chunk_seconds * self._sample_rate)
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-i", self.path,
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", str(self._sample_rate),
            "-ac", "1",
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        bytes_per_sample = 4  # float32
        chunk_bytes = chunk_samples * bytes_per_sample

        try:
            while True:
                raw = proc.stdout.read(chunk_bytes)
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype=np.float32).copy()
                yield samples
        finally:
            proc.stdout.close()
            proc.wait()


class MicrophoneAudioSource:
    def __init__(self, chunk_seconds: float = 5.0, sample_rate: int = 16000):
        self.chunk_seconds = chunk_seconds
        self._sample_rate = sample_rate
        self._stop = threading.Event()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def stop(self) -> None:
        self._stop.set()

    def chunks(self) -> Iterator[np.ndarray]:
        import sounddevice as sd

        chunk_samples = int(self.chunk_seconds * self._sample_rate)

        while not self._stop.is_set():
            audio = sd.rec(
                chunk_samples,
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            if self._stop.is_set():
                break
            yield audio.flatten()
