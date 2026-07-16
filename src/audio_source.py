import queue
import subprocess
import threading
from typing import Iterator, Protocol
import numpy as np


class AudioSource(Protocol):
    @property
    def sample_rate(self) -> int: ...
    def chunks(self) -> Iterator[np.ndarray]: ...


class FileAudioSource:
    def __init__(self, path: str, chunk_seconds: float = 1.0, sample_rate: int = 16000):
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
    def __init__(self, chunk_seconds: float = 1.0, sample_rate: int = 16000):
        self.chunk_seconds = chunk_seconds
        self._sample_rate = sample_rate
        self._stop = threading.Event()
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def stop(self) -> None:
        self._stop.set()

    def chunks(self) -> Iterator[np.ndarray]:
        import sounddevice as sd

        chunk_samples = int(self.chunk_seconds * self._sample_rate)
        buffer = np.empty((0,), dtype=np.float32)

        def callback(indata, frames, time_info, status):
            self._queue.put(indata[:, 0].copy())

        # InputStream records continuously in its own thread, so capture
        # never pauses while a chunk is being transcribed downstream.
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while not self._stop.is_set():
                try:
                    data = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                buffer = np.concatenate([buffer, data])
                while len(buffer) >= chunk_samples:
                    yield buffer[:chunk_samples]
                    buffer = buffer[chunk_samples:]
