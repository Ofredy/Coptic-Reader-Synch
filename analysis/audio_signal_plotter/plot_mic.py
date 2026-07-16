"""
Terminal-driven mic recorder — quick diagnostic tool, not part of the src/ pipeline.

Press Enter to start recording, say your word/phrase, press Enter again to stop.
Saves the full take as a .wav and a waveform .png, useful for comparing different
deliveries of the same word (e.g. spoken vs. chanted) by running this once per take.

Run:
    python analysis/audio_signal_plotter/plot_mic.py

Each run writes one .wav (raw audio) and one .png (waveform plot) to
analysis/audio_signal_plotter/recordings/, both timestamped together.
"""
import sys
import time
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
OUT_DIR = Path(__file__).parent / "recordings"

_recorded: list[np.ndarray] = []


def _audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    _recorded.append(indata[:, 0].copy())


def _save_full_history() -> None:
    if not _recorded:
        print("Nothing recorded — nothing to save.")
        return

    audio = np.concatenate(_recorded)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wav_path = OUT_DIR / f"mic_{timestamp}.wav"
    with wave.open(str(wav_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm16.tobytes())
    print(f"Saved audio to {wav_path}")

    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(len(audio)) / SAMPLE_RATE
    ax.plot(t, audio, linewidth=0.5)
    ax.set_xlabel("seconds")
    ax.set_ylabel("amplitude")
    ax.set_title(f"Full session — {len(audio) / SAMPLE_RATE:.1f}s")
    png_path = OUT_DIR / f"mic_{timestamp}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {png_path}")


def main() -> None:
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=_audio_callback)
    stream.start()
    input("Say something, then press Enter when done...")
    stream.stop()
    stream.close()

    _save_full_history()
    print("Done.")


if __name__ == "__main__":
    main()
