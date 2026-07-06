import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

from aligner import AlignedEntry, TextAligner
from audio_source import FileAudioSource
from transcriber import WhisperTranscriber


def _write_json(entries: list[AlignedEntry], output_path: str) -> None:
    data = [asdict(e) for e in entries]
    Path(output_path).write_text(json.dumps(data, indent=2))


def run(
    audio_path: str,
    reference_path: str,
    output_path: str,
    model_size: str = "base",
) -> list[AlignedEntry]:
    """Batch mode: transcribe full file, align, write alignment.json."""
    reference_text = Path(reference_path).read_text()
    transcriber = WhisperTranscriber(model_size=model_size)
    aligner = TextAligner(reference_text)

    for token in transcriber.transcribe_file(audio_path):
        aligner.feed(token)

    entries = aligner.results()
    _write_json(entries, output_path)
    return entries


def run_live(
    audio_path: str,
    reference_path: str,
    model_size: str = "base",
    chunk_seconds: float = 5.0,
) -> None:
    """
    Real-time demo mode.
    - Thread 1: plays audio via afplay
    - Thread 2: feeds chunks through Whisper → aligner
    - Main thread: redraws terminal every 0.5s with current line highlighted
    """
    reference_text = Path(reference_path).read_text()
    lines = reference_text.splitlines()
    total_lines = len(lines)

    transcriber = WhisperTranscriber(model_size=model_size)
    aligner = TextAligner(reference_text)
    source = FileAudioSource(audio_path, chunk_seconds=chunk_seconds)

    done = threading.Event()
    start_time: list[float] = []  # set when audio actually begins

    def play_audio():
        start_time.append(time.time())
        subprocess.run(["afplay", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        done.set()

    def process_chunks():
        for token in transcriber.transcribe_chunks(source):
            aligner.feed(token)
            if done.is_set():
                break

    audio_thread = threading.Thread(target=play_audio, daemon=True)
    chunk_thread = threading.Thread(target=process_chunks, daemon=True)

    audio_thread.start()
    chunk_thread.start()

    _render(lines, current_line=1)

    while not done.is_set():
        time.sleep(0.5)
        if start_time:
            elapsed = time.time() - start_time[0]
            current = _line_at_time(aligner.results(), elapsed) or aligner.current_line()
        else:
            current = 1
        _render(lines, current_line=current)

    # Final render
    _render(lines, current_line=len(lines))
    chunk_thread.join(timeout=5)
    print("\n\n[done]")


def _line_at_time(entries: list[AlignedEntry], elapsed_sec: float) -> int | None:
    """Return the line number of the last aligned word whose start_sec <= elapsed."""
    result = None
    for e in entries:
        if e.start_sec <= elapsed_sec:
            result = e.line_number
        else:
            break
    return result


def _render(lines: list[str], current_line: int) -> None:
    """Clear terminal and redraw all lines, highlighting the current one."""
    sys.stdout.write("\033[2J\033[H")  # clear screen, move cursor to top
    sys.stdout.write(f"  Tracking: line {current_line} of {len(lines)}\n\n")
    for i, line in enumerate(lines, start=1):
        if i == current_line:
            sys.stdout.write(f"\033[1;32m▶ {line}\033[0m\n")  # bold green
        else:
            sys.stdout.write(f"  {line}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Coptic Reader Sync")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("text", help="Path to reference text file")
    parser.add_argument("--mode", choices=["live", "batch"], default="live")
    parser.add_argument("--output", default="alignment.json", help="Output path for batch mode")
    parser.add_argument("--model", default="base")
    args = parser.parse_args()

    if args.mode == "live":
        run_live(args.audio, args.text, model_size=args.model)
    else:
        entries = run(args.audio, args.text, args.output, model_size=args.model)
        print(f"Aligned {len(entries)} words. Written to {args.output}")
