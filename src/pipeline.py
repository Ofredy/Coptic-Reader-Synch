import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

from aligner import AlignedEntry, TextAligner
from audio_source import AudioSource, FileAudioSource, MicrophoneAudioSource
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


def run_streaming(
    source: AudioSource,
    reference_path: str,
    model_size: str = "base",
    playback_path: str | None = None,
) -> None:
    """
    Streaming mode: feeds an AudioSource through Whisper → aligner in real time.
    - If playback_path is given, plays the file via afplay and stops when it ends.
    - Otherwise runs until Ctrl+C (mic mode).
    """
    reference_text = Path(reference_path).read_text()
    lines = reference_text.splitlines()

    transcriber = WhisperTranscriber(model_size=model_size)
    aligner = TextAligner(reference_text)

    done = threading.Event()
    start_time: list[float] = []

    def process_chunks():
        for token in transcriber.transcribe_chunks(source):
            aligner.feed(token)
            if done.is_set():
                break

    chunk_thread = threading.Thread(target=process_chunks, daemon=True)
    chunk_thread.start()

    if playback_path:
        def play_audio():
            start_time.append(time.time())
            subprocess.run(["afplay", playback_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            done.set()

        audio_thread = threading.Thread(target=play_audio, daemon=True)
        audio_thread.start()

    _render(lines, current_line=1)

    try:
        while not done.is_set():
            time.sleep(0.5)
            if playback_path and start_time:
                elapsed = time.time() - start_time[0]
                current = _line_at_time(aligner.results(), elapsed) or aligner.current_line()
            else:
                current = aligner.current_line()
            _render(lines, current_line=current)
    except KeyboardInterrupt:
        pass
    finally:
        done.set()
        if isinstance(source, MicrophoneAudioSource):
            source.stop()

    _render(lines, current_line=aligner.current_line())
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
    """Clear terminal and redraw a window of lines centered on current_line."""
    sys.stdout.write("\033[2J\033[H")  # clear screen, move cursor to top
    sys.stdout.write(f"  Tracking: line {current_line} of {len(lines)}\n\n")

    term_rows = shutil.get_terminal_size(fallback=(80, 24)).lines
    context = max(3, (term_rows - 6) // 2)  # lines of context above/below current

    start = max(1, current_line - context)
    end = min(len(lines), current_line + context)

    if start > 1:
        sys.stdout.write("  ⋮\n")
    for i in range(start, end + 1):
        line = lines[i - 1]
        if i == current_line:
            sys.stdout.write(f"\033[1;32m▶ {line}\033[0m\n")  # bold green
        else:
            sys.stdout.write(f"  {line}\n")
    if end < len(lines):
        sys.stdout.write("  ⋮\n")
    sys.stdout.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Coptic Reader Sync")
    parser.add_argument("text", help="Path to reference text file")
    parser.add_argument("--mode", choices=["mic", "recorded", "batch"], default="mic")
    parser.add_argument("--audio", help="Path to audio file (required for recorded and batch modes)")
    parser.add_argument("--output", default="alignment.json", help="Output path for batch mode")
    parser.add_argument("--model", default="base")
    args = parser.parse_args()

    if args.mode == "mic":
        run_streaming(MicrophoneAudioSource(), args.text, model_size=args.model)
    elif args.mode == "recorded":
        if not args.audio:
            parser.error("--audio is required for recorded mode")
        run_streaming(FileAudioSource(args.audio), args.text, model_size=args.model, playback_path=args.audio)
    else:
        if not args.audio:
            parser.error("--audio is required for batch mode")
        entries = run(args.audio, args.text, args.output, model_size=args.model)
        print(f"Aligned {len(entries)} words. Written to {args.output}")
