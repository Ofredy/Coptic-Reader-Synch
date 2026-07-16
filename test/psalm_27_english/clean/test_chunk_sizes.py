"""
Test: does the chunked streaming path (used by --mode mic / --mode recorded)
hold up across different chunk_seconds values?

Runs the same clean psalm_27 audio through the chunked path (FileAudioSource
+ WhisperTranscriber.transcribe_chunks + TextAligner — the same pieces
run_streaming() composes in src/pipeline.py) at a couple chunk sizes, and
checks it against the same schema/coverage/monotonicity gates as the batch
baseline in test_clean.py. Anchors are skipped here since chunk boundaries
shift word timestamps slightly relative to batch mode.

Only the first WINDOW_SECONDS of audio is used per run — Whisper inference
time scales with the number of chunks (audio_duration / chunk_seconds), and
at small chunk sizes that's the dominant cost, not real-time playback
(FileAudioSource has no throttling; it decodes and yields as fast as ffmpeg
delivers). Trimming the input keeps chunk count — and runtime — bounded even
at the small chunk_seconds=0.2 case.

Coverage is measured against how many words the batch (clean/) baseline
found in that same window, not the full 343-word reference count, since a
10s window can never contain most of the reference text.

Run with:
    pytest test/psalm_27_english/clean/test_chunk_sizes.py -v
"""
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

# allow imports from src/
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from aligner import TextAligner        # noqa: E402
from audio_source import FileAudioSource  # noqa: E402
from transcriber import WhisperTranscriber  # noqa: E402

HERE            = Path(__file__).parent
LANG            = HERE.parent
AUDIO           = LANG / "psalm_27.mp3"
REF             = LANG / "psalm_27.txt"
CLEAN_ALIGNMENT = HERE / "alignment.json"
GENERATED       = HERE / "generated"

WINDOW_SECONDS = 10.0

# Chunked mode has less context per Whisper call than batch, so coverage
# is expected to run lower — floor set well below the batch baseline (0.83).
COVERAGE_FLOOR = 0.60

REQUIRED_FIELDS = {
    "ref_word", "word_index", "char_offset",
    "line_number", "start_sec", "end_sec", "confidence",
}


def _trimmed_audio() -> Path:
    """First WINDOW_SECONDS of psalm_27.mp3, cut once and reused across cases."""
    GENERATED.mkdir(exist_ok=True)
    out = GENERATED / f"first_{WINDOW_SECONDS:g}s.mp3"
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(AUDIO), "-t", str(WINDOW_SECONDS), str(out)],
            check=True, capture_output=True,
        )
    return out


def _expected_word_count_in_window() -> int:
    """How many words the batch baseline (near-ground-truth) found in the
    first WINDOW_SECONDS — used as the coverage denominator instead of the
    full reference word count, which a short clip could never approach."""
    assert CLEAN_ALIGNMENT.exists(), (
        f"Clean alignment not found at {CLEAN_ALIGNMENT}. "
        "Run test/psalm_27_english/clean/test_clean.py first."
    )
    data = json.loads(CLEAN_ALIGNMENT.read_text())
    return sum(1 for e in data if e["start_sec"] <= WINDOW_SECONDS)


def _run_chunked_and_validate(chunk_seconds: float) -> float:
    audio_path = _trimmed_audio()
    expected_words = _expected_word_count_in_window()

    reference_text = REF.read_text()
    transcriber = WhisperTranscriber()
    aligner = TextAligner(reference_text)
    source = FileAudioSource(str(audio_path), chunk_seconds=chunk_seconds)

    for token in transcriber.transcribe_chunks(source):
        aligner.feed(token)

    entries = aligner.results()
    data = [asdict(e) for e in entries]

    assert len(data) > 0, f"chunk_seconds={chunk_seconds}: no entries"

    for e in data:
        missing = REQUIRED_FIELDS - e.keys()
        assert not missing, f"chunk_seconds={chunk_seconds}: entry missing fields {missing}\n{e}"
        assert isinstance(e["start_sec"], float)
        assert isinstance(e["end_sec"], float)
        assert isinstance(e["word_index"], int)
        assert isinstance(e["line_number"], int)
        assert 0.0 <= e["confidence"] <= 1.0

    coverage = len(entries) / expected_words
    assert coverage >= COVERAGE_FLOOR, (
        f"chunk_seconds={chunk_seconds}: coverage {coverage:.1%} < floor {COVERAGE_FLOOR:.0%} "
        f"({len(entries)}/{expected_words} words matched in first {WINDOW_SECONDS:g}s)"
    )

    times = [e["start_sec"] for e in data]
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1], (
            f"chunk_seconds={chunk_seconds}: non-monotone at index {i}: "
            f"{times[i-1]:.2f}s -> {times[i]:.2f}s"
        )

    print(f"\nchunk_seconds={chunk_seconds}: coverage {coverage:.1%} ({len(entries)}/{expected_words})")
    return coverage


@pytest.mark.parametrize(
    "chunk_seconds",
    [
        pytest.param(
            0.2,
            marks=pytest.mark.xfail(
                reason=(
                    "0.2s gives Whisper less audio per call than a typical spoken "
                    f"word takes to say, so coverage should collapse below the "
                    f"{COVERAGE_FLOOR:.0%} floor. strict=True: if this unexpectedly "
                    "passes, that's itself a failure — it'd mean the coverage floor "
                    "isn't actually sensitive to chunk size."
                ),
                strict=True,
            ),
        ),
        1.0,
    ],
)
def test_chunked_alignment(chunk_seconds):
    _run_chunked_and_validate(chunk_seconds)
