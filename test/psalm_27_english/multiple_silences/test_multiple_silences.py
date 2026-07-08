"""
Test: psalm_27_english alignment under injected silence conditions.

Injects silent segments into the clean audio and verifies:
  1. Schema and coverage
  2. Monotonic timestamps
  3. No words matched during silence windows (no hallucination)
  4. Line tracker resumes on the correct line after each silence

Run with:
    pytest test/psalm_27_english/multiple_silences/ -v
or all tests from root:
    pytest -v
"""
import json
import sys
from pathlib import Path

# allow imports from src/ and test/utils/
ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "test" / "utils"))

from pipeline import run                   # noqa: E402
from audio_corrupt import inject_silences  # noqa: E402

LANG            = ROOT / "test" / "psalm_27_english"
CLEAN_AUDIO     = LANG / "psalm_27.mp3"
CLEAN_ALIGNMENT = LANG / "clean" / "alignment.json"
REF             = LANG / "psalm_27.txt"
GENERATED       = Path(__file__).parent / "generated"

REF_WORD_COUNT     = 343
COVERAGE_THRESHOLD = 0.69

# Silences injected: (start_sec, duration_sec)
# Cumulative time offset after each silence is used to map corrupted → clean timestamps
SILENCES = [(30.0, 5.0), (85.0, 10.0), (140.0, 5.0)]

REQUIRED_FIELDS = {
    "ref_word", "word_index", "char_offset",
    "line_number", "start_sec", "end_sec", "confidence",
}

# How many lines of tolerance when checking line resumption after a silence
LINE_RESUME_TOLERANCE = 2


def _silence_windows(silences: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return (window_start, window_end) for each injected silence in corrupted time.

    audio_corrupt.py replaces sections of the original audio with silence —
    it does not insert (which would shift all subsequent timestamps). So the
    silence positions in the corrupted audio are the same as in the original.
    """
    return [(start, start + duration) for start, duration in silences]


def _line_at_time(alignment: list[dict], time_sec: float) -> int | None:
    """Return line_number of the last matched word at or before time_sec."""
    result = None
    for e in alignment:
        if e["start_sec"] <= time_sec:
            result = e["line_number"]
        else:
            break
    return result


def _run_on_corrupted(silences: list[tuple[float, float]], name: str) -> list[dict]:
    """Inject silences, run pipeline, save outputs to generated/, return JSON entries."""
    GENERATED.mkdir(exist_ok=True)
    audio_out = GENERATED / f"{name}.mp3"
    json_out  = GENERATED / f"{name}_alignment.json"

    inject_silences(CLEAN_AUDIO, silences, audio_out)
    run(str(audio_out), str(REF), str(json_out))

    return json.loads(json_out.read_text())


def _validate_base(data: list[dict], label: str) -> None:
    """Schema, coverage, and monotonic timestamp checks."""
    assert len(data) > 0, f"[{label}] No entries returned"

    for e in data:
        missing = REQUIRED_FIELDS - e.keys()
        assert not missing, f"[{label}] Entry missing fields: {missing}\n{e}"
        assert isinstance(e["start_sec"], float), f"[{label}] start_sec must be float"
        assert isinstance(e["end_sec"], float),   f"[{label}] end_sec must be float"
        assert isinstance(e["word_index"], int),  f"[{label}] word_index must be int"
        assert isinstance(e["line_number"], int), f"[{label}] line_number must be int"
        assert 0.0 <= e["confidence"] <= 1.0,     f"[{label}] confidence out of range: {e['confidence']}"

    coverage = len(data) / REF_WORD_COUNT
    assert coverage >= COVERAGE_THRESHOLD, (
        f"[{label}] Coverage {coverage:.1%} < {COVERAGE_THRESHOLD:.0%} "
        f"({len(data)}/{REF_WORD_COUNT} words matched)"
    )

    times = [e["start_sec"] for e in data]
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1], (
            f"[{label}] Non-monotone at index {i}: {times[i-1]:.2f}s → {times[i]:.2f}s"
        )

    print(f"\n[{label}] Coverage: {coverage:.1%} ({len(data)}/{REF_WORD_COUNT})")


def _validate_silence_windows(data: list[dict], silences: list[tuple[float, float]], label: str) -> None:
    """Assert no words were matched in the core of each injected silence window.

    A 1.0s buffer is applied to both edges:
    - Start: the last word before silence may be timestamped slightly late by Whisper
    - End: Whisper may hallucinate in the final moments of a silence chunk
    Only the inner portion of each window is checked.
    """
    BOUNDARY_BUFFER = 1.0
    windows = _silence_windows(silences)
    for win_start, win_end in windows:
        inner_start = win_start + BOUNDARY_BUFFER
        inner_end   = win_end   - BOUNDARY_BUFFER
        if inner_start >= inner_end:
            continue  # window too short to check after buffering
        inside = [e for e in data if inner_start <= e["start_sec"] <= inner_end]
        assert not inside, (
            f"[{label}] {len(inside)} word(s) hallucinated in silence core "
            f"{inner_start:.1f}s–{inner_end:.1f}s: {[e['ref_word'] for e in inside]}"
        )
    print(f"[{label}] No hallucinations in {len(windows)} silence windows ✓")


def _validate_line_resumption(
    data: list[dict],
    silences: list[tuple[float, float]],
    clean_alignment: list[dict],
    label: str,
) -> None:
    """After each silence, verify the line tracker resumes at the correct line.

    Strategy:
    - Find what line the clean audio was on just before each silence
    - Find the first matched word in the corrupted alignment after each silence ends
    - They should be within LINE_RESUME_TOLERANCE lines of each other
    - Line number must never go backwards after a silence
    """
    windows = _silence_windows(silences)

    for i, (win_start, win_end) in enumerate(windows):
        # What line were we on just before the silence?
        # Since audio_corrupt replaces audio (not inserts), corrupted timestamps = clean timestamps.
        expected_line = _line_at_time(clean_alignment, win_start)
        if expected_line is None:
            continue

        # First word matched after the silence ends in corrupted alignment
        after = [e for e in data if e["start_sec"] > win_end]
        if not after:
            continue
        resume_entry = after[0]
        resume_line  = resume_entry["line_number"]

        # Line must not go backwards
        assert resume_line >= expected_line - LINE_RESUME_TOLERANCE, (
            f"[{label}] Silence {i+1}: line went backwards after gap — "
            f"expected ≥ line {expected_line}, resumed at line {resume_line}"
        )

        # Line must not jump too far forward
        assert resume_line <= expected_line + LINE_RESUME_TOLERANCE, (
            f"[{label}] Silence {i+1}: line jumped too far after gap — "
            f"expected ~line {expected_line}, resumed at line {resume_line}"
        )

        print(
            f"[{label}] Silence {i+1} ({win_start:.0f}s–{win_end:.0f}s): "
            f"expected line ~{expected_line}, resumed at line {resume_line} ✓"
        )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_scattered_silences():
    """3 silences: ~30s (5s), ~85s (10s), ~140s (5s).

    Validates that the pipeline:
    - Maintains schema and coverage
    - Produces monotonic timestamps
    - Does not hallucinate words during silent gaps
    - Resumes line tracking at the correct line after each gap
    """
    assert CLEAN_ALIGNMENT.exists(), (
        f"Clean alignment not found at {CLEAN_ALIGNMENT}. "
        "Run test/psalm_27_english/clean/test_clean.py first."
    )
    clean_alignment = json.loads(CLEAN_ALIGNMENT.read_text())

    data = _run_on_corrupted(SILENCES, "scattered_silences")

    _validate_base(data, "scattered_silences")
    _validate_silence_windows(data, SILENCES, "scattered_silences")
    _validate_line_resumption(data, SILENCES, clean_alignment, "scattered_silences")


if __name__ == "__main__":
    test_scattered_silences()
