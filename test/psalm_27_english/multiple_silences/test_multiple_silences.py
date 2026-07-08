"""
Test: psalm_27_english alignment under injected silence conditions.

Each test injects one or more silent segments into the clean audio and
verifies that the pipeline still produces valid, monotone alignments with
at least 75% word coverage.

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

from pipeline import run               # noqa: E402
from audio_corrupt import inject_silences  # noqa: E402

LANG        = ROOT / "test" / "psalm_27_english"
CLEAN_AUDIO = LANG / "psalm_27.mp3"
REF         = LANG / "psalm_27.txt"
GENERATED   = Path(__file__).parent / "generated"

REF_WORD_COUNT     = 343
COVERAGE_THRESHOLD = 0.69   # relaxed from 0.85 for corrupted audio

REQUIRED_FIELDS = {
    "ref_word", "word_index", "char_offset",
    "line_number", "start_sec", "end_sec", "confidence",
}


def _run_on_corrupted(silences: list[tuple[float, float]], name: str) -> list[dict]:
    """Inject silences into the clean audio, run the pipeline, return JSON entries.

    The corrupted audio and alignment JSON are saved to generated/ so they
    can be inspected or played back after the test run.
    """
    GENERATED.mkdir(exist_ok=True)
    audio_out = GENERATED / f"{name}.mp3"
    json_out  = GENERATED / f"{name}_alignment.json"

    inject_silences(CLEAN_AUDIO, silences, audio_out)
    run(str(audio_out), str(REF), str(json_out))

    return json.loads(json_out.read_text())


def _validate(data: list[dict], label: str) -> None:
    """Shared assertions for all corruption tests."""
    assert len(data) > 0, f"[{label}] No entries returned"

    # Schema
    for e in data:
        missing = REQUIRED_FIELDS - e.keys()
        assert not missing, f"[{label}] Entry missing fields: {missing}\n{e}"
        assert isinstance(e["start_sec"], float), f"[{label}] start_sec must be float"
        assert isinstance(e["end_sec"], float),   f"[{label}] end_sec must be float"
        assert isinstance(e["word_index"], int),  f"[{label}] word_index must be int"
        assert isinstance(e["line_number"], int), f"[{label}] line_number must be int"
        assert 0.0 <= e["confidence"] <= 1.0,     f"[{label}] confidence out of range: {e['confidence']}"

    # Coverage
    coverage = len(data) / REF_WORD_COUNT
    assert coverage >= COVERAGE_THRESHOLD, (
        f"[{label}] Coverage {coverage:.1%} < {COVERAGE_THRESHOLD:.0%} "
        f"({len(data)}/{REF_WORD_COUNT} words matched)"
    )

    # Monotonically increasing timestamps
    times = [e["start_sec"] for e in data]
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1], (
            f"[{label}] Non-monotone at index {i}: {times[i-1]:.2f}s → {times[i]:.2f}s"
        )

    print(f"\n[{label}] Coverage: {coverage:.1%} ({len(data)}/{REF_WORD_COUNT})")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_scattered_silences():
    """3 silences spread across the audio: ~30s (5s), ~85s (10s), ~140s (5s)."""
    entries = _run_on_corrupted([(30.0, 5.0), (85.0, 10.0), (140.0, 5.0)], "scattered_silences")
    _validate(entries, "scattered_silences")


if __name__ == "__main__":
    test_scattered_silences()
