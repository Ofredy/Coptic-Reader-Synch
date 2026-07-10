"""
Test: psalm_27_english alignment against a SUNG/CHANTED rendition (Coptic-style).

Same reference text as the clean speech baseline (test/psalm_27_english/clean),
but the audio here is chanted rather than spoken. This is our first regression
test for melismatic/sung audio.

Word-level coverage is NOT the right pass/fail signal here: melismatic
singing stretches a single word over many seconds, so Whisper reliably
matches fewer individual words than it does on clean speech (measured
42% here vs. 83% on the clean baseline). What actually matters for this
app is whether the LINE tracker stays on the right line — and it only
takes one matched word per line to do that. Measured line coverage on
this clip is 27/28 (96.4%), matching what was observed watching the
live `--mode recorded` run track correctly. So line coverage is the
primary gate; word coverage is kept as a low sanity floor only, to catch
a fully broken pipeline rather than to demand speech-level precision.

Run with:
    pytest test/psalm_27_english/coptic/test_coptic_psalm_27.py -v
or:
    python test/psalm_27_english/coptic/test_coptic_psalm_27.py
"""
import json
import sys
from pathlib import Path

# allow imports from src/
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from pipeline import run  # noqa: E402

HERE  = Path(__file__).parent
LANG  = HERE.parent
AUDIO = HERE / "psalm.mp3"
REF   = LANG / "psalm_27.txt"
OUT   = HERE / "alignment.json"

# Total words / lines in psalm_27.txt (same reference as the clean/ baseline)
REF_WORD_COUNT = 343
REF_LINE_COUNT = len(REF.read_text().splitlines())

# Primary gate: line coverage. Measured 27/28 (96.4%) on this clip —
# threshold set with a little slack below that.
LINE_COVERAGE_THRESHOLD = 0.90

# Sanity floor only, not a real target. Word-level coverage on sung audio
# is expected to run well below the clean-speech baseline (0.83) because
# melismas stretch single words over many seconds. Measured 42% here —
# this floor just catches a fully broken pipeline (e.g. near-zero matches).
WORD_COVERAGE_FLOOR = 0.30
TOLERANCE_SEC = 8.0

# Anchor: {label: (word_index, expected_start_sec)}
# word_index is 0-based into the reference word list.
# Empty until calibrated — after the first run, pick a few well-separated
# words from alignment.json (e.g. one early, one mid, one late) and record
# their actual start_sec here so future runs are checked against them.
ANCHORS: dict[str, tuple[int, float]] = {}

REQUIRED_FIELDS = {
    "ref_word", "word_index", "char_offset",
    "line_number", "start_sec", "end_sec", "confidence",
}


def test_alignment():
    entries = run(str(AUDIO), str(REF), str(OUT))

    # 1. Output file exists and is valid JSON
    assert OUT.exists(), "alignment.json was not written"
    data = json.loads(OUT.read_text())
    assert isinstance(data, list) and len(data) > 0, "alignment.json is empty"

    # 2. Schema — every entry has required fields with correct types
    for e in data:
        missing = REQUIRED_FIELDS - e.keys()
        assert not missing, f"Entry missing fields: {missing}\n{e}"
        assert isinstance(e["start_sec"], float), "start_sec must be float"
        assert isinstance(e["end_sec"], float), "end_sec must be float"
        assert isinstance(e["word_index"], int), "word_index must be int"
        assert isinstance(e["line_number"], int), "line_number must be int"
        assert 0.0 <= e["confidence"] <= 1.0, f"confidence out of range: {e['confidence']}"

    # 3a. Word coverage — sanity floor only (see module docstring for why
    # this isn't the primary signal for sung/melismatic audio).
    word_coverage = len(entries) / REF_WORD_COUNT
    assert word_coverage >= WORD_COVERAGE_FLOOR, (
        f"Word coverage {word_coverage:.1%} < floor {WORD_COVERAGE_FLOOR:.0%} "
        f"({len(entries)}/{REF_WORD_COUNT} words matched) — pipeline looks broken, "
        f"not just lossy on melisma"
    )

    # 3b. Line coverage — the real pass/fail signal. Only one matched word
    # per line is needed to keep the line tracker on the right line.
    lines_hit = {e["line_number"] for e in data}
    line_coverage = len(lines_hit) / REF_LINE_COUNT
    missing_lines = sorted(set(range(1, REF_LINE_COUNT + 1)) - lines_hit)
    assert line_coverage >= LINE_COVERAGE_THRESHOLD, (
        f"Line coverage {line_coverage:.1%} < {LINE_COVERAGE_THRESHOLD:.0%} "
        f"({len(lines_hit)}/{REF_LINE_COUNT} lines matched) — missing lines: {missing_lines}"
    )

    # 4. Monotonically increasing timestamps
    times = [e["start_sec"] for e in data]
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1], (
            f"Non-monotone at index {i}: {times[i-1]:.2f}s → {times[i]:.2f}s"
        )

    # 5. Anchor accuracy (skipped until ANCHORS is populated post-calibration)
    by_index = {e["word_index"]: e for e in data}
    for label, (idx, expected_sec) in ANCHORS.items():
        assert idx in by_index, (
            f"Anchor '{label}' (word_index={idx}) not found in alignment. "
            f"Check ANCHORS after first run."
        )
        actual = by_index[idx]["start_sec"]
        diff = abs(actual - expected_sec)
        assert diff <= TOLERANCE_SEC, (
            f"Anchor '{label}': expected ~{expected_sec}s, got {actual:.2f}s "
            f"(diff={diff:.1f}s > tolerance={TOLERANCE_SEC}s)"
        )

    print(f"\nAll assertions passed.")
    print(f"  Line coverage:  {line_coverage:.1%} ({len(lines_hit)}/{REF_LINE_COUNT} lines)")
    print(f"  Word coverage:  {word_coverage:.1%} ({len(entries)}/{REF_WORD_COUNT} words)")
    print(f"  Entries:        {len(data)}")
    print(f"  Duration:       {data[-1]['end_sec']:.1f}s (last matched word)")


if __name__ == "__main__":
    test_alignment()
