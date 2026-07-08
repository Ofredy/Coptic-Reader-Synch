"""
Test: psalm_27_english alignment (clean audio baseline)

Run with:
    pytest test/psalm_27_english/clean/test_clean.py -v
or:
    python test/psalm_27_english/clean/test_clean.py
"""
import json
import sys
from pathlib import Path

# allow imports from src/
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from pipeline import run  # noqa: E402

HERE    = Path(__file__).parent
LANG    = HERE.parent
AUDIO   = LANG / "psalm_27.mp3"
REF     = LANG / "psalm_27.txt"
OUT     = HERE / "alignment.json"

# Total words in psalm_27.txt
REF_WORD_COUNT = 343

COVERAGE_THRESHOLD = 0.83
TOLERANCE_SEC = 5.0

# Anchor: {label: (word_index, expected_start_sec)}
# word_index is 0-based into the reference word list.
# Estimates — calibrate after first run by checking alignment.json.
ANCHORS = {
    "salvation":  (9,   6.13),
    "strengthen": (334, 172.45),
}

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

    # 3. Coverage
    coverage = len(entries) / REF_WORD_COUNT
    assert coverage >= COVERAGE_THRESHOLD, (
        f"Coverage {coverage:.1%} < {COVERAGE_THRESHOLD:.0%} "
        f"({len(entries)}/{REF_WORD_COUNT} words matched)"
    )

    # 4. Monotonically increasing timestamps
    times = [e["start_sec"] for e in data]
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1], (
            f"Non-monotone at index {i}: {times[i-1]:.2f}s → {times[i]:.2f}s"
        )

    # 5. Anchor accuracy
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
    print(f"  Coverage:  {coverage:.1%} ({len(entries)}/{REF_WORD_COUNT} words)")
    print(f"  Entries:   {len(data)}")
    print(f"  Duration:  {data[-1]['end_sec']:.1f}s (last matched word)")


if __name__ == "__main__":
    test_alignment()
