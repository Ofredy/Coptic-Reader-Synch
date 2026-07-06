import re
import threading
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

from transcriber import WordToken


@dataclass
class RefWord:
    word: str           # original text including punctuation
    clean: str          # lowercased, punctuation stripped — used for matching
    word_index: int     # 0-based index in reference word list
    char_offset: int    # character offset of word start in raw reference text
    line_number: int    # 1-based line number


@dataclass
class AlignedEntry:
    ref_word: str
    word_index: int
    char_offset: int
    line_number: int
    start_sec: float
    end_sec: float
    confidence: float   # rapidfuzz score normalized to [0.0, 1.0]


def _parse_reference(text: str) -> list[RefWord]:
    ref_words = []
    word_index = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        # find each whitespace-separated token and its offset within the full text
        line_start = _line_start_offset(text, line_number)
        for match in re.finditer(r'\S+', line):
            word = match.group()
            char_offset = line_start + match.start()
            clean = re.sub(r"[^\w']", "", word).lower()
            ref_words.append(RefWord(
                word=word,
                clean=clean,
                word_index=word_index,
                char_offset=char_offset,
                line_number=line_number,
            ))
            word_index += 1
    return ref_words


def _line_start_offset(text: str, line_number: int) -> int:
    offset = 0
    for i, line in enumerate(text.splitlines(keepends=True), start=1):
        if i == line_number:
            return offset
        offset += len(line)
    return offset


class TextAligner:
    def __init__(self, reference_text: str, window: int = 10, threshold: float = 80.0):
        self._ref_words = _parse_reference(reference_text)
        self._window = window
        self._threshold = threshold
        self._cursor = 0
        self._results: list[AlignedEntry] = []
        self._current_line = 1
        self._lock = threading.Lock()

    def feed(self, token: WordToken) -> Optional[AlignedEntry]:
        clean_token = re.sub(r"[^\w']", "", token.word).lower()
        if not clean_token:
            return None

        candidates = self._ref_words[self._cursor: self._cursor + self._window]
        if not candidates:
            return None

        best_score = 0.0
        best_idx = None
        for i, ref in enumerate(candidates):
            score = fuzz.ratio(clean_token, ref.clean)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score < self._threshold:
            return None

        ref = candidates[best_idx]
        entry = AlignedEntry(
            ref_word=ref.word,
            word_index=ref.word_index,
            char_offset=ref.char_offset,
            line_number=ref.line_number,
            start_sec=token.start_sec,
            end_sec=token.end_sec,
            confidence=round(best_score / 100.0, 4),
        )

        with self._lock:
            self._cursor += best_idx + 1
            self._results.append(entry)
            self._current_line = ref.line_number

        return entry

    def current_line(self) -> int:
        with self._lock:
            return self._current_line

    def results(self) -> list[AlignedEntry]:
        with self._lock:
            return list(self._results)

    @property
    def total_ref_words(self) -> int:
        return len(self._ref_words)
