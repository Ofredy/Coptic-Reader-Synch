import re
import threading
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

from transcriber import WordToken

MAX_WORDS_PER_SEC   = 4.0   # ~240 wpm — upper bound for realistic speech
MAX_CONSECUTIVE_MISSES = 5  # misses in a row before declaring lost
RECOVERY_WINDOW     = 50    # wider search window when lost


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

        # Re-sync state
        self._last_good_cursor = 0       # cursor position of last confident match
        self._last_good_time: Optional[float] = None  # timestamp of last confident match
        self._consecutive_misses = 0
        self._lost = False

    def feed(self, token: WordToken) -> Optional[AlignedEntry]:
        clean_token = re.sub(r"[^\w']", "", token.word).lower()
        if not clean_token:
            return None

        # In recovery mode search a wider window from the last known good position,
        # not from wherever the cursor drifted to.
        if self._lost:
            search_start = self._last_good_cursor
            search_end   = search_start + RECOVERY_WINDOW
        else:
            search_start = self._cursor
            search_end   = search_start + self._window

        candidates = self._ref_words[search_start:search_end]
        if not candidates:
            return None

        best_score = 0.0
        best_idx   = None
        for i, ref in enumerate(candidates):
            score = fuzz.ratio(clean_token, ref.clean)
            if score > best_score:
                best_score = score
                best_idx   = i

        # --- Low confidence: token didn't match anything useful ---
        if best_score < self._threshold:
            self._consecutive_misses += 1
            if self._consecutive_misses >= MAX_CONSECUTIVE_MISSES:
                self._lost = True
            return None

        # --- Temporal sanity check (normal mode only) ---
        # Reject the match if it would advance the cursor faster than speech allows.
        if not self._lost and self._last_good_time is not None:
            elapsed = token.start_sec - self._last_good_time
            if elapsed > 0:
                words_advanced = best_idx + 1
                if words_advanced > elapsed * MAX_WORDS_PER_SEC:
                    self._consecutive_misses += 1
                    if self._consecutive_misses >= MAX_CONSECUTIVE_MISSES:
                        self._lost = True
                    return None

        ref = candidates[best_idx]
        new_cursor = search_start + best_idx + 1

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
            self._cursor           = new_cursor
            self._last_good_cursor = new_cursor
            self._last_good_time   = token.start_sec
            self._consecutive_misses = 0
            self._lost             = False
            self._results.append(entry)
            self._current_line     = ref.line_number

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
