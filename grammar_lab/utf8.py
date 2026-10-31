"""Checking byte-level tokens against a character-level grammar.

Byte-level BPE tokenizers (Qwen's included) can split one multi-byte UTF-8 character across
tokens: "த" might be one token, but "ி" can end up as the bytes ``E0 AE`` in one token and
``BF`` in the next. Decoded on its own, such a token is just "�", which would be wrong to
check against the grammar. So tokens are checked as bytes: complete characters go to the
grammar, and the bytes of an unfinished character wait until the next token completes it.
An unfinished character is allowed only if some character it could still become is allowed.
"""

import codecs
from dataclasses import replace
from typing import Optional

from grammar_lab.gbnf import CharClass
from grammar_lab.recognizer import GrammarPosition, PositionView

MIN_CODE_POINT = {2: 0x80, 3: 0x800, 4: 0x10000}  # smallest character of each UTF-8 length


def split_utf8(data: bytes) -> Optional[tuple[str, bytes]]:
    """The complete characters in ``data`` and the bytes of an unfinished last character,
    or None if ``data`` is not a valid start of UTF-8 text."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        text = decoder.decode(data, final=False)
    except UnicodeDecodeError:
        return None
    pending, _ = decoder.getstate()
    return text, pending


def char_length(pending: bytes) -> int:
    """How many bytes the character that ``pending`` starts has in total."""
    lead = pending[0]
    return 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4


def possible_chars(pending: bytes) -> tuple[int, int]:
    """The range of code points a character starting with ``pending`` can still become."""
    length = char_length(pending)
    code = pending[0] & (0x7F >> length)
    for byte in pending[1:]:
        code = code << 6 | (byte & 0x3F)
    missing_bits = 6 * (length - len(pending))
    low, high = code << missing_bits, (code << missing_bits) | ((1 << missing_bits) - 1)
    return max(low, MIN_CODE_POINT[length]), min(high, 0x10FFFF)


def class_allows_some(cls: CharClass, low: int, high: int) -> bool:
    """Does the character class match at least one code point in [low, high]?"""
    overlapping = sorted((max(ord(a), low), min(ord(b), high)) for a, b in cls.ranges if ord(a) <= high and ord(b) >= low)
    if not cls.negated:
        return bool(overlapping)
    uncovered = low  # the class is negated: look for a code point no range covers
    for a, b in overlapping:
        if a > uncovered:
            return True
        uncovered = max(uncovered, b + 1)
    return uncovered <= high


class Utf8Position:
    """A grammar position plus the bytes of an unfinished character (usually none)."""

    def __init__(self, position: GrammarPosition, pending: bytes = b""):
        self.position = position
        self.pending = pending

    def advance_bytes(self, data: bytes) -> Optional["Utf8Position"]:
        """The position after appending a token's bytes, or None if that breaks the grammar."""
        split = split_utf8(self.pending + data)
        if split is None:
            return None
        text, pending = split
        position = self.position.advance(text)
        if position is None:
            return None
        if pending:
            low, high = possible_chars(pending)
            if not any(class_allows_some(c, low, high) for c in position.next_classes()):
                return None
        return Utf8Position(position, pending)

    @property
    def is_complete(self) -> bool:
        return self.position.is_complete and not self.pending

    @property
    def can_continue(self) -> bool:
        return bool(self.pending) or self.position.can_continue

    def next_classes(self) -> list[CharClass]:
        return self.position.next_classes()

    def describe(self) -> PositionView:
        view = self.position.describe()
        if not self.pending:
            return view
        written = f"{len(self.pending)} of {char_length(self.pending)} bytes written"
        return replace(view, next_chars=[f"the rest of a character ({written})"], can_end=False)
