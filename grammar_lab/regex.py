"""Convert a regex (a supported subset) into GBNF text the learner can read.

A regex constraint is a grammar: the whole output must match the regex, just as it must
match a grammar's root rule. The converted grammar has a single rule, ``root``.

Supported: literal characters and escaped punctuation, ``.``, character classes with
ranges and negation, ``\\d \\w \\s`` (and ``\\D \\W \\S``), ``\\n \\t \\r \\xhh \\uhhhh``,
groups ``( )``, ``(?: )`` and named groups, alternation, ``* + ?`` (lazy forms behave the
same, since only *which* strings match matters) and counted repeats ``{m}``, ``{m,}``,
``{m,n}``, which are written out as copies because the GBNF subset has no ``{m,n}``.

Rejected with a located error: anchors (``^ $ \\b \\B \\A \\Z``), lookarounds,
backreferences, possessive quantifiers and inline flags.
"""

from dataclasses import dataclass
from typing import Optional, Union

from grammar_lab.gbnf import GrammarError

ROOT = "root"


class RegexError(GrammarError):
    """A regex that cannot be converted. Regexes are one line, so only the column matters."""

    def __init__(self, message: str, column: int):
        super().__init__(message, 1, column)
        self.args = (f"Column {column}: {message}",)

    def __str__(self) -> str:
        return self.args[0]


# ---- regex syntax tree ------------------------------------------------------------------

@dataclass(frozen=True)
class Char:
    char: str


@dataclass(frozen=True)
class Class:
    ranges: tuple[tuple[str, str], ...]
    negated: bool = False


@dataclass(frozen=True)
class Seq:
    items: tuple["Node", ...]


@dataclass(frozen=True)
class Alt:
    options: tuple["Node", ...]


@dataclass(frozen=True)
class Rep:
    item: "Node"
    min: int
    max: Optional[int]


Node = Union[Char, Class, Seq, Alt, Rep]

DIGIT = (("0", "9"),)
WORD = (("a", "z"), ("A", "Z"), ("0", "9"), ("_", "_"))
SPACE = ((" ", " "), ("\t", "\t"), ("\n", "\n"), ("\r", "\r"))
SHORTHANDS = {"d": DIGIT, "w": WORD, "s": SPACE}
CHAR_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "0": "\0"}
ANCHOR_ESCAPES = "bBAZzG"


def regex_to_gbnf(regex: str) -> str:
    """GBNF text equivalent to ``regex`` (whole-output match), or raise RegexError."""
    tree = _RegexParser(regex).parse()
    body = " | ".join(_render_seq(o) for o in tree.options) if isinstance(tree, Alt) else _render_seq(tree)
    return f"{ROOT} ::= {body}\n"


# ---- parser -----------------------------------------------------------------------------

class _RegexParser:
    def __init__(self, text: str):
        self.text, self.pos = text, 0

    def error(self, message: str, pos: Optional[int] = None) -> RegexError:
        return RegexError(message, (self.pos if pos is None else pos) + 1)

    def peek(self, ahead: int = 0) -> str:
        i = self.pos + ahead
        return self.text[i] if i < len(self.text) else ""

    def parse(self) -> Node:
        node = self.alternation()
        if self.pos < len(self.text):  # only an unmatched ")" stops alternation early
            raise self.error("this `)` has no matching `(`")
        return node

    def alternation(self) -> Node:
        options = [self.sequence()]
        while self.peek() == "|":
            self.pos += 1
            options.append(self.sequence())
        return options[0] if len(options) == 1 else Alt(tuple(options))

    def sequence(self) -> Node:
        items = []
        while self.peek() not in ("", "|", ")"):
            items.append(self.quantified())
        return items[0] if len(items) == 1 else Seq(tuple(items))

    def quantified(self) -> Node:
        item = self.atom()
        c = self.peek()
        if c and c in "*+?":
            bounds = {"*": (0, None), "+": (1, None), "?": (0, 1)}[c]
            self.pos += 1
        elif c == "{" and self.counted_repeat_ahead():
            bounds = self.counted_repeat()
        else:
            return item
        if self.peek() == "?":  # lazy: matches the same strings
            self.pos += 1
        elif self.peek() == "+":
            raise self.error("possessive quantifiers like `*+` are not supported")
        if self.peek() and (self.peek() in "*+?" or (self.peek() == "{" and self.counted_repeat_ahead())):
            raise self.error("a quantifier cannot directly follow another quantifier")
        return Rep(item, *bounds)

    def counted_repeat_ahead(self) -> bool:
        end = self.text.find("}", self.pos)
        inside = self.text[self.pos + 1:end] if end > 0 else ""
        parts = inside.split(",")
        return (end > 0 and len(parts) <= 2 and parts[0].isdigit()
                and (len(parts) == 1 or parts[1] == "" or parts[1].isdigit()))

    def counted_repeat(self) -> tuple[int, Optional[int]]:
        start = self.pos
        end = self.text.index("}", self.pos)
        parts = self.text[self.pos + 1:end].split(",")
        self.pos = end + 1
        low = int(parts[0])
        high = low if len(parts) == 1 else (None if parts[1] == "" else int(parts[1]))
        if high is not None and high < low:
            raise self.error(f"`{{{low},{high}}}`: the maximum is smaller than the minimum", start)
        return low, high

    def atom(self) -> Node:
        c = self.peek()
        start = self.pos
        if c == "(":
            return self.group()
        if c == "[":
            return self.char_class()
        if c in "^$":
            raise self.error(f"anchors like `{c}` are not supported: the constraint always matches "
                             "the whole output, so they are not needed")
        if c in "*+?":
            raise self.error(f"`{c}` has nothing to repeat")
        if c == "{" and self.counted_repeat_ahead():
            raise self.error("`{` has nothing to repeat")
        self.pos += 1
        if c == ".":
            return Class((("\n", "\n"),), negated=True)
        if c == "\\":
            return self.escape(start, in_class=False)
        return Char(c)

    def group(self) -> Node:
        start = self.pos
        self.pos += 1
        if self.peek() == "?":
            ahead = self.text[self.pos:self.pos + 4]
            if ahead.startswith(("?=", "?!", "?<=", "?<!")):
                raise self.error("lookarounds like `(?=...)` and `(?<!...)` are not supported", start)
            if ahead.startswith("?P="):
                raise self.error("backreferences like `(?P=name)` are not supported", start)
            if ahead.startswith("?:"):
                self.pos += 2
            elif ahead.startswith(("?P<", "?<")):
                close = self.text.find(">", self.pos)
                if close < 0:
                    raise self.error("this group name is never closed with `>`", start)
                self.pos = close + 1
            else:
                raise self.error("inline flags and other `(?...)` groups are not supported", start)
        inner = self.alternation()
        if self.peek() != ")":
            raise self.error("this group is never closed with `)`", start)
        self.pos += 1
        return inner

    def escape(self, start: int, in_class: bool) -> Node:
        """The escape after a backslash (already consumed) at ``start``."""
        c = self.peek()
        if not c:
            raise self.error("the regex ends with a lone `\\`", start)
        self.pos += 1
        if c in SHORTHANDS:
            return Class(SHORTHANDS[c])
        if c.lower() in SHORTHANDS:
            if in_class:
                raise self.error(f"`\\{c}` inside a character class is not supported", start)
            return Class(SHORTHANDS[c.lower()], negated=True)
        if c in CHAR_ESCAPES:
            return Char(CHAR_ESCAPES[c])
        if c in "xu":
            digits = 2 if c == "x" else 4
            code = self.text[self.pos:self.pos + digits]
            if len(code) != digits or any(d not in "0123456789abcdefABCDEF" for d in code):
                raise self.error(f"`\\{c}` needs {digits} hex digits", start)
            self.pos += digits
            return Char(chr(int(code, 16)))
        if c.isdigit() or c == "k":
            raise self.error(f"backreferences like `\\{c}` are not supported", start)
        if c in ANCHOR_ESCAPES and not in_class:
            raise self.error(f"anchors like `\\{c}` are not supported: the constraint always matches "
                             "the whole output, so they are not needed", start)
        if c.isalnum():
            raise self.error(f"unknown escape `\\{c}`", start)
        return Char(c)

    def char_class(self) -> Class:
        start = self.pos
        self.pos += 1
        negated = self.peek() == "^"
        if negated:
            self.pos += 1
        ranges: list[tuple[str, str]] = []
        first = True
        while True:
            c = self.peek()
            if not c:
                raise self.error("this character class is never closed with `]`", start)
            if c == "]" and not first:
                self.pos += 1
                return Class(tuple(ranges), negated)
            first = False
            low = self.class_char()
            if isinstance(low, tuple):  # a shorthand like \d
                ranges.extend(low)
                continue
            high = low
            if self.peek() == "-" and self.peek(1) not in ("]", ""):
                self.pos += 1
                high = self.class_char()
                if isinstance(high, tuple):
                    raise self.error("a range cannot end with a shorthand like `\\d`")
                if high < low:
                    raise self.error(f"the range `{low}-{high}` is backwards")
            ranges.append((low, high))

    def class_char(self) -> Union[str, tuple[tuple[str, str], ...]]:
        start = self.pos
        c = self.peek()
        self.pos += 1
        if c != "\\":
            return c
        node = self.escape(start, in_class=True)
        if isinstance(node, Class):
            return node.ranges
        return node.char


# ---- rendering as GBNF --------------------------------------------------------------------

def _render_seq(node: Node) -> str:
    """A node as a sequence of GBNF items, joining adjacent characters into one literal."""
    items = node.items if isinstance(node, Seq) else (node,)
    parts: list[str] = []
    run: list[str] = []
    for item in items:
        if isinstance(item, Char):
            run.append(item.char)
            continue
        if run:
            parts.append(_literal("".join(run)))
            run = []
        parts.append(_render_item(item))
    if run:
        parts.append(_literal("".join(run)))
    return " ".join(parts) if parts else '""'


def _render_item(node: Node) -> str:
    if isinstance(node, Char):
        return _literal(node.char)
    if isinstance(node, Class):
        return _char_class(node)
    if isinstance(node, Alt):
        return "(" + " | ".join(_render_seq(o) for o in node.options) + ")"
    if isinstance(node, Seq):
        return f"({_render_seq(node)})"
    return _render_repeat(node)


def _render_repeat(node: Rep) -> str:
    atom = _render_item(node.item)
    if isinstance(node.item, Rep):
        atom = f"({atom})"
    simple = {(0, None): "*", (1, None): "+", (0, 1): "?"}.get((node.min, node.max))
    if simple:
        return atom + simple
    copies = [atom] * node.min
    copies += [atom + "*"] if node.max is None else [atom + "?"] * (node.max - node.min)
    return " ".join(copies) if copies else '""'


LITERAL_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r"}
CLASS_ESCAPES = {"\\": "\\\\", "]": "\\]", "[": "\\[", "-": "\\-", "^": "\\^",
                 "\n": "\\n", "\t": "\\t", "\r": "\\r"}


def _literal(text: str) -> str:
    return '"' + "".join(LITERAL_ESCAPES.get(c, c) for c in text) + '"'


def _char_class(node: Class) -> str:
    def char(c: str) -> str:
        return CLASS_ESCAPES.get(c, c)

    body = "".join(char(lo) if lo == hi else f"{char(lo)}-{char(hi)}" for lo, hi in node.ranges)
    return f"[{'^' if node.negated else ''}{body}]"
