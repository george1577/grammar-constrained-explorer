"""Parse GBNF-subset text into a Grammar (rules of plain expression objects).

Supported: quoted literals with escapes, character classes with ranges and negation,
rule references, sequences, groups ``( )``, alternation, ``* + ?`` and ``#`` comments.
Anything else (including counted repeats ``{m,n}``) is reported as a GrammarError with
its line and column.
"""

from dataclasses import dataclass
from typing import Optional, Union


class GrammarError(ValueError):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"Line {line}, column {column}: {message}")
        self.message, self.line, self.column = message, line, column


@dataclass(frozen=True)
class Literal:
    text: str


@dataclass(frozen=True)
class CharClass:
    ranges: tuple[tuple[str, str], ...]  # inclusive (low, high) pairs
    negated: bool = False

    def matches(self, char: str) -> bool:
        inside = any(low <= char <= high for low, high in self.ranges)
        return inside != self.negated


@dataclass(frozen=True)
class RuleRef:
    name: str
    line: int = 0
    column: int = 0


@dataclass(frozen=True)
class Seq:
    items: tuple["Expr", ...]


@dataclass(frozen=True)
class Alt:
    options: tuple["Expr", ...]


@dataclass(frozen=True)
class Repeat:
    item: "Expr"
    min: int
    max: Optional[int]  # None = unbounded


Expr = Union[Literal, CharClass, RuleRef, Seq, Alt, Repeat]


@dataclass(frozen=True)
class Grammar:
    rules: dict[str, Expr]  # in the order they were written


ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "[": "[", "]": "]", "-": "-", "^": "^"}


REPEATS = {"*": (0, None), "+": (1, None), "?": (0, 1)}


def parse_gbnf(text: str) -> Grammar:
    return _Parser(text).parse()


class _Parser:
    def __init__(self, text: str):
        self.text, self.pos = text, 0

    # ---- helpers ----------------------------------------------------------------

    def where(self, pos: Optional[int] = None) -> tuple[int, int]:
        pos = self.pos if pos is None else pos
        line = self.text.count("\n", 0, pos) + 1
        column = pos - (self.text.rfind("\n", 0, pos) + 1) + 1
        return line, column

    def error(self, message: str, pos: Optional[int] = None) -> GrammarError:
        return GrammarError(message, *self.where(pos))

    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def skip_space(self):
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c.isspace():
                self.pos += 1
            elif c == "#":
                end = self.text.find("\n", self.pos)
                self.pos = len(self.text) if end < 0 else end
            else:
                break

    def name_at(self, pos: int) -> str:
        end = pos
        while end < len(self.text) and (self.text[end].isalnum() or self.text[end] in "-_"):
            end += 1
        return self.text[pos:end]

    def at_rule_start(self) -> bool:
        """Is the next thing `name ::=`, i.e. the start of the next rule?"""
        name = self.name_at(self.pos)
        if not name:
            return False
        rest = self.pos + len(name)
        while rest < len(self.text) and self.text[rest] in " \t":
            rest += 1
        return self.text.startswith("::=", rest)

    # ---- grammar ----------------------------------------------------------------

    def parse(self) -> Grammar:
        rules: dict[str, Expr] = {}
        refs: list[RuleRef] = []
        self.refs = refs
        self.rule_starts: dict[str, tuple[int, int]] = {}  # rule name -> (line, column)
        self.skip_space()
        while self.pos < len(self.text):
            start = self.pos
            name = self.name_at(self.pos)
            if not name or not self.at_rule_start():
                raise self.error("expected a rule like `name ::= ...`")
            if name in rules:
                raise self.error(f"rule `{name}` is defined twice", start)
            self.rule_starts[name] = self.where(start)
            self.pos += len(name)
            self.skip_space()
            self.pos += len("::=")
            rules[name] = self.alternation()
            self.skip_space()
        if not rules:
            raise GrammarError("the grammar has no rules", 1, 1)
        for ref in refs:
            if ref.name not in rules:
                raise GrammarError(f"rule `{ref.name}` is used but never defined", ref.line, ref.column)
        _reject_left_recursion(rules, self.rule_starts)
        return Grammar(rules)

    def alternation(self) -> Expr:
        options = [self.sequence()]
        while self.peek() == "|":
            self.pos += 1
            options.append(self.sequence())
        return options[0] if len(options) == 1 else Alt(tuple(options))

    def sequence(self) -> Expr:
        items = []
        while True:
            self.skip_space()
            c = self.peek()
            if c in ("", "|", ")") or self.at_rule_start():
                break
            items.append(self.postfix())
        return items[0] if len(items) == 1 else Seq(tuple(items))

    def postfix(self) -> Expr:
        item = self.primary()
        c = self.peek()
        if c in REPEATS:
            self.pos += 1
            return Repeat(item, *REPEATS[c])
        if c == "{":
            raise self.error("counted repeats like `{m,n}` are not supported by this app; "
                             "write the item out several times, using `?` for optional copies")
        return item

    def primary(self) -> Expr:
        c = self.peek()
        if c == "(":
            start = self.pos
            self.pos += 1
            inner = self.alternation()
            self.skip_space()
            if self.peek() != ")":
                raise self.error("this group is never closed with `)`", start)
            self.pos += 1
            return inner
        if c == '"':
            return self.literal()
        if c == "[":
            return self.char_class()
        name = self.name_at(self.pos)
        if name:
            line, column = self.where()
            self.pos += len(name)
            ref = RuleRef(name, line, column)
            self.refs.append(ref)
            return ref
        raise self.error(f"unexpected `{c}`")

    def literal(self) -> Literal:
        start = self.pos
        self.pos += 1
        chars = []
        while True:
            c = self.peek()
            if c in ("", "\n"):
                raise self.error("this quoted literal is never closed", start)
            self.pos += 1
            if c == '"':
                return Literal("".join(chars))
            chars.append(self.escape() if c == "\\" else c)

    def char_class(self) -> CharClass:
        start = self.pos
        self.pos += 1
        negated = self.peek() == "^"
        if negated:
            self.pos += 1
        ranges = []
        while True:
            c = self.peek()
            if c in ("", "\n"):
                raise self.error("this character class is never closed with `]`", start)
            self.pos += 1
            if c == "]":
                return CharClass(tuple(ranges), negated)
            low = self.escape() if c == "\\" else c
            high = low
            if self.peek() == "-" and self.text[self.pos + 1:self.pos + 2] not in ("]", ""):
                self.pos += 1
                c = self.peek()
                self.pos += 1
                high = self.escape() if c == "\\" else c
            ranges.append((low, high))

    def escape(self) -> str:
        """Decode the escape after a backslash (the backslash is already consumed)."""
        c = self.peek()
        if c in ESCAPES:
            self.pos += 1
            return ESCAPES[c]
        raise self.error(f"unknown escape `\\{c}`", self.pos - 1)


# ---- left recursion ---------------------------------------------------------------
# A rule is left-recursive if it can start with itself, directly (`expr ::= expr "+" term`)
# or through other rules and empty-matching prefixes (`a ::= ws b`, `b ::= a "x"`). The
# recognizer would expand such a rule forever, so it is rejected up front.

LEFT_RECURSION_HINT = ('rewrite it with a repeat instead, e.g. `expr ::= term ("+" term)*` '
                       'rather than `expr ::= expr "+" term | term`')


def _nullable_rules(rules: dict[str, Expr]) -> set[str]:
    nullable: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, body in rules.items():
            if name not in nullable and _can_be_empty(body, nullable):
                nullable.add(name)
                changed = True
    return nullable


def _can_be_empty(expr: Expr, nullable: set[str]) -> bool:
    if isinstance(expr, Literal):
        return expr.text == ""
    if isinstance(expr, CharClass):
        return False
    if isinstance(expr, RuleRef):
        return expr.name in nullable
    if isinstance(expr, Seq):
        return all(_can_be_empty(item, nullable) for item in expr.items)
    if isinstance(expr, Alt):
        return any(_can_be_empty(option, nullable) for option in expr.options)
    return expr.min == 0 or _can_be_empty(expr.item, nullable)  # Repeat


def _first_refs(expr: Expr, nullable: set[str]) -> set[str]:
    """The rules that may be expanded before any character of ``expr`` is matched."""
    if isinstance(expr, RuleRef):
        return {expr.name}
    if isinstance(expr, Seq):
        out: set[str] = set()
        for item in expr.items:
            out |= _first_refs(item, nullable)
            if not _can_be_empty(item, nullable):
                break
        return out
    if isinstance(expr, Alt):
        return set().union(*(_first_refs(option, nullable) for option in expr.options))
    if isinstance(expr, Repeat):
        return _first_refs(expr.item, nullable)
    return set()


def _reject_left_recursion(rules: dict[str, Expr], starts: dict[str, tuple[int, int]]):
    nullable = _nullable_rules(rules)
    first = {name: _first_refs(body, nullable) for name, body in rules.items()}
    for name in rules:  # report the first rule, in written order, that can start with itself
        path = _path_back_to(name, first)
        if path:
            chain = " → ".join([name, *path])
            raise GrammarError(f"rule `{name}` is left-recursive: it can start with itself ({chain}); "
                               f"{LEFT_RECURSION_HINT}", *starts[name])


def _path_back_to(target: str, first: dict[str, set[str]]) -> list[str]:
    """A chain of first-position references from ``target`` back to itself, or []."""
    stack = [(child, [child]) for child in sorted(first[target])]
    seen: set[str] = set()
    while stack:
        name, path = stack.pop()
        if name == target:
            return path
        if name in seen:
            continue
        seen.add(name)
        stack.extend((child, path + [child]) for child in sorted(first[name]))
    return []
