"""A character-level, incremental recognizer for a parsed Grammar.

The grammar is compiled into flat *alternatives*: each is a rule name plus a tuple of
elements, where an element is either a CharClass (match one character) or the index of
a rule (expand it). Groups and repeats become generated helper rules, e.g. ``x*`` becomes
``R ::= x R | (empty)``.

A ``GrammarPosition`` is the set of every *way* the text so far can sit inside the grammar.
Each way is a tuple of *open rules*, outermost first, each held as ``(alternative, next element
index)``. The innermost open rule always points at a character element; an empty way means the
root rule is complete. A rule reference that is the last element of its alternative (a tail call)
replaces that open rule instead of nesting inside it, so ``R ::= x R`` does not grow the way. That
is why the trace's "Open rules" can leave out a rule once its last element has been entered.
"""

from dataclasses import dataclass
from typing import Optional

from grammar_lab.gbnf import Alt, CharClass, Expr, Grammar, GrammarError, Literal, Repeat, RuleRef, Seq

Element = CharClass | int  # a character to match, or a rule index to expand
OpenRule = tuple[int, int]      # (alternative, next element index)
Way = tuple[OpenRule, ...]      # the open rules, outermost first


@dataclass(frozen=True)
class Alternative:
    rule: str                  # the user's rule name, or a generated name like "author/*"
    elements: tuple[Element, ...]


class CompiledGrammar:
    def __init__(self, grammar: Grammar, root: str):
        if root not in grammar.rules:
            raise GrammarError(f"the root rule `{root}` is not defined in the grammar", 1, 1)
        self.alternatives: list[Alternative] = []
        self.rule_alternatives: list[list[int]] = []  # rule index -> alternative ids
        self.rule_names: list[str] = []
        self._index = {name: i for i, name in enumerate(grammar.rules)}
        for name in grammar.rules:
            self._new_rule(name)
        for name, body in grammar.rules.items():
            self._add_alternatives(self._index[name], name, body)
        self.root = self._index[root]

    def _new_rule(self, name: str) -> int:
        self.rule_names.append(name)
        self.rule_alternatives.append([])
        return len(self.rule_names) - 1

    def _add_alternatives(self, rule: int, name: str, expr: Expr):
        options = expr.options if isinstance(expr, Alt) else (expr,)
        for option in options:
            self.alternatives.append(Alternative(name, self._elements(name, option)))
            self.rule_alternatives[rule].append(len(self.alternatives) - 1)

    def _elements(self, name: str, expr: Expr) -> tuple[Element, ...]:
        if isinstance(expr, Literal):
            return tuple(CharClass(((c, c),)) for c in expr.text)
        if isinstance(expr, CharClass):
            return (expr,)
        if isinstance(expr, RuleRef):
            return (self._index[expr.name],)
        if isinstance(expr, Seq):
            return tuple(e for item in expr.items for e in self._elements(name, item))
        if isinstance(expr, Alt):
            helper = self._new_rule(f"{name}/group")
            self._add_alternatives(helper, f"{name}/group", expr)
            return (helper,)
        if isinstance(expr, Repeat):
            return self._repeat(name, expr)
        raise TypeError(f"unknown grammar expression {expr!r}")

    def _repeat(self, name: str, expr: Repeat) -> tuple[Element, ...]:
        item = self._elements(name, expr.item)
        required = item * expr.min
        if expr.max is None:  # R ::= item R | (empty)
            helper_name = f"{name}/*"
            helper = self._new_rule(helper_name)
            for elements in (item + (helper,), ()):
                self.alternatives.append(Alternative(helper_name, elements))
                self.rule_alternatives[helper].append(len(self.alternatives) - 1)
            return required + (helper,)
        # up to (max - min) optional copies, each R ::= item | (empty)
        optional: tuple[Element, ...] = ()
        for _ in range(expr.max - expr.min):
            helper_name = f"{name}/?"
            helper = self._new_rule(helper_name)
            for elements in (item, ()):
                self.alternatives.append(Alternative(helper_name, elements))
                self.rule_alternatives[helper].append(len(self.alternatives) - 1)
            optional += (helper,)
        return required + optional

    def start(self) -> "GrammarPosition":
        ways: set[Way] = set()
        self._expand_rule(self.root, (), ways, set())
        return GrammarPosition(self, frozenset(ways))

    # ---- closure: expand rule references until every way's innermost open rule is at a character ----

    def _expand_rule(self, rule: int, rest: Way, out: set[Way], seen: set[Way]):
        for alt in self.rule_alternatives[rule]:
            self._closure(rest + ((alt, 0),), out, seen)

    def _closure(self, way: Way, out: set[Way], seen: set[Way]):
        if way in seen:
            return
        seen.add(way)
        if not way:
            out.add(way)
            return
        alt, i = way[-1]
        elements = self.alternatives[alt].elements
        if i == len(elements):  # this alternative is finished: return to the caller
            self._closure(way[:-1], out, seen)
            return
        element = elements[i]
        if isinstance(element, CharClass):
            out.add(way)
            return
        rest = way[:-1] + ((alt, i + 1),) if i + 1 < len(elements) else way[:-1]
        self._expand_rule(element, rest, out, seen)

    def step(self, ways: frozenset[Way], char: str) -> frozenset[Way]:
        out: set[Way] = set()
        seen: set[Way] = set()
        for way in ways:
            if not way:
                continue
            alt, i = way[-1]
            if self.alternatives[alt].elements[i].matches(char):
                self._closure(way[:-1] + ((alt, i + 1),), out, seen)
        return frozenset(out)


class GrammarPosition:
    """Where the output so far sits in the grammar. Immutable; advancing returns a new one."""

    def __init__(self, grammar: CompiledGrammar, ways: frozenset[Way]):
        self._grammar = grammar
        self.ways = ways
        self._after_char: dict[str, frozenset[Way]] = {}

    def advance(self, text: str) -> Optional["GrammarPosition"]:
        """The position after appending ``text``, or None if that breaks the grammar."""
        if not text:
            return self
        ways = self._after_char.get(text[0])
        if ways is None:
            ways = self._after_char[text[0]] = self._grammar.step(self.ways, text[0])
        for char in text[1:]:
            if not ways:
                break
            ways = self._grammar.step(ways, char)
        return GrammarPosition(self._grammar, ways) if ways else None

    @property
    def is_complete(self) -> bool:
        """Does the output so far fully match the root rule?"""
        return () in self.ways

    @property
    def can_continue(self) -> bool:
        """Could any more characters follow?"""
        return any(self.ways)

    def next_classes(self) -> list[CharClass]:
        """The character classes the next character must match (any one of them), in grammar order."""
        alternatives = self._grammar.alternatives
        innermost = sorted({way[-1] for way in self.ways if way})
        return list(dict.fromkeys(alternatives[alt].elements[i] for alt, i in innermost))

    def describe(self) -> "PositionView":
        """Which rules are open and what characters may come next, for display."""
        alternatives = self._grammar.alternatives
        paths: dict[str, tuple[int, int]] = {}  # rule path -> earliest innermost open rule, for grammar order
        for way in self.ways:
            if not way:
                continue
            names: list[str] = []
            for alt, _ in way:
                name = alternatives[alt].rule.split("/")[0]  # helpers like "author/*" belong to `author`
                if not names or names[-1] != name:
                    names.append(name)
            path = " › ".join(names)
            paths[path] = min(paths.get(path, way[-1]), way[-1])
        return PositionView(
            open_rules=sorted(paths, key=paths.__getitem__),
            next_chars=list(dict.fromkeys(show_char_class(c) for c in self.next_classes())),
            can_end=self.is_complete,
        )


@dataclass(frozen=True)
class PositionView:
    """A grammar position, described for display."""
    open_rules: list[str]   # e.g. "record › author": the open rules, root first
    next_chars: list[str]   # e.g. '[a-zA-Z ]' or '"|"': what the next character may be
    can_end: bool           # does the output so far fully match the root rule?


_ESCAPED = {"\n": r"\n", "\t": r"\t", "\r": r"\r", "\\": "\\\\"}


def _show_char(char: str, specials: str) -> str:
    if char in _ESCAPED:
        return _ESCAPED[char]
    return "\\" + char if char in specials else char


def show_char_class(cls: CharClass) -> str:
    """GBNF-style text for a character class: '"|"' for one character, otherwise like '[a-z ]'."""
    if not cls.negated and len(cls.ranges) == 1 and cls.ranges[0][0] == cls.ranges[0][1]:
        return '"' + _show_char(cls.ranges[0][0], '"') + '"'
    body = "".join(
        _show_char(low, "]-^") if low == high else f"{_show_char(low, ']-^')}-{_show_char(high, ']-^')}"
        for low, high in cls.ranges
    )
    return f"[{'^' if cls.negated else ''}{body}]"


def start_position(grammar: Grammar, root: str) -> GrammarPosition:
    return CompiledGrammar(grammar, root).start()
