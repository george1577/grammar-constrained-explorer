"""Tab ② (regex constraint) through its pipeline entry point, with a fake model."""

import pytest

from grammar_lab.pipelines import run_grammar, run_regex
from grammar_lab.presets import BOOK_REGEX
from tests.fakes import EOS, FakeModel

CHOLERA_SCRIPT = [
    ["Gabriel"], [" García", " Garcia"], [" Márquez", " Marquez"],
    [" |"], [" Love"], [" in"], [" the"], [" Time"], [" of"], [" Cholera"],
    [" |"], ["1985", " "], ["1985"], ["\n"],
]

HAND_WRITTEN_GBNF = """
record ::= field " | " field " | " year
field  ::= [a-zA-Z ]+ | "NULL"
year   ::= [1-2] [0-9] [0-9] [0-9] | "NULL"
"""


def run_book_regex(model, regex=BOOK_REGEX.regex, max_new_tokens=30):
    return run_regex(model, regex, BOOK_REGEX.system_prompt, BOOK_REGEX.inputs["Cholera"], max_new_tokens)


def test_notebook_regex_is_shown_as_the_grammar_it_becomes():
    result = run_book_regex(FakeModel(CHOLERA_SCRIPT))

    assert BOOK_REGEX.regex == r"([a-zA-Z ]+|NULL) \| ([a-zA-Z ]+|NULL) \| ([1-2][0-9][0-9][0-9]|NULL)"
    assert result.grammar_text == (
        'root ::= ([a-zA-Z ]+ | "NULL") " | " ([a-zA-Z ]+ | "NULL") " | " '
        '([1-2] [0-9] [0-9] [0-9] | "NULL")\n'
    )


def test_regex_gives_the_same_constrained_output_as_the_equivalent_hand_written_grammar():
    by_regex = run_book_regex(FakeModel(CHOLERA_SCRIPT))
    by_grammar = run_grammar(FakeModel(CHOLERA_SCRIPT), HAND_WRITTEN_GBNF, "record",
                             BOOK_REGEX.system_prompt, BOOK_REGEX.inputs["Cholera"], 30)

    assert by_regex.constrained.text == "Gabriel Garcia Marquez | Love in the Time of Cholera | 1985"
    assert by_regex.constrained.text == by_grammar.constrained.text
    assert by_regex.complete and by_grammar.complete
    assert by_regex.unconstrained.text.startswith("Gabriel García Márquez | Love")


@pytest.mark.parametrize("regex, column, words", [
    (r"^[a-z]+", 1, "anchors"),
    (r"[a-z]+$", 7, "anchors"),
    (r"\bword", 1, "anchors"),
    (r"(?=NULL)[A-Z]+", 1, "lookarounds"),
    (r"[a-z]+(?<!x)", 7, "lookarounds"),
    (r"(a)\1", 4, "backreferences"),
    (r"(?P<x>a)(?P=x)", 9, "backreferences"),
    (r"([a-z]+", 1, "never closed"),
    (r"a)", 2, "no matching"),
])
def test_unsupported_regex_features_give_clear_located_errors_and_no_runs(regex, column, words):
    result = run_book_regex(FakeModel(CHOLERA_SCRIPT), regex=regex)

    assert result.constrained is None and result.unconstrained is None
    assert result.error.column == column
    assert words in str(result.error)
    assert str(result.error).startswith(f"Column {column}:")


def test_shorthands_counted_repeats_and_escapes_become_grammar():
    result = run_book_regex(FakeModel([["1985"], [EOS]]), regex=r"(?:\d{4}|NULL)\.?")

    assert result.grammar_text == 'root ::= ([0-9] [0-9] [0-9] [0-9] | "NULL") "."?\n'
    assert result.constrained.text == "1985"
    assert result.complete


def test_dot_allows_anything_but_a_newline():
    result = run_book_regex(FakeModel([["Gabriel"], ["\n", " García"], [EOS]]), regex=r".+")

    assert result.grammar_text == "root ::= [^\\n]+\n"
    assert result.constrained.text == "Gabriel García"
