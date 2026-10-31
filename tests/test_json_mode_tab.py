"""Tab ③ (JSON mode) through its pipeline entry point, with a fake model."""

from grammar_lab.pipelines import run_json_mode
from grammar_lab.presets import JSON_BOOK
from tests.fakes import EOS, IM_START, FakeModel

JSON_VOCAB = [
    EOS, IM_START,
    "{", "{\n", "}", "\n}", "\"", " \"", "\":", "\": ", ",", ",\n", "\n", " ", "  ",
    "author", "writer", "title", "year", "Valluvar", "The", " Tirukkural",
    "1985", "NULL", "null", "true", "```", "json", "[", "]",
]


def json_model(script):
    return FakeModel(script, vocab=JSON_VOCAB)


def run_book_json(model, paragraph="Tirukkural", limit_whitespace=True, max_new_tokens=40):
    return run_json_mode(model, JSON_BOOK.system_prompt, JSON_BOOK.inputs[paragraph],
                         limit_whitespace=limit_whitespace, max_new_tokens=max_new_tokens)


def test_json_mode_lets_the_model_choose_its_own_keys_and_write_the_string_null():
    script = [["{"], ["\""], ["writer"], ["\": "], ["\""], ["Valluvar"], ["\""], [","],
              [" \""], ["year"], ["\": "], ["\""], ["NULL"], ["\""], ["}"], [EOS]]
    result = run_book_json(json_model(script))

    assert result.constrained.text == '{"writer": "Valluvar", "year": "NULL"}'
    assert result.complete
    assert result.constrained.stop_reason == "end of sequence"


def test_the_model_cannot_start_with_anything_but_a_json_object():
    result = run_book_json(json_model([["```", "json", "{"], ["}"], [EOS]]))

    first = result.constrained.steps[0]
    assert [(c.text, c.allowed) for c in first.before[:3]] == [("```", False), ("json", False), ("{", True)]
    assert result.constrained.text == "{}"
    assert result.unconstrained.text.startswith("```")


# After "{}" the object is complete, but the model keeps preferring a newline over ending.
WHITESPACE_LOVER = [["{"], ["}"]] + [["\n", EOS]] * 60


def test_with_whitespace_unlimited_a_whitespace_favouring_model_loops_until_the_token_limit():
    result = run_book_json(json_model(WHITESPACE_LOVER), limit_whitespace=False, max_new_tokens=40)

    assert result.constrained.text == "{}" + "\n" * 38
    assert result.constrained.stop_reason == "token limit"
    assert "ws     ::= [ \\t\\n]*" in result.grammar_text


def test_with_whitespace_limited_the_same_model_cannot_loop_and_ends():
    result = run_book_json(json_model(WHITESPACE_LOVER), limit_whitespace=True, max_new_tokens=40)

    assert result.constrained.text == "{}\n"
    assert result.constrained.stop_reason == "end of sequence"
    assert result.complete
    third = result.constrained.steps[3]  # a second newline is disallowed, so end-of-sequence wins
    assert [(c.text, c.allowed) for c in third.before[:2]] == [("\n", False), (EOS, True)]


def test_limit_whitespace_still_allows_a_newline_and_indent_between_members():
    script = [["{\n"], ["  "], ["\""], ["author"], ["\": "], ["\""], ["Valluvar"], ["\""], ["\n}"], [EOS]]
    result = run_book_json(json_model(script))

    assert result.constrained.text == '{\n  "author": "Valluvar"\n}'
    assert result.complete


def test_json_mode_grammar_is_shown_and_is_the_default_with_limited_whitespace():
    limited = run_book_json(json_model([["{"], ["}"], [EOS]]))
    unlimited = run_book_json(json_model([["{"], ["}"], [EOS]]), limit_whitespace=False)

    assert limited.grammar_text.startswith("root   ::= object\n")
    assert 'ws     ::= (" " | "\\n" indent)?' in limited.grammar_text
    assert "[ \\t\\n]*" not in limited.grammar_text
    assert run_json_mode(json_model([["{"], ["}"], [EOS]]), "", "").grammar_text == limited.grammar_text
    assert unlimited.grammar_text != limited.grammar_text
