"""Tab ① (grammar-constrained generation) through its pipeline entry point, with a fake model."""

from grammar_lab.pipelines import StopReason, run_grammar
from grammar_lab.presets import BOOK_RECORD
from tests.fakes import CHOLERA_SCRIPT, TIRUKKURAL_SCRIPT, FakeModel, run_book


def test_grammar_wins_over_accented_tokens_the_model_prefers():
    result = run_book(FakeModel(CHOLERA_SCRIPT))

    assert result.unconstrained.text.startswith("Gabriel García Márquez | Love")
    assert result.constrained.text == "Gabriel Garcia Marquez | Love in the Time of Cholera |1985"
    assert result.complete


def test_end_of_sequence_and_special_tokens_are_disallowed_until_the_output_is_complete():
    result = run_book(FakeModel(TIRUKKURAL_SCRIPT), paragraph="Tirukkural")

    assert result.unconstrained.text == ""  # unconstrained, the model just stops
    assert result.constrained.text == "Valluvar | The Tirukkural |NULL"
    assert result.complete


def test_end_of_sequence_is_allowed_once_the_output_fully_matches_the_root_rule():
    model = FakeModel([["Valluvar"], ["<eos>", " The"], [" The"]])
    result = run_grammar(model, grammar="name ::= [a-zA-Z ]*", root="name",
                         system_prompt="Who wrote it?", user_text="The Tirukkural", max_new_tokens=10)

    assert result.constrained.text == "Valluvar"
    assert result.constrained.stop_reason == StopReason.END_OF_SEQUENCE
    assert result.complete


def test_output_at_the_token_limit_is_incomplete():
    result = run_book(FakeModel(CHOLERA_SCRIPT), max_new_tokens=3)

    assert result.constrained.text == "Gabriel Garcia Marquez"
    assert result.constrained.stop_reason == StopReason.TOKEN_LIMIT
    assert not result.complete


def test_generation_stops_when_the_grammar_is_complete_and_nothing_more_can_follow():
    result = run_book(FakeModel(CHOLERA_SCRIPT))

    assert result.constrained.stop_reason == StopReason.GRAMMAR_COMPLETE
    assert len(result.constrained.token_ids) == 12  # "1985" is token 12; the "\n" after it never comes


def test_a_grammar_that_fails_to_parse_gives_a_located_error_and_no_runs():
    model = FakeModel(CHOLERA_SCRIPT)
    result = run_grammar(model, grammar='record ::= author "|" year\nauthor ::= [a-z]*\n',
                         root="record", system_prompt="", user_text="", max_new_tokens=5)

    assert result.constrained is None and result.unconstrained is None
    assert (result.error.line, result.error.column) == (1, 23)
    assert "`year`" in str(result.error)
    assert not result.complete


def test_both_runs_use_the_notebook_system_prompt_and_paragraph():
    model = FakeModel(CHOLERA_SCRIPT)
    run_book(model)

    expected = [
        {"role": "system", "content": BOOK_RECORD.system_prompt},
        {"role": "user", "content": BOOK_RECORD.inputs["Cholera"]},
    ]
    assert model.prompts == [expected]
    assert "Gabriel García Márquez and published in 1985" in BOOK_RECORD.inputs["Cholera"]
