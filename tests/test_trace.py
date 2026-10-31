"""The step trace of a constrained run (tab ①'s step slider), through the pipeline entry point."""

import pytest

from grammar_lab.pipelines import exact_count, run_grammar
from grammar_lab.presets import BOOK_RECORD
from tests.fakes import CHOLERA_SCRIPT, TIRUKKURAL_SCRIPT, FakeModel, run_book


def texts(choices):
    return [c.text for c in choices]


def test_each_step_records_the_top_ten_before_the_mask_with_disallowed_tokens_flagged():
    step = run_book(FakeModel(CHOLERA_SCRIPT)).constrained.steps[1]  # after "Gabriel"

    # The model wants " García", then " Garcia", then the rest in vocabulary order.
    assert [(c.text, c.allowed) for c in step.before] == [
        (" García", False), (" Garcia", True), ("<eos>", False), ("<|im_start|>", False),
        ("Gabriel", True), (" Márquez", False), (" Marquez", True), (" |", True), ("|", True), ("| ", True),
    ]
    assert step.before[0].prob > step.before[1].prob > step.before[2].prob


def test_each_step_records_the_top_ten_allowed_tokens_renormalised_and_the_chosen_token():
    step = run_book(FakeModel(CHOLERA_SCRIPT)).constrained.steps[1]

    assert texts(step.after) == [" Garcia", "Gabriel", " Marquez", " |", "|", "| ", " Love", " in", " the", " Time"]
    assert all(c.allowed for c in step.after)
    assert sum(c.prob for c in step.after) == pytest.approx(1.0)
    assert step.after[0].prob > step.before[1].prob  # renormalising raises the allowed tokens' share
    assert step.chosen.text == " Garcia"


def test_each_step_counts_the_tokens_checked_and_disallowed_until_ten_allowed_were_found():
    step = run_book(FakeModel(CHOLERA_SCRIPT)).constrained.steps[1]

    # " García", " Garcia", then ids 0..13 except the two already seen: 14 checked, stopping at " Time".
    # Disallowed: " García", <eos>, <|im_start|>, " Márquez".
    assert (step.checked, step.disallowed) == (14, 4)


def test_each_step_records_the_grammar_position_open_rules_and_what_may_come_next():
    steps = run_book(FakeModel(CHOLERA_SCRIPT)).constrained.steps

    after_gabriel = steps[1].grammar_position
    assert after_gabriel.open_rules == ["record › author", "record › separator"]
    assert after_gabriel.next_chars == ['[a-zA-Z ]', '"|"']
    assert not after_gabriel.can_end

    after_second_separator = steps[11].grammar_position  # "...Cholera |" written, the year is next
    # `year` is the last thing in `record` (and `unk` the last in `year`), so nothing of the caller is left open.
    assert after_second_separator.open_rules == ["year", "unk"]
    assert after_second_separator.next_chars == ['[1-2]', '"N"']


def test_the_run_lists_the_steps_where_the_mask_changed_the_models_top_choice():
    run = run_book(FakeModel(CHOLERA_SCRIPT)).constrained

    assert run.mask_changed_steps == [1, 2]  # " García" and " Márquez" were disallowed
    assert run.steps[1].mask_changed_top and not run.steps[0].mask_changed_top


def test_whitespace_and_newlines_inside_tokens_are_shown_visibly():
    model = FakeModel([["\n", " ", "Valluvar"]])
    step = run_grammar(model, grammar='name ::= [a-zA-Z \\n]*', root="name",
                       system_prompt="", user_text="", max_new_tokens=1).constrained.steps[0]

    shown = [c.shown for c in step.before[:3]]
    assert shown == ["⏎", "␣", "Valluvar"]
    first_step = run_book(FakeModel(CHOLERA_SCRIPT)).constrained.steps[1]
    assert [c.shown for c in first_step.after[3:6]] == ["␣|", "|", "|␣"]  # " |", "|" and "| " can be told apart


@pytest.mark.parametrize("paragraph, script", [("Cholera", CHOLERA_SCRIPT), ("Tirukkural", TIRUKKURAL_SCRIPT)])
def test_the_first_character_pre_filter_changes_nothing_but_speed(paragraph, script):
    def run(prefilter):
        return run_grammar(FakeModel(script), BOOK_RECORD.grammar, BOOK_RECORD.root, BOOK_RECORD.system_prompt,
                           BOOK_RECORD.inputs[paragraph], max_new_tokens=50, prefilter=prefilter).constrained

    with_filter, without_filter = run(True), run(False)
    assert with_filter.text == without_filter.text
    assert with_filter.steps == without_filter.steps  # same top-10s, flags, probabilities and counts


def test_exact_count_reports_allowed_and_disallowed_tokens_across_the_full_vocabulary():
    model = FakeModel(CHOLERA_SCRIPT)
    step = run_book(model).constrained.steps[1]  # after "Gabriel", inside `author`

    count = exact_count(model, step)

    # Disallowed: <eos>, <|im_start|>, " García", " Márquez", "1985", "198", "5", "\n" and the empty token.
    assert (count.allowed, count.disallowed) == (18, 9)


@pytest.mark.parametrize("paragraph, script", [("Cholera", CHOLERA_SCRIPT), ("Tirukkural", TIRUKKURAL_SCRIPT)])
def test_exact_count_agrees_with_the_token_by_token_check_at_every_step(paragraph, script):
    model = FakeModel(script)
    run = run_book(model, paragraph=paragraph).constrained

    for step in run.steps:
        count = exact_count(model, step)
        assert count.allowed + count.disallowed == len(model.tokenizer.token_texts)
        assert all((c.token_id in count.allowed_ids) == c.allowed for c in step.before)
        assert {c.token_id for c in step.after} <= count.allowed_ids
        first_allowed_before = next(c for c in step.before if c.token_id in count.allowed_ids)
        assert step.chosen.token_id == first_allowed_before.token_id
