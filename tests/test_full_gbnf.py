"""Tab ① with the full GBNF subset, through the pipeline entry point, with a fake model."""

from grammar_lab.pipelines import run_grammar
from grammar_lab.presets import ARITHMETIC
from tests.fakes import EOS, IM_START, VOCAB, FakeModel

ARITH_VOCAB = [
    EOS, IM_START,
    "bill", "_apples", "mae", " total", "total",
    "3", "2", "4", "5", "6",
    " +", "+", " =", "=", " >", ">", " ", "\n", "(", ")",
]


def arith(script):
    return FakeModel(script, vocab=ARITH_VOCAB)


def run_arithmetic(model, question, max_new_tokens=40):
    return run_grammar(model, ARITHMETIC.grammar, ARITHMETIC.root, ARITHMETIC.system_prompt,
                       ARITHMETIC.inputs[question], max_new_tokens)


def test_arithmetic_preset_produces_an_equation_for_total_apples():
    script = [["bill"], ["_apples"], [" +"], ["mae"], ["_apples"], [" ="], [" total"], ["_apples"],
              ["\n"], [EOS]]
    result = run_arithmetic(arith(script), "Total apples")

    assert result.constrained.text == "bill_apples +mae_apples = total_apples\n"
    assert result.constrained.stop_reason == "end of sequence"
    assert result.complete


def test_a_grammar_too_strict_forces_equals_where_the_model_wants_greater_than():
    script = [["3"], [" +"], ["2"], [" >", " ="], ["2"], [" +"], ["4"]]
    result = run_arithmetic(arith(script), "More apples than oranges?", max_new_tokens=5)

    assert result.unconstrained.text == "3 +2 >2"
    assert result.constrained.text == "3 +2 =2"
    assert result.constrained.stop_reason == "token limit"
    assert not result.complete


NAME_AND_OPTIONAL_YEAR = r"""
# an author, optionally followed by a year
answer ::= name (" |" year)?   # the year part may be left out
name   ::= [^|0-9\n]+          # anything but a bar, digit or newline: accents allowed
year   ::= [0-9]+
"""


def run_small(script, grammar, root, max_new_tokens=10):
    return run_grammar(FakeModel(script), grammar, root, "Extract it.", "Some paragraph.", max_new_tokens)


def test_comments_negated_classes_and_optional_groups():
    with_year = run_small([["Gabriel"], [" García"], [" |"], ["1985"], [EOS]], NAME_AND_OPTIONAL_YEAR, "answer")
    without_year = run_small([["Gabriel"], [EOS]], NAME_AND_OPTIONAL_YEAR, "answer")

    assert with_year.constrained.text == "Gabriel García |1985"
    assert with_year.complete
    assert without_year.constrained.text == "Gabriel"
    assert without_year.constrained.stop_reason == "end of sequence"


def test_negated_class_disallows_the_excluded_characters():
    result = run_small([["Gabriel"], ["1985", "|", " Garcia"], [EOS]], "name ::= [^|0-9]+", "name")

    assert result.constrained.text == "Gabriel Garcia"


def test_plus_needs_at_least_one_repetition():
    result = run_small([[EOS, "1985"], [EOS]], 'year ::= [0-9]+', "year")

    assert result.constrained.text == "1985"
    assert result.complete


def test_choosing_another_root_rule_constrains_the_whole_output_to_that_rule():
    from grammar_lab.presets import BOOK_RECORD
    result = run_small([["Gabriel", "1985"], [EOS]], BOOK_RECORD.grammar, "year")

    assert result.constrained.text == "1985"
    assert result.complete


def test_left_recursion_is_rejected_with_a_rewrite_hint():
    grammar = 'expr ::= term | expr "+" term\nterm ::= [0-9]+\n'
    result = run_small([["1985"]], grammar, "expr")

    assert result.constrained is None and result.unconstrained is None
    assert (result.error.line, result.error.column) == (1, 1)
    assert "left-recursive" in str(result.error)
    assert 'term ("+" term)*' in str(result.error)


def test_indirect_left_recursion_through_a_nullable_prefix_is_rejected():
    grammar = 'a ::= ws b\nb ::= a "x" | "y"\nws ::= [ ]*\n'
    result = run_small([["1985"]], grammar, "a")

    assert "left-recursive" in str(result.error)
    assert (result.error.line, result.error.column) == (1, 1)


def test_counted_repeats_are_reported_as_unsupported():
    result = run_small([["1985"]], "year ::= [0-9]{4}", "year")

    assert (result.error.line, result.error.column) == (1, 15)
    assert "{m,n}" in str(result.error) and "not supported" in str(result.error)


def test_an_unclosed_group_gives_a_located_error():
    result = run_small([["1985"]], 'x ::= "a"\nyear ::= ("1" [0-9]', "year")

    assert (result.error.line, result.error.column) == (2, 10)
    assert "never closed" in str(result.error)


def test_arithmetic_runs_use_the_notebook_system_prompt_and_questions():
    model = arith([[EOS]])
    run_arithmetic(model, "Total apples", max_new_tokens=1)
    run_arithmetic(model, "More apples than oranges?", max_new_tokens=1)

    system, user = model.prompts[0]
    assert "You are a math instructor." in system["content"]
    assert "Respond with the mathematical expression" in system["content"]
    assert user["content"].endswith("How many total apples do Bill and Mae have?\n")
    assert model.prompts[1][1]["content"].endswith("Do Bill and Mae have more apples than oranges?\n")


def test_root_rule_choices_list_the_rules_in_order_even_while_the_grammar_is_broken():
    from grammar_lab.pipelines import rule_names

    assert rule_names(ARITHMETIC.grammar) == ["root", "expr", "term", "ident", "num", "ws"]
    assert rule_names('# comment\nrecord ::= author "|" (\nauthor-name ::= [a-z]*') == ["record", "author-name"]


# Qwen wants to end right after "3 +2 = 5". The notebook grammar needs a newline after every
# line, so end-of-sequence is disallowed there; the default preset puts newlines between lines.
WANTS_TO_STOP_AFTER_THE_ANSWER = [["3"], [" +"], ["2"], [" ="], [" "], ["5"], [EOS, "\n"], [EOS]]


def test_the_default_arithmetic_preset_lets_the_model_stop_right_after_the_answer():
    from grammar_lab.presets import ARITHMETIC_STOPPABLE, GRAMMAR_PRESETS
    result = run_grammar(arith(WANTS_TO_STOP_AFTER_THE_ANSWER), ARITHMETIC_STOPPABLE.grammar,
                         ARITHMETIC_STOPPABLE.root, ARITHMETIC_STOPPABLE.system_prompt,
                         ARITHMETIC_STOPPABLE.inputs["Total apples"], 20)

    assert result.constrained.text == "3 +2 = 5"
    assert result.constrained.stop_reason == "end of sequence"
    assert list(GRAMMAR_PRESETS.values())[0] is ARITHMETIC_STOPPABLE


def test_the_notebook_arithmetic_grammar_disallows_ending_before_the_newline():
    result = run_arithmetic(arith(WANTS_TO_STOP_AFTER_THE_ANSWER), "Total apples", max_new_tokens=20)

    step = result.constrained.steps[6]
    assert [(c.text, c.allowed) for c in step.before[:2]] == [(EOS, False), ("\n", True)]
    assert result.constrained.text == "3 +2 = 5\n"
