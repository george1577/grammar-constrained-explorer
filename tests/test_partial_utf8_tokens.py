"""Byte-level tokens that hold only part of a multi-byte UTF-8 character.

Qwen's tokenizer splits rarer characters (Tamil letters, for example) across tokens. The
grammar must judge the character those bytes make up, never the "�" a lone byte decodes to.
"""

import pytest

from grammar_lab.pipelines import exact_count, run_grammar, run_json_mode
from tests.fakes import EOS, IM_START, FakeModel

I_ACUTE = "í".encode()       # b"\xc3\xad"
E_ACUTE_TAIL = "é".encode()[1:]  # b"\xa9": completes "é" after b"\xc3"
LEAD, TAIL = I_ACUTE[:1], I_ACUTE[1:]

UTF8_VOCAB = [
    EOS, IM_START,
    "{", "}", "\"", "\":", " ", "\"}",
    "author", "Garc", "a", "Garcia",
    LEAD, TAIL, E_ACUTE_TAIL, b"a\xc3",
]


def model(script):
    return FakeModel(script, vocab=UTF8_VOCAB)


def run_author_grammar(script, grammar, max_new_tokens=10, prefilter=True):
    return run_grammar(model(script), grammar, "name", "", "", max_new_tokens, prefilter=prefilter)


GARCIA_IN_PIECES = [["{"], ["\""], ["author"], ["\":"], [" "], ["\""], ["Garc"], [LEAD], [TAIL], ["a"], ["\"}"], [EOS]]


def test_a_character_split_across_tokens_is_written_whole_in_json_strings():
    result = run_json_mode(model(GARCIA_IN_PIECES), "", "")

    assert result.constrained.text == '{"author": "García"}'
    assert result.complete


def test_a_partial_character_is_disallowed_when_no_character_it_could_become_is_allowed():
    result = run_author_grammar([["Garc"], [LEAD, "a"], [EOS]], 'name ::= [a-zA-Z ]*')

    step = result.constrained.steps[1]
    assert (step.before[0].text, step.before[0].allowed) == ("<0xC3>", False)
    assert step.chosen.text == "a"
    assert result.constrained.text == "Garca"


def test_a_partial_character_is_allowed_only_if_it_can_still_become_an_allowed_character():
    result = run_author_grammar([["Garc"], [LEAD], [E_ACUTE_TAIL, TAIL], ["a"], [EOS]], 'name ::= [a-zA-Zí]*')

    assert result.constrained.steps[1].chosen.text == "<0xC3>"
    after_lead = result.constrained.steps[2]
    assert [(c.text, c.allowed) for c in after_lead.before[:2]] == [("<0xA9>", False), ("<0xAD>", True)]
    assert result.constrained.text == "García"
    assert result.complete


def test_end_of_sequence_is_disallowed_in_the_middle_of_a_character():
    result = run_author_grammar([["Garc"], [b"a\xc3"], [EOS, TAIL], [EOS]], 'name ::= [^"]*')

    in_the_middle = result.constrained.steps[2]
    assert [(c.text, c.allowed) for c in in_the_middle.before[:2]] == [(EOS, False), ("<0xAD>", True)]
    assert result.constrained.text == "Garcaí"
    assert result.constrained.steps[1].chosen.text == "a<0xC3>"


def test_the_grammar_position_says_a_character_is_unfinished():
    result = run_json_mode(model(GARCIA_IN_PIECES), "", "")

    position = result.constrained.steps[8].grammar_position  # after "Garc" and b"\xc3"
    assert position.next_chars == ["the rest of a character (1 of 2 bytes written)"]
    assert not position.can_end


def test_pre_filter_and_exact_count_agree_with_the_full_check_on_partial_characters():
    with_filter = run_author_grammar(GARCIA_IN_PIECES[5:], 'name ::= "\\"" [a-zA-Zí]* "\\"}"', prefilter=True)
    without = run_author_grammar(GARCIA_IN_PIECES[5:], 'name ::= "\\"" [a-zA-Zí]* "\\"}"', prefilter=False)

    assert with_filter.constrained.steps == without.constrained.steps
    for step in with_filter.constrained.steps:
        count = exact_count(model(GARCIA_IN_PIECES), step)
        assert all((c.token_id in count.allowed_ids) == c.allowed for c in step.before)


@pytest.mark.parametrize("token, shown", [(LEAD, "<0xC3>"), (b"a\xc3", "a<0xC3>"), (I_ACUTE[:2], "í")])
def test_partial_characters_are_shown_as_their_bytes(token, shown):
    vocab = UTF8_VOCAB + ([token] if token not in UTF8_VOCAB else [])
    result = run_grammar(FakeModel([[token]], vocab=vocab), 'name ::= [^"]*', "name", "", "", 1)

    assert result.constrained.steps[0].before[0].text == shown
