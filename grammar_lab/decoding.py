"""Greedy decoding, with or without a grammar.

With a grammar, each step walks the candidates from the highest logit down and picks the
first *allowed token*: one that keeps the output a valid prefix of the grammar. Special
tokens are never allowed, except end-of-sequence once the output fully matches the root rule.
"""

from dataclasses import dataclass
from typing import Optional

from grammar_lab.model_boundary import ModelBoundary, NextTokenScores, TokenizerView
from grammar_lab.recognizer import GrammarPosition

END_OF_SEQUENCE = "end of sequence"
GRAMMAR_COMPLETE = "grammar complete"   # complete and nothing more can follow
TOKEN_LIMIT = "token limit"
NO_ALLOWED_TOKEN = "no allowed token"


@dataclass
class Run:
    token_ids: list[int]
    text: str
    stop_reason: str
    complete: Optional[bool] = None  # constrained runs only: does the output match the root rule?


def check_token(token_id: int, position: GrammarPosition, tokens: TokenizerView) -> Optional[GrammarPosition]:
    """The grammar position after this token, or None if the token is disallowed.

    End-of-sequence, when allowed, returns the unchanged position.
    """
    if token_id in tokens.eos_ids:
        return position if position.is_complete else None
    text = tokens.token_texts[token_id]
    if token_id in tokens.special_ids or not text:
        return None
    return position.advance(text)


def first_allowed(scores: NextTokenScores, position: GrammarPosition, tokens: TokenizerView):
    """(token id, next position) for the highest-logit allowed token, or None if there is none."""
    for token_id in scores.ranked_ids():
        after = check_token(token_id, position, tokens)
        if after is not None:
            return token_id, after
    return None


def generate(model: ModelBoundary, prompt_ids: list[int], max_new_tokens: int,
             position: Optional[GrammarPosition] = None) -> Run:
    """Greedy generation; constrained by the grammar when a start position is given."""
    tokens = model.tokenizer
    constrained = position is not None
    next_scores = model.start(prompt_ids)
    generated: list[int] = []
    prev = None
    stop = TOKEN_LIMIT
    for _ in range(max_new_tokens):
        scores = next_scores(prev)
        if constrained:
            choice = first_allowed(scores, position, tokens)
            if choice is None:
                stop = NO_ALLOWED_TOKEN
                break
            chosen, position = choice
        else:
            chosen = scores.ranked_ids()[0]
        if chosen in tokens.eos_ids:
            stop = END_OF_SEQUENCE
            break
        generated.append(chosen)
        prev = chosen
        if constrained and position.is_complete and not position.can_continue:
            stop = GRAMMAR_COMPLETE
            break

    if constrained:
        return Run(generated, tokens.text_of(generated), stop, complete=position.is_complete)
    return Run(generated, tokens.decode(generated), stop)
