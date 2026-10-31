"""Greedy decoding, with or without a grammar.

With a grammar, each step walks the candidates from the highest logit down, checking each
one, until TOP_K *allowed tokens* (ones that keep the output a valid prefix of the grammar)
have been found. The first allowed token is chosen. Special tokens are never allowed, except
end-of-sequence once the output fully matches the root rule. Every step is recorded as a
``Step`` so the trace view can show why each token was chosen.
"""

from dataclasses import dataclass, field
from typing import Optional

from grammar_lab.model_boundary import ModelBoundary, NextTokenScores, TokenizerView
from grammar_lab.recognizer import GrammarPosition, PositionView

END_OF_SEQUENCE = "end of sequence"
GRAMMAR_COMPLETE = "grammar complete"   # complete and nothing more can follow
TOKEN_LIMIT = "token limit"
NO_ALLOWED_TOKEN = "no allowed token"

TOP_K = 10  # tokens shown before and after the mask


@dataclass(frozen=True)
class TokenChoice:
    token_id: int
    text: str       # the token's text, or a special token's name
    prob: float     # before the mask: over the whole vocabulary; after: renormalised over the allowed top-k
    allowed: bool

    @property
    def shown(self) -> str:
        """The token with visible whitespace: " 3" is shown as "␣3", a newline as "⏎"."""
        return visible_token(self.text)


def visible_token(text: str) -> str:
    if not text:
        return "∅"
    return text.replace(" ", "␣").replace("\n", "⏎").replace("\t", "⇥").replace("\r", "↵")


@dataclass
class Step:
    """One decoding step of a constrained run."""
    before: list[TokenChoice]   # top-k by logit, allowed or not
    after: list[TokenChoice]    # top-k allowed tokens, renormalised
    checked: int                # tokens checked before top-k allowed ones were found
    disallowed: int             # of those, how many were disallowed
    position: GrammarPosition = field(repr=False, compare=False)  # the grammar position checked against

    @property
    def chosen(self) -> Optional[TokenChoice]:
        return self.after[0] if self.after else None

    @property
    def mask_changed_top(self) -> bool:
        """Was the model's top choice disallowed, so the mask changed what was written?"""
        return bool(self.before) and not self.before[0].allowed

    @property
    def grammar_position(self) -> PositionView:
        """Which rules were open and what could come next, before this step's token."""
        return self.position.describe()


@dataclass
class Run:
    token_ids: list[int]
    text: str
    stop_reason: str
    complete: Optional[bool] = None  # constrained runs only: does the output match the root rule?
    steps: list[Step] = field(default_factory=list)  # constrained runs only

    @property
    def mask_changed_steps(self) -> list[int]:
        """Indexes of the steps where the model's top choice was disallowed."""
        return [i for i, step in enumerate(self.steps) if step.mask_changed_top]


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


def constrained_step(scores: NextTokenScores, position: GrammarPosition, tokens: TokenizerView):
    """Check tokens from the highest logit down until TOP_K allowed ones are found.

    Returns the recorded Step and the position after its chosen token (None if nothing is allowed).
    """
    before: list[TokenChoice] = []
    allowed: list[tuple[int, GrammarPosition]] = []
    checked = 0
    for token_id in scores.ranked_ids():
        if len(allowed) == TOP_K:  # then at least TOP_K were checked, so `before` is full too
            break
        after = check_token(token_id, position, tokens)
        checked += 1
        if len(before) < TOP_K:
            before.append(TokenChoice(token_id, tokens.name_of(token_id), scores.prob(token_id), after is not None))
        if after is not None:
            allowed.append((token_id, after))
    total = sum(scores.prob(i) for i, _ in allowed)
    after_mask = [TokenChoice(i, tokens.name_of(i), scores.prob(i) / total, True) for i, _ in allowed]
    step = Step(before, after_mask, checked, checked - len(allowed), position)
    return step, (allowed[0][1] if allowed else None)


def generate(model: ModelBoundary, prompt_ids: list[int], max_new_tokens: int,
             position: Optional[GrammarPosition] = None) -> Run:
    """Greedy generation; constrained by the grammar when a start position is given."""
    tokens = model.tokenizer
    constrained = position is not None
    next_scores = model.start(prompt_ids)
    generated: list[int] = []
    prev = None
    stop = TOKEN_LIMIT
    steps: list[Step] = []
    for _ in range(max_new_tokens):
        scores = next_scores(prev)
        if constrained:
            step, next_position = constrained_step(scores, position, tokens)
            steps.append(step)
            if step.chosen is None:
                stop = NO_ALLOWED_TOKEN
                break
            chosen, position = step.chosen.token_id, next_position
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
        return Run(generated, tokens.text_of(generated), stop, complete=position.is_complete, steps=steps)
    return Run(generated, tokens.decode(generated), stop)
