"""The model boundary: the only things the rest of the app needs from a language model.

Two parts:
- a ``TokenizerView``: each token id's decoded text, which ids are special (and which of
  those end the sequence), and how to build a chat prompt;
- ``start(prompt_ids)``: returns a ``next_scores(prev_token_or_None)`` function that keeps
  a KV cache between steps and returns the next token's ``NextTokenScores``.

``lm.py`` implements this with torch/transformers; tests implement it with tiny fakes.
Nothing outside ``lm.py`` imports torch.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence


@dataclass(frozen=True)
class TokenizerView:
    token_texts: Sequence[str]          # token id -> decoded text ("" for special tokens)
    special_ids: frozenset[int]         # never allowed by a grammar, except end-of-sequence ids
    eos_ids: frozenset[int]             # end-of-sequence ids (a subset of special_ids)
    chat_prompt_ids: Callable[[list[dict]], list[int]]
    decode: Callable[[list[int]], str]  # proper detokenisation, skipping special tokens

    def text_of(self, token_ids: list[int]) -> str:
        """The output exactly as the grammar saw it: the tokens' texts joined together."""
        return "".join(self.token_texts[i] for i in token_ids if i not in self.special_ids)


class NextTokenScores(Protocol):
    """The model's scores for the next token."""

    def ranked_ids(self) -> Sequence[int]:
        """Every token id, highest logit first."""

    def logit(self, token_id: int) -> float: ...


NextScoresFn = Callable[[Optional[int]], NextTokenScores]


class ModelBoundary(Protocol):
    tokenizer: TokenizerView

    def start(self, prompt_ids: list[int]) -> NextScoresFn:
        """A fresh next_scores function (with its own KV cache) for this prompt."""


class ListScores:
    """NextTokenScores over a plain list of logits."""

    def __init__(self, logits: Sequence[float]):
        self._logits = list(logits)
        self._ranked: Optional[list[int]] = None

    def ranked_ids(self) -> list[int]:
        if self._ranked is None:
            self._ranked = sorted(range(len(self._logits)), key=lambda i: -self._logits[i])
        return self._ranked

    def logit(self, token_id: int) -> float:
        return self._logits[token_id]
