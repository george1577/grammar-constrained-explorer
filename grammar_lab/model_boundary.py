"""The model boundary: the only things the rest of the app needs from a language model.

Two parts:
- a ``TokenizerView``: each token id's decoded text, which ids are special (and which of
  those end the sequence), and how to build a chat prompt;
- ``start(prompt_ids)``: returns a ``next_scores(prev_token_or_None)`` function that keeps
  a KV cache between steps and returns the next token's ``NextTokenScores``.

``lm.py`` implements this with torch/transformers; tests implement it with tiny fakes.
Nothing outside ``lm.py`` imports torch.
"""

import codecs
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol, Sequence


def _hex_bytes(error: UnicodeDecodeError) -> tuple[str, int]:
    return "".join(f"<0x{b:02X}>" for b in error.object[error.start:error.end]), error.end


codecs.register_error("grammar_lab_hex", _hex_bytes)


@dataclass(frozen=True)
class TokenizerView:
    token_texts: Sequence[str]          # token id -> decoded text ("" for special tokens; "�" for partial characters)
    special_ids: frozenset[int]         # never allowed by a grammar, except end-of-sequence ids
    eos_ids: frozenset[int]             # end-of-sequence ids (a subset of special_ids)
    chat_prompt_ids: Callable[[list[dict]], list[int]]
    decode: Callable[[list[int]], str]  # proper detokenisation, skipping special tokens
    special_names: Mapping[int, str] = field(default_factory=dict)  # e.g. 151645 -> "<|im_end|>", for display
    # token id -> raw UTF-8 bytes. A byte-level token may hold only part of a multi-byte character.
    # Empty means every token is whole characters, so its bytes are its text's UTF-8.
    token_bytes: Sequence[bytes] = ()

    def bytes_of(self, token_id: int) -> bytes:
        """The token's raw bytes: what the grammar checks."""
        return self.token_bytes[token_id] if self.token_bytes else self.token_texts[token_id].encode()

    def name_of(self, token_id: int) -> str:
        """Text for showing a token: its text, the special token's name, or bytes of a partial
        character written like "<0xC3>"."""
        if token_id in self.special_ids:
            return self.special_names.get(token_id, f"<special {token_id}>")
        return self.bytes_of(token_id).decode("utf-8", errors="grammar_lab_hex")

    def text_of(self, token_ids: list[int]) -> str:
        """The output exactly as the grammar saw it: the tokens' bytes joined, then decoded."""
        data = b"".join(self.bytes_of(i) for i in token_ids if i not in self.special_ids)
        return data.decode("utf-8", errors="replace")


class NextTokenScores(Protocol):
    """The model's scores for the next token."""

    def ranked_ids(self) -> Sequence[int]:
        """Every token id, highest logit first."""

    def logit(self, token_id: int) -> float: ...

    def prob(self, token_id: int) -> float:
        """Softmax probability over the whole vocabulary."""


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
        top = max(self._logits)
        self._log_total = top + math.log(sum(math.exp(x - top) for x in self._logits))

    def ranked_ids(self) -> list[int]:
        if self._ranked is None:
            self._ranked = sorted(range(len(self._logits)), key=lambda i: -self._logits[i])
        return self._ranked

    def logit(self, token_id: int) -> float:
        return self._logits[token_id]

    def prob(self, token_id: int) -> float:
        return math.exp(self._logits[token_id] - self._log_total)
