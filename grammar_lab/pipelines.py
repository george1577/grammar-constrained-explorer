"""One entry point per tab: connects the model boundary to the grammar engine."""

from dataclasses import dataclass
from typing import Optional

from grammar_lab.decoding import Run, generate
from grammar_lab.gbnf import GrammarError, parse_gbnf
from grammar_lab.model_boundary import ModelBoundary
from grammar_lab.recognizer import start_position


@dataclass
class GrammarRunResult:
    constrained: Optional[Run] = None
    unconstrained: Optional[Run] = None
    error: Optional[GrammarError] = None  # set when the grammar fails to parse

    @property
    def complete(self) -> bool:
        """✅ complete (the constrained output fully matches the root rule) or ⚠️ incomplete."""
        return bool(self.constrained and self.constrained.complete)


def chat_messages(system_prompt: str, user_text: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]


def run_grammar(model: ModelBoundary, grammar: str, root: str, system_prompt: str, user_text: str,
                max_new_tokens: int = 256, prefilter: bool = True) -> GrammarRunResult:
    """Tab ①: a GBNF grammar-constrained run next to an unconstrained run of the same prompt.

    ``prefilter`` switches the first-character pre-filter used while checking tokens (same results either way).
    """
    try:
        start = start_position(parse_gbnf(grammar), root)
    except GrammarError as error:
        return GrammarRunResult(error=error)
    prompt = model.tokenizer.chat_prompt_ids(chat_messages(system_prompt, user_text))
    return GrammarRunResult(
        constrained=generate(model, prompt, max_new_tokens, start, prefilter=prefilter),
        unconstrained=generate(model, prompt, max_new_tokens),
    )
