"""One entry point per tab: connects the model boundary to the grammar engine."""

import re
from dataclasses import dataclass
from typing import Optional

from grammar_lab import decoding, json_grammar, json_schema
from grammar_lab.decoding import ExactCount, Run, Step, generate
from grammar_lab.gbnf import GrammarError, parse_gbnf
from grammar_lab.model_boundary import ModelBoundary
from grammar_lab.recognizer import start_position
from grammar_lab.regex import ROOT, regex_to_gbnf


@dataclass
class GrammarRunResult:
    constrained: Optional[Run] = None
    unconstrained: Optional[Run] = None
    error: Optional[GrammarError] = None  # set when the grammar (or regex) fails to parse
    grammar_text: Optional[str] = None    # the GBNF actually used, when generated (tabs ②–④)
    dataclass_view: Optional[str] = None  # tab ④: the schema as read-only Python dataclasses

    @property
    def complete(self) -> bool:
        """✅ complete (the constrained output fully matches the root rule) or ⚠️ incomplete."""
        return bool(self.constrained and self.constrained.complete)


def exact_count(model: ModelBoundary, step: Step) -> ExactCount:
    """On demand, for one recorded step: how many tokens of the full vocabulary the grammar allows."""
    return decoding.exact_count(model.tokenizer, step)


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


def run_regex(model: ModelBoundary, regex: str, system_prompt: str, user_text: str,
              max_new_tokens: int = 256) -> GrammarRunResult:
    """Tab ②: the regex is converted to GBNF, then decoded exactly like tab ①."""
    try:
        grammar_text = regex_to_gbnf(regex)
    except GrammarError as error:
        return GrammarRunResult(error=error)
    result = run_grammar(model, grammar_text, ROOT, system_prompt, user_text, max_new_tokens)
    result.grammar_text = grammar_text
    return result


def run_json_mode(model: ModelBoundary, system_prompt: str, user_text: str, limit_whitespace: bool = True,
                  max_new_tokens: int = 256) -> GrammarRunResult:
    """Tab ③: a grammar allowing any JSON object, so keys and value types are up to the model."""
    grammar_text = json_grammar.json_gbnf(limit_whitespace)
    result = run_grammar(model, grammar_text, json_grammar.ROOT, system_prompt, user_text, max_new_tokens)
    result.grammar_text = grammar_text
    return result


def run_structured(model: ModelBoundary, schema: str, system_prompt: str, user_text: str,
                   max_new_tokens: int = 256) -> GrammarRunResult:
    """Tab ④: the JSON Schema is converted to GBNF (fields in order, all required), then decoded."""
    preview = preview_schema(schema)
    if preview.error:
        return preview
    result = run_grammar(model, preview.grammar_text, json_schema.ROOT, system_prompt, user_text, max_new_tokens)
    result.grammar_text, result.dataclass_view = preview.grammar_text, preview.dataclass_view
    return result


def preview_schema(schema: str) -> GrammarRunResult:
    """Tab ④ without running the model: the generated GBNF and dataclass view, or the schema error."""
    try:
        return GrammarRunResult(grammar_text=json_schema.schema_to_gbnf(schema),
                                dataclass_view=json_schema.schema_to_dataclass(schema))
    except GrammarError as error:
        return GrammarRunResult(error=error)


def rule_names(grammar: str) -> list[str]:
    """The rule names defined in GBNF text, in order, for choosing the root rule.

    A plain line scan, so the choices stay available while the grammar has errors.
    """
    names = re.findall(r"^[ \t]*([A-Za-z0-9_-]+)[ \t]*::=", grammar, flags=re.MULTILINE)
    return list(dict.fromkeys(names))
