"""One entry point per tab: connects the model boundary to the grammar engine.

Every ``run_*`` entry point takes ``(model, <constraint>, system_prompt, user_text, max_new_tokens,
<options>)``. Tabs ②–④ also have a ``preview_*`` helper that builds the grammar without running
the model, so the UI can show it (or the error) live while the input is edited. This module is the
UI's only door into ``grammar_lab``: the types the UI needs are re-exported here.
"""

import re
from dataclasses import dataclass, replace
from typing import Optional

from grammar_lab import decoding, json_grammar, json_schema
from grammar_lab.decoding import TOP_K, ExactCount, Run, Step, StopReason, generate
from grammar_lab.gbnf import GrammarError, parse_gbnf
from grammar_lab.model_boundary import ModelBoundary
from grammar_lab.recognizer import start_position
from grammar_lab.regex import ROOT as REGEX_ROOT
from grammar_lab.regex import regex_to_gbnf

__all__ = [
    "TOP_K", "ExactCount", "GrammarError", "GrammarRunResult", "Run", "Step", "StopReason",
    "exact_count", "preview_json_mode", "preview_regex", "preview_schema", "rule_names",
    "run_grammar", "run_json_mode", "run_regex", "run_structured",
]


@dataclass
class GrammarRunResult:
    constrained: Optional[Run] = None
    unconstrained: Optional[Run] = None
    error: Optional[GrammarError] = None  # set when the grammar (or regex, or schema) fails to parse
    grammar_text: Optional[str] = None    # the GBNF actually used, when generated (tabs ②–④)
    dataclass_view: Optional[str] = None  # tab ④: the schema as read-only Python dataclasses

    @property
    def complete(self) -> bool:
        """✅ complete (fully matches the root rule, not stopped by the token limit) or ⚠️ incomplete."""
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


def _run_preview(model: ModelBoundary, preview: GrammarRunResult, root: str, system_prompt: str, user_text: str,
                 max_new_tokens: int) -> GrammarRunResult:
    """Decode with a previewed (generated) grammar, keeping what the preview shows."""
    if preview.error:
        return preview
    runs = run_grammar(model, preview.grammar_text, root, system_prompt, user_text, max_new_tokens)
    return replace(preview, constrained=runs.constrained, unconstrained=runs.unconstrained, error=runs.error)


def preview_regex(regex: str) -> GrammarRunResult:
    """Tab ② without running the model: the generated GBNF, or the regex error."""
    try:
        return GrammarRunResult(grammar_text=regex_to_gbnf(regex))
    except GrammarError as error:
        return GrammarRunResult(error=error)


def run_regex(model: ModelBoundary, regex: str, system_prompt: str, user_text: str,
              max_new_tokens: int = 256) -> GrammarRunResult:
    """Tab ②: the regex is converted to GBNF, then decoded exactly like tab ①."""
    return _run_preview(model, preview_regex(regex), REGEX_ROOT, system_prompt, user_text, max_new_tokens)


def preview_json_mode(limit_whitespace: bool = True) -> GrammarRunResult:
    """Tab ③ without running the model: the grammar allowing any JSON object."""
    return GrammarRunResult(grammar_text=json_grammar.json_gbnf(limit_whitespace))


def run_json_mode(model: ModelBoundary, system_prompt: str, user_text: str, max_new_tokens: int = 256,
                  limit_whitespace: bool = True) -> GrammarRunResult:
    """Tab ③: a grammar allowing any JSON object, so keys and value types are up to the model."""
    return _run_preview(model, preview_json_mode(limit_whitespace), json_grammar.ROOT, system_prompt, user_text,
                        max_new_tokens)


def preview_schema(schema: str, limit_whitespace: bool = True) -> GrammarRunResult:
    """Tab ④ without running the model: the generated GBNF and dataclass view, or the schema error."""
    try:
        return GrammarRunResult(grammar_text=json_schema.schema_to_gbnf(schema, limit_whitespace),
                                dataclass_view=json_schema.schema_to_dataclass(schema))
    except GrammarError as error:
        return GrammarRunResult(error=error)


def run_structured(model: ModelBoundary, schema: str, system_prompt: str, user_text: str,
                   max_new_tokens: int = 256, limit_whitespace: bool = True) -> GrammarRunResult:
    """Tab ④: the JSON Schema is converted to GBNF (fields in order, all required), then decoded."""
    return _run_preview(model, preview_schema(schema, limit_whitespace), json_schema.ROOT, system_prompt, user_text,
                        max_new_tokens)


def rule_names(grammar: str) -> list[str]:
    """The rule names defined in GBNF text, in order, for choosing the root rule.

    A plain line scan, so the choices stay available while the grammar has errors.
    """
    names = re.findall(r"^[ \t]*([A-Za-z0-9_-]+)[ \t]*::=", grammar, flags=re.MULTILINE)
    return list(dict.fromkeys(names))
