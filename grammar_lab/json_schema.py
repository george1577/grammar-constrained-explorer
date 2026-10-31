"""Convert a JSON Schema (a supported subset) into GBNF text, plus a Python dataclass view.

Structured output is a grammar: the schema becomes rules that spell out each object's keys
in schema order, so the model can only fill in the values. Every field is required in the
output, even one with a default or one left out of ``required``: a grammar either allows a
key or it doesn't, and making fields optional in any order would multiply the alternatives.
Enums become literal alternatives, so only their values can be written.

Supported: ``type`` object, string, number, integer or boolean; ``properties``; ``required``;
``enum``; nested objects; and the annotations ``title``, ``description`` and ``default``.
"""

import json
import re

from grammar_lab.gbnf import gbnf_literal
from grammar_lab.json_grammar import LIMITED_WS, NUMBER_RULES, STRING_RULES

ROOT = "root"
PRIMITIVE_RULES = {"string": STRING_RULES, **NUMBER_RULES}


def schema_to_gbnf(schema_text: str) -> str:
    """GBNF text whose root rule matches exactly the JSON the schema describes."""
    schema = json.loads(schema_text)
    rules: dict[str, str] = {}
    used: set[str] = set()
    _object_rule(ROOT, schema, rules, used)
    shared = "".join(PRIMITIVE_RULES[name] for name in PRIMITIVE_RULES if name in used)
    return "".join(f"{name} ::= {body}\n" for name, body in rules.items()) + shared + LIMITED_WS


def _object_rule(name: str, schema: dict, rules: dict[str, str], used: set[str]) -> str:
    rules[name] = ""  # reserve the name (and its place) before nested rules are added
    members = []
    for key, field in schema.get("properties", {}).items():
        members.append(f"{gbnf_literal(json.dumps(key, ensure_ascii=False) + ':')} ws {_value(key, field, rules, used)}")
    # One field per line: '"{" ws', then each `"key": value` (with "," between), then '"}" ws'.
    lines = ['"{" ws', *(m + ' "," ws' for m in members[:-1]), *members[-1:], '"}" ws']
    rules[name] = "\n    ".join(lines)
    return name


def _value(key: str, field: dict, rules: dict[str, str], used: set[str]) -> str:
    if "enum" in field:
        name = _rule_name(key, rules)
        options = " | ".join(gbnf_literal(json.dumps(v, ensure_ascii=False)) for v in field["enum"])
        rules[name] = f"({options}) ws"
        return name
    kind = field["type"]
    if kind == "object":
        return _object_rule(_rule_name(key, rules), field, rules, used)
    used.add(kind)
    return kind


def _rule_name(key: str, rules: dict[str, str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", key).strip("-") or "field"
    if base in PRIMITIVE_RULES or base in ("ws", "indent", "char", "hex"):
        base += "-value"
    name, n = base, 2
    while name in rules:
        name, n = f"{base}-{n}", n + 1
    return name
