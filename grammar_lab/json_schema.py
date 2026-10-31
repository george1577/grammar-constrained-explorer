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
from typing import Optional

from grammar_lab.gbnf import GrammarError, gbnf_literal
from grammar_lab.json_grammar import NUMBER_RULES, STRING_RULES, whitespace_rules

ROOT = "root"
PRIMITIVE_RULES = {"string": STRING_RULES, **NUMBER_RULES}


TYPES = ("object", "string", "number", "integer", "boolean")
ANNOTATIONS = {"title", "description", "default"}
OBJECT_KEYWORDS = ANNOTATIONS | {"type", "properties", "required", "additionalProperties"}
VALUE_KEYWORDS = ANNOTATIONS | {"type", "enum"}
JSON_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool}


class SchemaError(GrammarError):
    """A schema that is not valid JSON or uses something outside the supported subset."""


def parse_schema(schema_text: str) -> dict:
    """The schema as a dict, checked against the supported subset (or raise SchemaError)."""
    try:
        schema = json.loads(schema_text)
    except json.JSONDecodeError as error:
        raise SchemaError(f"the schema is not valid JSON: {error.msg}", error.lineno, error.colno) from None
    _Checker(schema_text).top(schema)
    return schema


class _Checker:
    """Walks the schema; each error points at the line of the key it is about."""

    def __init__(self, text: str):
        self.text = text

    def error(self, message: str, path: tuple[str, ...]) -> SchemaError:
        fields = [path[i + 1] for i in range(len(path) - 1) if path[i] == "properties"]
        about = f"field `{' › '.join(fields)}`: " if fields else ""
        return SchemaError(about + message, *self.locate(path))

    def locate(self, path: tuple[str, ...]) -> tuple[int, int]:
        """Line and column of the deepest key of ``path`` found in the text, searching key by key."""
        pos = 0
        for key in path:
            match = re.compile(re.escape(json.dumps(key, ensure_ascii=False)) + r"\s*:").search(self.text, pos)
            if not match:
                break
            pos = match.start()
        line = self.text.count("\n", 0, pos) + 1
        return line, pos - (self.text.rfind("\n", 0, pos) + 1) + 1

    def top(self, schema):
        if not isinstance(schema, dict):
            raise self.error("the schema must be a JSON object (`{...}`)", ())
        if schema.get("type") != "object":
            raise self.error('the top level must be `"type": "object"` with `properties`', ("type",))
        self.object(schema, ())

    def object(self, schema: dict, path: tuple[str, ...]):
        self.keywords(schema, OBJECT_KEYWORDS, path)
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise self.error("`properties` must be an object mapping field names to schemas", path + ("properties",))
        if schema.get("additionalProperties", False) is not False:
            raise self.error("`additionalProperties` can only be false: only the listed fields can appear",
                             path + ("additionalProperties",))
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
            raise self.error("`required` must be a list of field names", path + ("required",))
        for name in required:
            if name not in properties:
                raise self.error(f"`required` lists `{name}`, which is not in `properties`", path + ("required",))
        for name, field in properties.items():
            self.field(field, path + ("properties", name))

    def field(self, field, path: tuple[str, ...]):
        if not isinstance(field, dict):
            raise self.error("each field's schema must be a JSON object", path)
        kind = field.get("type")
        if isinstance(kind, list):
            raise self.error("a list of types is not supported: give a single `type`", path + ("type",))
        if kind is not None and kind not in TYPES:
            raise self.error(f"type `{kind}` is not supported (use {', '.join(TYPES)})", path + ("type",))
        if kind == "object":
            self.object(field, path)
            return
        self.keywords(field, VALUE_KEYWORDS, path)
        if "enum" in field:
            self.enum(field["enum"], kind, path + ("enum",))
        elif kind is None:
            raise self.error("needs a `type` or an `enum`", path)

    def enum(self, values, kind, path: tuple[str, ...]):
        if not isinstance(values, list) or not values:
            raise self.error("`enum` needs at least one value", path)
        expected = JSON_TYPES.get(kind)
        for value in values:
            shown = json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list)):
                raise self.error(f"enum value `{shown}` is not supported: use strings, numbers, booleans or null",
                                 path)
            number_given_bool = kind in ("number", "integer") and isinstance(value, bool)
            if expected and (not isinstance(value, expected) or number_given_bool):
                raise self.error(f"enum value `{shown}` is not a {kind}", path)

    def keywords(self, schema: dict, allowed: set[str], path: tuple[str, ...]):
        for keyword in schema:
            if keyword not in allowed:
                raise self.error(f"`{keyword}` is not supported in this app's JSON Schema subset "
                                 f"(supported here: {', '.join(sorted(allowed))})", path + (keyword,))


def schema_to_gbnf(schema_text: str, limit_whitespace: bool = True) -> str:
    """GBNF text whose root rule matches exactly the JSON the schema describes, or raise SchemaError.

    ``limit_whitespace`` chooses the ``ws`` rule, as in JSON mode (see json_grammar.py).
    """
    schema = parse_schema(schema_text)
    rules: dict[str, str] = {}
    used: set[str] = set()
    _object_rule(ROOT, schema, rules, used)
    shared = "".join(PRIMITIVE_RULES[name] for name in PRIMITIVE_RULES if name in used)
    return "".join(f"{name} ::= {body}\n" for name, body in rules.items()) + shared + whitespace_rules(limit_whitespace)


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


# ---- read-only dataclass view -----------------------------------------------------------

PYTHON_TYPES = {"string": "str", "number": "float", "integer": "int", "boolean": "bool"}


def schema_to_dataclass(schema_text: str) -> str:
    """Python source for dataclasses (and Enums) equivalent to the schema, for reading only.

    Fields keep schema order. A field outside ``required`` with no default is shown as
    ``T | None = None``; in the grammar it is still required.
    """
    schema = parse_schema(schema_text)
    blocks: list[str] = []
    imports: set[str] = set()
    _dataclass(schema.get("title") or "Model", schema, blocks, imports)
    header = ["from dataclasses import dataclass"]
    header += [line for line in ("from enum import Enum", "from typing import Literal") if line in imports]
    return "\n".join(header) + "\n\n\n" + "\n\n".join(blocks)


def _dataclass(name: str, schema: dict, blocks: list[str], imports: set[str]) -> str:
    required = set(schema.get("required", []))
    lines = ["@dataclass", f"class {name}:"]
    for key, field in schema.get("properties", {}).items():
        annotation, default = _annotation(key, field, blocks, imports)
        if default is None and key not in required:
            annotation, default = f"{annotation} | None", "None"
        lines.append(f"    {key}: {annotation}" + (f" = {default}" if default is not None else ""))
    if len(lines) == 2:
        lines.append("    pass")
    blocks.append("\n".join(lines) + "\n")
    return name


def _annotation(key: str, field: dict, blocks: list[str], imports: set[str]) -> tuple[str, Optional[str]]:
    """The field's type annotation and default (as Python source, or None if it has no default)."""
    default = repr(field["default"]) if "default" in field else None
    if field.get("type") == "object":
        return _dataclass(field.get("title") or _camel(key), field, blocks, imports), default
    if "enum" in field and all(isinstance(v, str) for v in field["enum"]):
        imports.add("from enum import Enum")
        enum_name = field.get("title") or _camel(key) + "Enum"
        members = {value: _member_name(value) for value in field["enum"]}
        body = "\n".join(f"    {member} = {value!r}" for value, member in members.items())
        blocks.append(f"class {enum_name}(str, Enum):\n{body}\n")
        if field.get("default") in members:
            default = f"{enum_name}.{members[field['default']]}"
        return enum_name, default
    if "enum" in field:
        imports.add("from typing import Literal")
        return f"Literal[{', '.join(repr(v) for v in field['enum'])}]", default
    return PYTHON_TYPES[field["type"]], default


def _camel(key: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[^A-Za-z0-9]+", key) if part) or "Field"


def _member_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "EMPTY"
    return f"_{name}" if name[0].isdigit() else name
