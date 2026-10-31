"""GBNF grammars for JSON: any valid JSON object (JSON mode), and the shared building blocks
(strings, numbers, whitespace) that the schema converter reuses.

JSON mode only says "the output is a JSON object": which keys appear and what type each value
has are left to the model. Whitespace between tokens is optional in JSON, so an unlimited
``ws`` rule lets a model keep writing spaces or newlines forever (a *whitespace loop*). With
``limit_whitespace`` each ``ws`` allows at most one space, or one newline plus a short indent.
"""

ROOT = "root"
MAX_INDENT = 12  # spaces or tabs allowed after a newline when whitespace is limited

UNLIMITED_WS = "ws     ::= [ \\t\\n]*\n"
LIMITED_WS = ('ws     ::= (" " | "\\n" indent)?\n'
              "indent ::= " + " ".join(["[ \\t]?"] * MAX_INDENT) + "\n")

# A JSON string: any character except `"`, `\` and control characters, or an escape.
STRING_RULES = """string ::= "\\"" char* "\\"" ws
char   ::= [^"\\\\\\n\\r\\t] | "\\\\" (["\\\\/bfnrt] | "u" hex hex hex hex)
hex    ::= [0-9a-fA-F]
"""


def whitespace_rules(limit_whitespace: bool) -> str:
    return LIMITED_WS if limit_whitespace else UNLIMITED_WS


def json_gbnf(limit_whitespace: bool = True) -> str:
    """GBNF text for any JSON object (with any keys and any values inside)."""
    return (
        f"{ROOT}   ::= object\n"
        'value  ::= object | array | string | number | ("true" | "false" | "null") ws\n'
        'object ::= "{" ws (member ("," ws member)*)? "}" ws\n'
        'member ::= string ":" ws value\n'
        'array  ::= "[" ws (value ("," ws value)*)? "]" ws\n'
        + STRING_RULES
        + 'number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws\n'
        + whitespace_rules(limit_whitespace)
    )
