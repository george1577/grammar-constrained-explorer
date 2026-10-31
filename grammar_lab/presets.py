"""Presets taken from the Grammar chapter's notebooks (examples/02_grammar), stored as data.

Grammars keep GBNF escape sequences as written (e.g. "\\n"); the parser decodes them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GrammarPreset:
    name: str
    system_prompt: str
    grammar: str
    root: str
    inputs: dict[str, str]  # input name -> user text
    callout: str = ""       # a lesson shown next to this preset (Markdown)


CHOLERA = """
Love in the Time of Cholera (Spanish: El amor en los tiempos del cólera) is a novel written in Spanish
by Colombian Nobel Prize-winning author Gabriel García Márquez and published in 1985.
"""

# The notebook's text has a trailing space after "5th century CE."; kept explicit here.
TIRUKKURAL = (
    "\nThe Tirukkural (Tamil: திருக்குறள், lit. 'sacred verses')\n"
    "is a classic Tamil language text whose authorship is traditionally attributed to Valluvar,\n"
    "also known in full as Thiruvalluvar. The text has been dated variously from 300 BCE to 5th century CE. \n"
    "The traditional accounts describe it as the last work of the third Sangam, but linguistic analysis\n"
    "suggests a later date of 450 to 500 CE and that it was composed after the Sangam period.\n"
)

# Notebook 1, parse_book_info (transformers-cfg)
BOOK_RECORD = GrammarPreset(
    name="Book record (author | title | year)",
    system_prompt="""
    You will be given a short paragraph about a book.
    Extract the author, title, and publication year of the book.
    Return the result as author | title | year
    If any piece of information is not found, fill the spot with NULL
    """,
    grammar="""record ::= author separator title separator year
author ::= [a-zA-Z ]* | unk
title ::= [a-zA-Z ]* | unk
year ::= [1-2][0-9][0-9][0-9] | unk
unk ::= "NULL"
separator ::= "|"
""",
    root="record",
    inputs={"Cholera": CHOLERA, "Tirukkural": TIRUKKURAL},
    callout=(
        "**Why does \"García Márquez\" become \"Garcia Marquez\"?** The `author` rule only allows "
        "the characters in `[a-zA-Z ]`. Tokens such as ` García` contain `í`, which is not in that "
        "class, so they are **disallowed** and the model has to pick the best token that is allowed: "
        "` Garcia`. A character class limits which characters can appear at all, so a grammar that "
        "validates the format can also quietly change the content."
    ),
)

# Notebook 1, get_expression_that_solves (transformers-cfg). The second question is the
# "grammar too strict" lesson: the answer needs `>`, but the grammar only allows `=`.
ARITHMETIC = GrammarPreset(
    name="Arithmetic expression",
    system_prompt="""
    You are a math instructor. I will ask you a math question.
    Respond with the mathematical expression that can be used to solve the problem.
    """,
    grammar="""root  ::= (expr "=" ws term "\\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \\t\\n]*
""",
    root="root",
    inputs={
        "Total apples": """
Bill has 3 apples and 2 oranges.
Mae has 2 apples and 4 oranges.
How many total apples do Bill and Mae have?
""",
        "More apples than oranges?": """
Bill has 3 apples and 2 oranges.
Mae has 2 apples and 4 oranges.
Do Bill and Mae have more apples than oranges?
""",
    },
    callout=(
        "**A grammar that is too strict.** The \"more apples than oranges?\" question needs a "
        "comparison such as `3 + 2 > 2 + 4`, but the grammar only allows `expr = term`. Every `>` is "
        "**disallowed**, so the model is pushed into odd equations and often loops until the token "
        "limit, giving ⚠️ incomplete output."
    ),
)

GRAMMAR_PRESETS = {p.name: p for p in (ARITHMETIC, BOOK_RECORD)}


@dataclass(frozen=True)
class RegexPreset:
    name: str
    system_prompt: str
    regex: str
    inputs: dict[str, str]


# Notebook 1, the Outlines version of parse_book_info. Unlike the GBNF: `+` (not `*`) and
# a bar with a space on each side.
BOOK_REGEX = RegexPreset(
    name="Book record (author | title | year)",
    system_prompt=BOOK_RECORD.system_prompt,
    regex=r"([a-zA-Z ]+|NULL) \| ([a-zA-Z ]+|NULL) \| ([1-2][0-9][0-9][0-9]|NULL)",
    inputs={"Cholera": CHOLERA, "Tirukkural": TIRUKKURAL},
)

REGEX_PRESETS = {BOOK_REGEX.name: BOOK_REGEX}


# ---- read-only notebook snippets ------------------------------------------------------------

TRANSFORMERS_CFG_SNIPPET = '''from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

grammar = IncrementalGrammarConstraint(grammar_str, "root", pipe.tokenizer)
grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

input_message = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": math_problem}
]
results = pipe(input_message,
               max_new_tokens=256,
               do_sample=False,
               logits_processor=[grammar_processor])
'''

# The notebook's convert_message_to_prompt has no `return` (so the generator gets None) and
# writes ChatML tags that Phi-3 does not use; the snippet adds the missing `return prompt`.
OUTLINES_SNIPPET = '''import outlines
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

def convert_message_to_prompt(messages):
    prompt = ""
    for message in messages:
        prompt += f"""
<|im_start|>{message['role']}
{message['content']}
<|im_end|>
        """
    prompt += """
    <|im_start|>assistant
    """
    return prompt  # missing in the notebook

generator = outlines.generate.regex(
    model,
    r"([a-zA-Z ]+|NULL) \\| ([a-zA-Z ]+|NULL) \\| ([1-2][0-9][0-9][0-9]|NULL)",
    sampler=outlines.samplers.greedy(),
)
structured = generator(convert_message_to_prompt(input_message), max_tokens=30)
'''


# ---- JSON mode (notebook 2a) ----------------------------------------------------------------

@dataclass(frozen=True)
class PromptPreset:
    name: str
    system_prompt: str
    inputs: dict[str, str]


# Notebook 2a, parse_book_info (OpenAI JSON mode). Its message role is "developer", OpenAI's
# newer name for "system". The prompt differs from notebook 1 only in its third line.
JSON_BOOK = PromptPreset(
    name="Book record as JSON",
    system_prompt="""
    You will be given a short paragraph about a book.
    Extract the author, title, and publication year of the book.
    Return the result as JSON with the keys author, title, and year.
    If any piece of information is not found, fill the spot with NULL
    """,
    inputs={"Cholera": CHOLERA, "Tirukkural": TIRUKKURAL},
)

JSON_PRESETS = {JSON_BOOK.name: JSON_BOOK}

OPENAI_JSON_MODE_SNIPPET = '''from openai import OpenAI
client = OpenAI()

input_message = [
    {"role": "developer", "content": system_prompt},
    {"role": "user", "content": paragraph}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=input_message,
    response_format={"type": "json_object"}
)
return response.choices[0].message.content
'''
