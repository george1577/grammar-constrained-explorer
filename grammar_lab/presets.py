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
        "**Why does \"García Márquez\" lose its accents?** The `author` rule only allows the characters "
        "in `[a-zA-Z ]`. Tokens such as ` García` contain `í`, which is not in that class, so they are "
        "**disallowed** and the model has to pick the best token that is allowed: ` Garcia`. With Qwen "
        "0.5B it gets worse: after ` M` the model is ~100% sure of `ár`, which is disallowed, and the "
        "best allowed token is `arez`, so the name comes out as \"Garcia Marez\" (and \"cólera\" as "
        "\"cholera\"). A character class limits which characters can appear at all, so a grammar that "
        "validates the format can also quietly change the content."
    ),
)

# Notebook 1, get_expression_that_solves (transformers-cfg). The second question is the
# "grammar too strict" lesson: the answer needs `>`, but the grammar only allows `=`.
ARITHMETIC = GrammarPreset(
    name="Arithmetic expression (notebook grammar: newline after every line)",
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
        "**The model can't stop where it wants to.** This is the notebook's grammar exactly: every line "
        "must end with `\"\\n\"`. With Qwen, step 6 of *Total apples* shows end-of-sequence at about 83% "
        "right after `3 +2 = 5`, but it is **disallowed** because the output only matches the root rule "
        "once the newline is written. The model writes `⏎⏎` instead, has lost its moment to stop, and "
        "keeps adding equations until the token limit. *More apples than oranges?* loops the same way, "
        "as it does in the notebook. Compare the default preset, which puts the newline *between* lines."
    ),
)

# The notebook's arithmetic grammar with one change, made for Qwen: newlines go between lines
# instead of after every line, so the output is complete (and may end) right after an equation.
ARITHMETIC_STOPPABLE = GrammarPreset(
    name="Arithmetic expression",
    system_prompt=ARITHMETIC.system_prompt,
    grammar="""root  ::= line ("\\n" line)*
line  ::= expr "=" ws term
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \\t\\n]*
""",
    root="root",
    inputs=ARITHMETIC.inputs,
    callout=(
        "**A grammar that is too strict.** The \"more apples than oranges?\" question needs a "
        "comparison such as `3 + 2 > 2 + 4`, but the grammar only allows `expr = term`. Every `>` is "
        "**disallowed**, so the model is pushed into an odd equation (Qwen 0.5B writes "
        "`2 +3 -2 -4 = 1`) that is ✅ complete but answers nothing. *Total apples* gives `3 +2 = 5`.\n\n"
        "This grammar differs from the notebook's in one place: the newline goes *between* lines "
        "(`root ::= line (\"\\n\" line)*`) rather than after every line, so the model may stop right "
        "after an equation. The next preset keeps the notebook's grammar and shows what goes wrong."
    ),
)

GRAMMAR_PRESETS = {p.name: p for p in (ARITHMETIC_STOPPABLE, ARITHMETIC, BOOK_RECORD)}


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

# Not in the notebook. Qwen 0.5B follows JSON_BOOK's key names, so this variant leaves the
# "with the keys author, title, and year" out of the prompt to show that the grammar doesn't
# choose the keys either: only the prompt did.
JSON_BOOK_NO_KEYS = PromptPreset(
    name="Book record as JSON, keys not named",
    system_prompt=JSON_BOOK.system_prompt.replace(" with the keys author, title, and year", ""),
    inputs=JSON_BOOK.inputs,
)

JSON_PRESETS = {p.name: p for p in (JSON_BOOK, JSON_BOOK_NO_KEYS)}

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


# ---- structured output (notebook 2b) ------------------------------------------------------------

@dataclass(frozen=True)
class SchemaPreset:
    name: str
    system_prompt: str
    schema: str              # JSON Schema text, editable in the app
    inputs: dict[str, str]


# Notebook 2b's Invoice dataclass written as JSON Schema: fields in the same order, `UKP` (not
# `GBP`) as in the notebook, and `currency` left out of `required` because it has a default.
INVOICE_SCHEMA = """{
  "title": "Invoice",
  "type": "object",
  "properties": {
    "purpose": {"type": "string"},
    "amount": {"type": "number"},
    "currency": {
      "title": "CurrencyEnum",
      "type": "string",
      "enum": ["USD", "UKP", "INR", "EUR"],
      "default": "USD"
    }
  },
  "required": ["purpose", "amount"]
}
"""

# Notebook 2b, parse_invoice_info (pydantic-ai). The prompt never mentions currency: it is
# inferred from the schema alone.
INVOICE = SchemaPreset(
    name="Invoice",
    system_prompt="""
    You will be given a short snippet from an email that represents an invoice.
    Extract the purpose and amount of the invoice.
    """,
    schema=INVOICE_SCHEMA,
    inputs={
        "Taxi ($32.30)": "\nRequesting reimbursement for taxi-ride to airport. I paid $32.30.\n",
        "Clients (€43)": "\nI spent €43 entertaining our French clients.\n",
        "Dinner (₹820)": "\nMy dinner at the airport cost ₹820\n",
    },
)

SCHEMA_PRESETS = {INVOICE.name: INVOICE}

PYDANTIC_AI_SNIPPET = '''from dataclasses import dataclass
from enum import Enum
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

class CurrencyEnum(str, Enum):
    USD = 'USD'
    UKP = 'UKP'
    INR = 'INR'
    EUR = 'EUR'

@dataclass
class Invoice:
    purpose: str
    amount: float
    currency: CurrencyEnum = CurrencyEnum.USD

model = GeminiModel("gemini-1.5-flash", api_key=os.getenv('GEMINI_API_KEY'))

def parse_invoice_info(paragraph: str) -> str:
    system_prompt = """
    You will be given a short snippet from an email that represents an invoice.
    Extract the purpose and amount of the invoice.
    """

    agent = Agent(model,
                  result_type=Invoice,
                  system_prompt=system_prompt)

    response = agent.run_sync(paragraph)
    return response.data
'''
