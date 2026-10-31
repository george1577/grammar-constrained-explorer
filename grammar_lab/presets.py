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
)

GRAMMAR_PRESETS = {BOOK_RECORD.name: BOOK_RECORD}
