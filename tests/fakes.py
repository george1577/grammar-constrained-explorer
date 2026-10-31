"""A fake tokenizer view and fake model, so tests never download a real model, plus the
book-record scripts shared by the grammar-tab and step-trace tests.

The fake model ignores the text so far: at step i it ranks the tokens listed in
``script[i]`` first (in that order), then every other token in vocabulary order.
That lets a test say "the model wants X here" and check whether the grammar lets it.
"""

from grammar_lab.model_boundary import ListScores, TokenizerView
from grammar_lab.pipelines import run_grammar
from grammar_lab.presets import BOOK_RECORD

EOS = "<eos>"
IM_START = "<|im_start|>"

VOCAB = [
    EOS, IM_START,
    "Gabriel", " García", " Garcia", " Márquez", " Marquez",
    " |", "|", "| ", " Love", " in", " the", " Time", " of", " Cholera",
    "1985", "198", "5", "NULL", " NULL",
    "Valluvar", " The", " Tirukkural",
    "\n", " ", "",
]
IDS = {text: i for i, text in enumerate(VOCAB)}


def fake_tokenizer(prompts: list, vocab: list = VOCAB) -> TokenizerView:
    """``prompts`` collects every chat prompt built, so tests can see what was asked.

    ``vocab`` must contain EOS and IM_START. An entry may be ``bytes`` instead of ``str``: a
    byte-level token, possibly holding only part of a multi-byte UTF-8 character, as in
    Qwen's tokenizer. Its text is then decoded the way a real tokenizer does, with "�".
    """
    ids = {text: i for i, text in enumerate(vocab)}
    special = (EOS, IM_START)
    raw = [b"" if t in special else (t if isinstance(t, bytes) else t.encode()) for t in vocab]

    def chat_prompt_ids(messages: list[dict]) -> list[int]:
        prompts.append(messages)
        return [ids[IM_START]]

    return TokenizerView(
        token_texts=[data.decode("utf-8", errors="replace") for data in raw],
        token_bytes=raw,
        special_ids=frozenset({ids[EOS], ids[IM_START]}),
        eos_ids=frozenset({ids[EOS]}),
        chat_prompt_ids=chat_prompt_ids,
        special_names={ids[EOS]: EOS, ids[IM_START]: IM_START},
        decode=lambda token_ids: b"".join(raw[i] for i in token_ids).decode("utf-8", errors="replace"),
    )


class FakeModel:
    def __init__(self, script: list[list], vocab: list = VOCAB):
        self.script = script
        self.vocab = vocab
        self.ids = {text: i for i, text in enumerate(vocab)}
        self.prompts: list[list[dict]] = []
        self.tokenizer = fake_tokenizer(self.prompts, vocab)

    def start(self, prompt_ids: list[int]):
        state = {"step": 0}

        def next_scores(prev):
            preferred = self.script[min(state["step"], len(self.script) - 1)]
            state["step"] += 1
            logits = [-float(i) / 100 for i in range(len(self.vocab))]  # vocabulary order, all low
            for rank, text in enumerate(preferred):
                logits[self.ids[text]] = 10.0 - rank
            return ListScores(logits)

        return next_scores


# ---- tab ①'s book-record preset, shared by the grammar-tab and step-trace tests ----

CHOLERA_SCRIPT = [
    ["Gabriel"],
    [" García", " Garcia"],
    [" Márquez", " Marquez"],
    [" |"],
    [" Love"], [" in"], [" the"], [" Time"], [" of"], [" Cholera"],
    [" |"],
    ["1985"],
    ["\n"],  # the model would keep going; the grammar is already complete
    [" The"],
]


def run_book(model, paragraph="Cholera", max_new_tokens=50):
    return run_grammar(
        model,
        grammar=BOOK_RECORD.grammar,
        root=BOOK_RECORD.root,
        system_prompt=BOOK_RECORD.system_prompt,
        user_text=BOOK_RECORD.inputs[paragraph],
        max_new_tokens=max_new_tokens,
    )


TIRUKKURAL_SCRIPT = [
    ["<eos>", "<|im_start|>", "Valluvar"],  # wants to stop before writing anything
    [" |"], [" The"], [" Tirukkural"], [" |"],
    ["<eos>", "<|im_start|>", "", "NULL"],  # wants to stop with the year still missing
    ["<eos>"],
]
