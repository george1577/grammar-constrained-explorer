"""A fake tokenizer view and fake model, so tests never download a real model.

The fake model ignores the text so far: at step i it ranks the tokens listed in
``script[i]`` first (in that order), then every other token in vocabulary order.
That lets a test say "the model wants X here" and check whether the grammar lets it.
"""

from grammar_lab.model_boundary import ListScores, TokenizerView

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


def fake_tokenizer(prompts: list) -> TokenizerView:
    """``prompts`` collects every chat prompt built, so tests can see what was asked."""

    def chat_prompt_ids(messages: list[dict]) -> list[int]:
        prompts.append(messages)
        return [IDS[IM_START]]

    return TokenizerView(
        token_texts=["" if t in (EOS, IM_START) else t for t in VOCAB],
        special_ids=frozenset({IDS[EOS], IDS[IM_START]}),
        eos_ids=frozenset({IDS[EOS]}),
        chat_prompt_ids=chat_prompt_ids,
        special_names={IDS[EOS]: EOS, IDS[IM_START]: IM_START},
        decode=lambda ids: "".join(VOCAB[i] for i in ids if VOCAB[i] not in (EOS, IM_START)),
    )


class FakeModel:
    def __init__(self, script: list[list[str]]):
        self.script = script
        self.prompts: list[list[dict]] = []
        self.tokenizer = fake_tokenizer(self.prompts)

    def start(self, prompt_ids: list[int]):
        state = {"step": 0}

        def next_scores(prev):
            preferred = self.script[min(state["step"], len(self.script) - 1)]
            state["step"] += 1
            logits = [-float(i) / 100 for i in range(len(VOCAB))]  # vocabulary order, all low
            for rank, text in enumerate(preferred):
                logits[IDS[text]] = 10.0 - rank
            return ListScores(logits)

        return next_scores
