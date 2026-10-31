"""The only module that touches torch and transformers. Implements the model boundary."""

import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grammar_lab.model_boundary import NextScoresFn, TokenizerView

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]


def pick_device() -> str:
    # Apple's MPS backend is opt-in: recent torch builds crash on older macOS versions,
    # and a 0.5B model is fast enough on CPU. Set GCE_DEVICE=mps (or cuda/cpu) to override.
    if os.environ.get("GCE_DEVICE"):
        return os.environ["GCE_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


class TensorScores:
    """NextTokenScores over a 1-D logits tensor."""

    def __init__(self, logits: torch.Tensor):
        self._logits = logits
        self._ranked: Optional[list[int]] = None
        self._probs: Optional[torch.Tensor] = None

    def ranked_ids(self) -> list[int]:
        if self._ranked is None:
            self._ranked = torch.argsort(self._logits, descending=True).tolist()
        return self._ranked

    def logit(self, token_id: int) -> float:
        return float(self._logits[token_id])

    def prob(self, token_id: int) -> float:
        if self._probs is None:
            self._probs = torch.softmax(self._logits, dim=-1)
        return float(self._probs[token_id])


def build_tokenizer_view(tokenizer, vocab_size: int, eos_ids: set[int]) -> TokenizerView:
    """Decode every token once. Ids past the tokenizer's vocabulary (padding rows of the
    model's output layer) and added tokens such as <|im_end|> count as special."""
    special = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder) | set(eos_ids)
    special |= set(range(len(tokenizer), vocab_size))
    texts = [
        "" if i in special else tokenizer.decode([i], clean_up_tokenization_spaces=False)
        for i in range(vocab_size)
    ]

    def chat_prompt_ids(messages: list[dict]) -> list[int]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    return TokenizerView(
        token_texts=texts,
        special_ids=frozenset(special),
        eos_ids=frozenset(eos_ids),
        chat_prompt_ids=chat_prompt_ids,
        decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        special_names={i: (tokenizer.convert_ids_to_tokens(i) if i < len(tokenizer) else None) or f"<pad {i}>"
                       for i in special},
    )


class LanguageModel:
    def __init__(self, model_id: str = MODELS[0]):
        self.model_id = model_id
        self.device = pick_device()
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device)
        self.model.eval()

        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        vocab_size = self.model.get_output_embeddings().weight.shape[0]
        self.tokenizer = build_tokenizer_view(hf_tokenizer, vocab_size, eos_ids)

    def start(self, prompt_ids: list[int]) -> NextScoresFn:
        """A next_scores(prev_token) function that reuses the KV cache between steps."""
        state = {"past": None}

        @torch.no_grad()
        def next_scores(prev: Optional[int]) -> TensorScores:
            ids = prompt_ids if prev is None else [prev]
            out = self.model(
                input_ids=torch.tensor([ids], device=self.device),
                past_key_values=state["past"],
                use_cache=True,
            )
            state["past"] = out.past_key_values
            return TensorScores(out.logits[0, -1].float().cpu())

        return next_scores
