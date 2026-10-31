# Grammar-Constrained Explorer

An interactive Streamlit app for **seeing** how a formal grammar limits which tokens a language model may produce, step by step. It runs a small open model (Qwen2.5) on your laptop.

It's a companion to the *Grammar* examples in
[Generative AI Design Patterns](https://github.com/lakshmanok/generative-ai-design-patterns)
(`examples/02_grammar`), covering notebooks 1, 2a and 2b.

> 🚧 Work in progress. Terms are defined in [GLOSSARY.md](GLOSSARY.md); design decisions live in [docs/adr/](docs/adr/).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), about 2 GB of free RAM, and an internet connection the first time (the model, about 1 GB, is downloaded from Hugging Face and cached).

```bash
uv sync
uv run streamlit run app.py
```

The first page load downloads and loads the model, which takes a little while; later runs reuse the cache.

Run the tests (no model download needed; they use a fake tokenizer and a fake model):

```bash
uv run pytest
```

### Configuration

| Setting | How | Default |
|---|---|---|
| Model | Sidebar dropdown | `Qwen/Qwen2.5-0.5B-Instruct` (`Qwen/Qwen2.5-1.5B-Instruct` is better but about 3× slower) |
| Device | `GCE_DEVICE=mps uv run streamlit run app.py` (`cpu`, `cuda` or `mps`) | `cuda` if available, otherwise `cpu` |

> MPS is not chosen automatically: recent torch builds can crash on older macOS versions, and the 0.5B model is fast enough on CPU.
