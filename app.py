"""Grammar-Constrained Explorer: a Streamlit app for seeing how a grammar limits what an LLM may write.

Run with:  uv run streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from grammar_lab.decoding import Run
from grammar_lab.lm import MODELS, LanguageModel
from grammar_lab.pipelines import run_grammar
from grammar_lab.presets import BOOK_RECORD
from trace_view import show_trace

st.set_page_config(page_title="Grammar-Constrained Explorer", page_icon="🧩", layout="wide")

# Results kept in session_state are grammar_lab objects. When that code changes while the app is
# running, Streamlit reloads it but old results keep their old shape. So drop stored results
# whenever the code changes.
RESULT_KEYS = ("t1",)


def code_version() -> str:
    files = sorted((Path(__file__).parent / "grammar_lab").glob("*.py"))
    return "|".join(f"{f.name}:{f.stat().st_mtime_ns}" for f in files)


if st.session_state.get("_code_version") != (version := code_version()):
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)
    st.session_state["_code_version"] = version


@st.cache_resource(show_spinner="Loading model (first run downloads it)…")
def get_model(model_id: str) -> LanguageModel:
    return LanguageModel(model_id)


def visible(text: str) -> str:
    return text.replace("\n", "⏎\n")


def show_run(title: str, run: Run, constrained: bool):
    st.markdown(f"**{title}**")
    st.code(visible(run.text) or "(empty)", language=None, wrap_lines=True)
    if constrained:
        if run.complete:
            st.success("✅ Complete: the output fully matches the root rule.")
        else:
            st.warning(f"⚠️ Incomplete output: stopped ({run.stop_reason}) before matching the root rule.")
    st.caption(f"{len(run.token_ids)} tokens · stopped: {run.stop_reason}")


# ---- sidebar ------------------------------------------------------------------------

with st.sidebar:
    st.title("🧩 Grammar-Constrained Explorer")
    model_id = st.selectbox("Model", MODELS, help="0.5B is fast; 1.5B writes better but is ~3× slower.")
    lm = get_model(model_id)
    st.caption(f"Running on **{lm.device}** (set `GCE_DEVICE` to change)")

(tab1,) = st.tabs(["① Grammar-constrained generation"])

# ---- tab ①: GBNF grammar ------------------------------------------------------------

with tab1:
    st.header("① Grammar-constrained generation")
    st.markdown(
        "At every step the model scores every token. The grammar then decides which tokens are "
        "**allowed**: only tokens that keep the output a valid start of the grammar can be chosen. "
        "End-of-sequence is allowed only once the output fully matches the **root rule**."
    )
    preset = BOOK_RECORD
    left, right = st.columns([2, 1])
    with left:
        paragraph = st.radio("Paragraph", list(preset.inputs), horizontal=True, key="t1_input")
        st.text_area("Input text", preset.inputs[paragraph], height=130, disabled=True, key=f"t1_text_{paragraph}")
    with right:
        max_new_tokens = st.slider("Max new tokens", 10, 256, 64, key="t1_max")
        st.markdown(f"Root rule: `{preset.root}`")
    st.markdown("**Grammar (GBNF)**")
    st.code(preset.grammar, language=None)
    with st.expander("System prompt (from the notebook)"):
        st.code(preset.system_prompt, language=None)

    if st.button("Generate both runs", type="primary", key="t1_go"):
        with st.spinner("Generating without and with the grammar…"):
            st.session_state.t1 = run_grammar(
                lm, preset.grammar, preset.root, preset.system_prompt, preset.inputs[paragraph], max_new_tokens
            )

    res = st.session_state.get("t1")
    if res:
        if res.error:
            st.error(f"Grammar error: {res.error}")
        else:
            c1, c2 = st.columns(2)
            with c1:
                show_run("Unconstrained run", res.unconstrained, constrained=False)
            with c2:
                show_run("Grammar-constrained", res.constrained, constrained=True)
            show_trace(res.constrained, lm, key="t1")
