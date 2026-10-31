"""Grammar-Constrained Explorer: a Streamlit app for seeing how a grammar limits what an LLM may write.

Run with:  uv run streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from grammar_lab.decoding import Run
from grammar_lab.lm import MODELS, LanguageModel
from grammar_lab.pipelines import GrammarRunResult, rule_names, run_grammar, run_regex
from grammar_lab.presets import GRAMMAR_PRESETS, OUTLINES_SNIPPET, REGEX_PRESETS, TRANSFORMERS_CFG_SNIPPET
from trace_view import show_trace

st.set_page_config(page_title="Grammar-Constrained Explorer", page_icon="🧩", layout="wide")

# Results kept in session_state are grammar_lab objects. When that code changes while the app is
# running, Streamlit reloads it but old results keep their old shape. So drop stored results
# whenever the code changes.
RESULT_KEYS = ("t1", "t2")


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


def show_error(error, source: str):
    """A parse error shown inline, with a caret under the place it points at."""
    st.error(str(error))
    lines = source.split("\n")
    if 1 <= error.line <= len(lines):
        line = lines[error.line - 1]
        st.code(f"{line}\n{' ' * (error.column - 1)}^", language=None)


def show_results(res: GrammarRunResult, source: str):
    if res.error:
        show_error(res.error, source)
        return
    c1, c2 = st.columns(2)
    with c1:
        show_run("Unconstrained run", res.unconstrained, constrained=False)
    with c2:
        show_run("Grammar-constrained", res.constrained, constrained=True)


# ---- sidebar ------------------------------------------------------------------------

with st.sidebar:
    st.title("🧩 Grammar-Constrained Explorer")
    model_id = st.selectbox("Model", MODELS, help="0.5B is fast; 1.5B writes better but is ~3× slower.")
    lm = get_model(model_id)
    st.caption(f"Running on **{lm.device}** (set `GCE_DEVICE` to change)")

tab1, tab2 = st.tabs(["① Grammar-constrained generation", "② Regex constraint"])

# ---- tab ①: GBNF grammar ------------------------------------------------------------

with tab1:
    st.header("① Grammar-constrained generation")
    st.markdown(
        "At every step the model scores every token. The grammar then decides which tokens are "
        "**allowed**: only tokens that keep the output a valid start of the grammar can be chosen. "
        "End-of-sequence is allowed only once the output fully matches the **root rule**."
    )
    preset = GRAMMAR_PRESETS[st.selectbox("Preset (from notebook 1)", list(GRAMMAR_PRESETS), key="t1_preset")]
    left, right = st.columns([2, 1])
    with left:
        input_name = st.radio("Input", list(preset.inputs), horizontal=True, key=f"t1_input_{preset.name}")
        user_text = st.text_area("Input text", preset.inputs[input_name], height=130,
                                 key=f"t1_text_{preset.name}_{input_name}")
        grammar = st.text_area("Grammar (GBNF)", preset.grammar, height=180, key=f"t1_grammar_{preset.name}",
                               help="Supported: \"literals\" with escapes, [a-z] classes (and [^...]), "
                                    "( ) groups, |, * + ?, rule references and # comments.")
    with right:
        rules = rule_names(grammar) or [preset.root]
        root = st.selectbox("Root rule", rules, index=rules.index(preset.root) if preset.root in rules else 0,
                            key=f"t1_root_{preset.name}", help="The rule the whole output must match.")
        max_new_tokens = st.slider("Max new tokens", 10, 256, 64, key="t1_max")
    with st.expander("System prompt (from the notebook)"):
        system_prompt = st.text_area("System prompt", preset.system_prompt, height=130,
                                     key=f"t1_system_{preset.name}", label_visibility="collapsed")
    if preset.callout:
        st.info(preset.callout)

    if st.button("Generate both runs", type="primary", key="t1_go"):
        with st.spinner("Generating without and with the grammar…"):
            st.session_state.t1 = (grammar, run_grammar(lm, grammar, root, system_prompt, user_text, max_new_tokens))

    if "t1" in st.session_state:
        used_grammar, res = st.session_state.t1
        show_results(res, used_grammar)
        if not res.error:
            show_trace(res.constrained, lm, key="t1")

    with st.expander("The same thing in the notebook (transformers-cfg)"):
        st.code(TRANSFORMERS_CFG_SNIPPET, language="python")

# ---- tab ②: regex constraint ----------------------------------------------------------

with tab2:
    st.header("② Regex constraint")
    st.markdown(
        "A regex constraint is a grammar too. The regex is converted into a GBNF grammar with a "
        "single rule, `root`, and then decoded with exactly the same engine as tab ①: the whole "
        "output must match the regex. Anchors, lookarounds and backreferences are not supported."
    )
    rpreset = REGEX_PRESETS[st.selectbox("Preset (from notebook 1)", list(REGEX_PRESETS), key="t2_preset")]
    left, right = st.columns([2, 1])
    with left:
        input_name = st.radio("Input", list(rpreset.inputs), horizontal=True, key=f"t2_input_{rpreset.name}")
        user_text = st.text_area("Input text", rpreset.inputs[input_name], height=130,
                                 key=f"t2_text_{rpreset.name}_{input_name}")
    with right:
        max_new_tokens = st.slider("Max new tokens", 10, 256, 64, key="t2_max")
    with st.expander("System prompt (from the notebook)"):
        system_prompt = st.text_area("System prompt", rpreset.system_prompt, height=130,
                                     key=f"t2_system_{rpreset.name}", label_visibility="collapsed")
    rcol, gcol = st.columns(2)
    with rcol:
        regex = st.text_input("Regex", rpreset.regex, key=f"t2_regex_{rpreset.name}")
    with gcol:
        st.markdown("**Generated grammar (GBNF)**")
        res = st.session_state.get("t2")
        if res and res.grammar_text:
            st.code(res.grammar_text, language=None, wrap_lines=True)
        else:
            st.caption("Shown after you generate.")

    if st.button("Generate both runs", type="primary", key="t2_go"):
        with st.spinner("Converting the regex and generating without and with it…"):
            st.session_state.t2 = run_regex(lm, regex, system_prompt, user_text, max_new_tokens)
        st.rerun()

    res = st.session_state.get("t2")
    if res:
        show_results(res, regex if res.error else res.grammar_text)
        if not res.error:
            show_trace(res.constrained, lm, key="t2")

    with st.expander("The same thing in the notebook (Outlines)"):
        st.code(OUTLINES_SNIPPET, language="python")
        st.caption("The notebook's `convert_message_to_prompt` has no `return`; `return prompt` is added "
                   "here. Its cell has no saved output.")
