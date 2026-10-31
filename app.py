"""Grammar-Constrained Explorer: a Streamlit app for seeing how a grammar limits what an LLM may write.

Run with:  uv run streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from grammar_lab.decoding import TOKEN_LIMIT, Run
from grammar_lab.json_grammar import json_gbnf
from grammar_lab.lm import MODELS, LanguageModel
from grammar_lab.pipelines import (GrammarRunResult, preview_schema, rule_names, run_grammar, run_json_mode, run_regex,
                                  run_structured)
from grammar_lab.presets import (GRAMMAR_PRESETS, JSON_PRESETS, OPENAI_JSON_MODE_SNIPPET, OUTLINES_SNIPPET,
                                 PYDANTIC_AI_SNIPPET, REGEX_PRESETS, SCHEMA_PRESETS, TRANSFORMERS_CFG_SNIPPET)
from trace_view import show_trace

st.set_page_config(page_title="Grammar-Constrained Explorer", page_icon="🧩", layout="wide")

# Results kept in session_state are grammar_lab objects. When that code changes while the app is
# running, Streamlit reloads it but old results keep their old shape. So drop stored results
# whenever the code changes.
RESULT_KEYS = ("t1", "t2", "t3", "t4")


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
            if run.stop_reason == TOKEN_LIMIT:
                st.caption("…but the model never chose to stop: it kept writing until the token limit "
                           "(for example, a whitespace loop).")
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

tab1, tab2, tab3, tab4 = st.tabs(["① Grammar-constrained generation", "② Regex constraint", "③ JSON mode",
                                  "④ Structured output"])

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

# ---- tab ③: JSON mode ---------------------------------------------------------------------

with tab3:
    st.header("③ JSON mode")
    st.markdown(
        "JSON mode is a grammar too: one that allows **any** valid JSON object. The output is "
        "guaranteed to parse, but the grammar says nothing about which keys appear or what type each "
        "value has. Only the prompt asks for `author`, `title` and `year`."
    )
    jpreset = JSON_PRESETS[st.selectbox("Preset (from notebook 2a)", list(JSON_PRESETS), key="t3_preset")]
    left, right = st.columns([2, 1])
    with left:
        input_name = st.radio("Input", list(jpreset.inputs), horizontal=True, key=f"t3_input_{jpreset.name}")
        user_text = st.text_area("Input text", jpreset.inputs[input_name], height=130,
                                 key=f"t3_text_{jpreset.name}_{input_name}")
    with right:
        limit_whitespace = st.toggle(
            "Limit whitespace", value=True, key="t3_limit_ws",
            help="On: each gap allows at most one space, or one newline plus a short indent. Off: unlimited "
                 "whitespace, so a model that keeps choosing newlines or spaces can loop until the token limit.")
        max_new_tokens = st.slider("Max new tokens", 10, 256, 96, key="t3_max")
    with st.expander("System prompt (from the notebook)"):
        system_prompt = st.text_area("System prompt", jpreset.system_prompt, height=130,
                                     key=f"t3_system_{jpreset.name}", label_visibility="collapsed")
    with st.expander("Generated JSON grammar (GBNF)"):
        st.code(json_gbnf(limit_whitespace), language=None, wrap_lines=True)
    st.info(
        "**Keys and value types are not controlled.** Inside an object the grammar allows any string as a "
        "key and any JSON value after it, so the model may pick its own keys (`writer` instead of "
        "`author`) or write the string `\"NULL\"` where JSON's `null` was meant. JSON strings allow any "
        "character, so accents such as \"García Márquez\" survive here, unlike tab ①'s `[a-zA-Z ]`. "
        "Whitespace is optional everywhere in JSON: turn **Limit whitespace** off to watch a model that "
        "favours newlines or spaces keep writing them, because the grammar never forces it to stop."
    )

    if st.button("Generate both runs", type="primary", key="t3_go"):
        with st.spinner("Generating without and with the JSON grammar…"):
            st.session_state.t3 = run_json_mode(lm, system_prompt, user_text, limit_whitespace, max_new_tokens)

    res = st.session_state.get("t3")
    if res:
        show_results(res, res.grammar_text)
        show_trace(res.constrained, lm, key="t3")

    with st.expander("The same thing in the notebook (OpenAI JSON mode)"):
        st.code(OPENAI_JSON_MODE_SNIPPET, language="python")
        st.caption("The notebook sends the system prompt with role `developer`, OpenAI's newer name for "
                   "`system`; this app sends it as `system`.")

# ---- tab ④: structured output ---------------------------------------------------------------

with tab4:
    st.header("④ Structured output")
    st.markdown(
        "Structured output is a grammar built from a schema. Each object's keys are spelled out in "
        "schema order and each value must have the field's type, so the model only fills in the "
        "values. Enums become literal alternatives: only their values can be written."
    )
    spreset = SCHEMA_PRESETS[st.selectbox("Preset (from notebook 2b)", list(SCHEMA_PRESETS), key="t4_preset")]
    left, right = st.columns([2, 1])
    with left:
        input_name = st.radio("Input", list(spreset.inputs), horizontal=True, key=f"t4_input_{spreset.name}")
        user_text = st.text_area("Input text", spreset.inputs[input_name], height=100,
                                 key=f"t4_text_{spreset.name}_{input_name}")
    with right:
        max_new_tokens = st.slider("Max new tokens", 10, 256, 64, key="t4_max")
    with st.expander("System prompt (from the notebook)"):
        system_prompt = st.text_area("System prompt", spreset.system_prompt, height=110,
                                     key=f"t4_system_{spreset.name}", label_visibility="collapsed")

    scol, vcol = st.columns(2)
    with scol:
        schema = st.text_area(
            "JSON Schema", spreset.schema, height=380, key=f"t4_schema_{spreset.name}",
            help="Supported: type object, string, number, integer or boolean; properties; required; enum; "
                 "nested objects; and title, description and default.")
    preview = preview_schema(schema)
    with vcol:
        if preview.error:
            st.markdown("**Schema error**")
            show_error(preview.error, schema)
        else:
            view_tab, gbnf_tab = st.tabs(["Python dataclass view", "Generated grammar (GBNF)"])
            with view_tab:
                st.code(preview.dataclass_view, language="python")
                st.caption("Read-only: the schema as notebook 2b's kind of dataclass.")
            with gbnf_tab:
                st.code(preview.grammar_text, language=None, wrap_lines=True)
    st.info(
        "**Why is a field with a default still required, and why are fields in schema order?** A grammar "
        "either allows a key at a position or it doesn't. Writing every field, in the order the schema "
        "lists them, gives one simple sequence of keys. Letting fields be left out, or come in any order, "
        "would multiply the alternatives the grammar has to spell out. So `currency` must be written even "
        "though it defaults to `USD`, and `purpose` always comes first. The model has to infer the "
        "currency, since the system prompt never mentions it. Field order also matters for quality: "
        "the model writes earlier fields first and can use them when choosing later ones."
    )

    if st.button("Generate both runs", type="primary", key="t4_go", disabled=bool(preview.error)):
        with st.spinner("Converting the schema and generating without and with it…"):
            st.session_state.t4 = (schema, run_structured(lm, schema, system_prompt, user_text, max_new_tokens))

    if "t4" in st.session_state:
        used_schema, res = st.session_state.t4
        show_results(res, used_schema if res.error else res.grammar_text)
        if not res.error:
            show_trace(res.constrained, lm, key="t4")

    with st.expander("The same thing in the notebook (pydantic-ai + Gemini)"):
        st.code(PYDANTIC_AI_SNIPPET, language="python")
        st.caption("pydantic-ai 0.0.53 as in the notebook: later versions renamed `result_type=` to "
                   "`output_type=` and `response.data` to `response.output`.")
