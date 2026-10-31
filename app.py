"""Grammar-Constrained Explorer: a Streamlit app for seeing how a grammar limits what an LLM may write.

Run with:  uv run streamlit run app.py
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Optional

import streamlit as st

from grammar_lab.lm import MODELS, LanguageModel
from grammar_lab.pipelines import (GrammarError, GrammarRunResult, Run, preview_json_mode, preview_regex,
                                   preview_schema, rule_names, run_grammar, run_json_mode, run_regex, run_structured)
from grammar_lab.presets import (GRAMMAR_PRESETS, JSON_PRESETS, OPENAI_JSON_MODE_SNIPPET, OUTLINES_SNIPPET,
                                 PYDANTIC_AI_SNIPPET, REGEX_PRESETS, SCHEMA_PRESETS, TRANSFORMERS_CFG_SNIPPET)
from trace_view import show_trace

st.set_page_config(page_title="Grammar-Constrained Explorer", page_icon="🧩", layout="wide")


class Tab(StrEnum):
    """The four tabs. The value prefixes every widget and session-state key the tab owns."""
    GRAMMAR = "grammar"
    REGEX = "regex"
    JSON_MODE = "json_mode"
    STRUCTURED = "structured"


@dataclass
class TabResult:
    """What a tab keeps in session state after Generate (stored under the tab's key)."""
    result: GrammarRunResult
    source: str  # the grammar, regex or schema as submitted: error carets point into it


@dataclass
class TabInputs:
    preset: object           # the chosen preset (its fields differ by tab)
    user_text: str
    max_new_tokens: int
    system_prompt: str
    left: object             # the input column, for tab-specific widgets below the input text
    right: object            # the settings column, for tab-specific widgets below the slider


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
            st.success("✅ Complete: the output fully matches the root rule and the model stopped by itself.")
        else:
            st.warning(f"⚠️ Incomplete output: stopped ({run.stop_reason}) before the model chose to end on a "
                       "full match of the root rule. Hitting the token limit is always incomplete, even if the "
                       "output so far matches (for example, a whitespace loop).")
    st.caption(f"{len(run.token_ids)} tokens · stopped: {run.stop_reason}")


def show_error(error: GrammarError, source: str):
    """A parse error shown inline, with a caret under the place it points at."""
    st.error(str(error))
    lines = source.split("\n")
    if 1 <= error.line <= len(lines):
        line = lines[error.line - 1]
        st.code(f"{line}\n{' ' * (error.column - 1)}^", language=None)


def show_results(tab_result: TabResult):
    result = tab_result.result
    if result.error:
        show_error(result.error, tab_result.source)
        return
    unconstrained_column, constrained_column = st.columns(2)
    with unconstrained_column:
        show_run("Unconstrained run", result.unconstrained, constrained=False)
    with constrained_column:
        show_run("Grammar-constrained", result.constrained, constrained=True)


def tab_inputs(tab: Tab, presets: dict, notebook: str, default_max_tokens: int = 64, text_height: int = 130,
               system_height: int = 130) -> TabInputs:
    """The layout every tab shares: preset, input choice and text, max-tokens slider, system prompt."""
    preset = presets[st.selectbox(f"Preset (from notebook {notebook})", list(presets), key=f"{tab}_preset")]
    left, right = st.columns([2, 1])
    with left:
        input_name = st.radio("Input", list(preset.inputs), horizontal=True, key=f"{tab}_input_{preset.name}")
        user_text = st.text_area("Input text", preset.inputs[input_name], height=text_height,
                                 key=f"{tab}_text_{preset.name}_{input_name}")
    with right:
        max_new_tokens = st.slider("Max new tokens", 10, 256, default_max_tokens, key=f"{tab}_max")
    with st.expander("System prompt (from the notebook)"):
        system_prompt = st.text_area("System prompt", preset.system_prompt, height=system_height,
                                     key=f"{tab}_system_{preset.name}", label_visibility="collapsed")
    return TabInputs(preset, user_text, max_new_tokens, system_prompt, left, right)


def generate_and_show(tab: Tab, run: Callable[[], GrammarRunResult], source: str, spinner: str,
                      disabled: bool = False):
    """The Generate button, then the stored result and its step trace."""
    if st.button("Generate both runs", type="primary", key=f"{tab}_go", disabled=disabled):
        with st.spinner(spinner):
            st.session_state[tab] = TabResult(run(), source)
    tab_result: Optional[TabResult] = st.session_state.get(tab)
    if tab_result:
        show_results(tab_result)
        if not tab_result.result.error:
            show_trace(tab_result.result.constrained, lm, key=tab)


def notebook_snippet(title: str, code: str, caption: Optional[str] = None):
    with st.expander(f"The same thing in the notebook ({title})"):
        st.code(code, language="python")
        if caption:
            st.caption(caption)


def show_generated_grammar(preview: GrammarRunResult, source: str):
    """A preview's generated GBNF, or its error with a caret."""
    if preview.error:
        show_error(preview.error, source)
    else:
        st.code(preview.grammar_text, language=None, wrap_lines=True)


# ---- sidebar ------------------------------------------------------------------------

with st.sidebar:
    st.title("🧩 Grammar-Constrained Explorer")
    model_id = st.selectbox("Model", MODELS, help="0.5B is fast; 1.5B writes better but is ~3× slower.")
    lm = get_model(model_id)
    st.caption(f"Running on **{lm.device}** (set `GCE_DEVICE` to change)")

grammar_tab, regex_tab, json_mode_tab, structured_tab = st.tabs(
    ["① Grammar-constrained generation", "② Regex constraint", "③ JSON mode", "④ Structured output"])

# ---- tab ①: GBNF grammar ------------------------------------------------------------

with grammar_tab:
    st.header("① Grammar-constrained generation")
    st.markdown(
        "At every step the model scores every token. The grammar then decides which tokens are "
        "**allowed**: only tokens that keep the output a valid start of the grammar can be chosen. "
        "End-of-sequence is allowed only once the output fully matches the **root rule**."
    )
    inputs = tab_inputs(Tab.GRAMMAR, GRAMMAR_PRESETS, "1")
    preset = inputs.preset
    with inputs.left:
        grammar = st.text_area("Grammar (GBNF)", preset.grammar, height=180, key=f"{Tab.GRAMMAR}_grammar_{preset.name}",
                               help="Supported: \"literals\" with escapes, [a-z] classes (and [^...]), "
                                    "( ) groups, |, * + ?, rule references and # comments.")
    with inputs.right:
        rules = rule_names(grammar) or [preset.root]
        root = st.selectbox("Root rule", rules, index=rules.index(preset.root) if preset.root in rules else 0,
                            key=f"{Tab.GRAMMAR}_root_{preset.name}", help="The rule the whole output must match.")
    if preset.callout:
        st.info(preset.callout)

    generate_and_show(
        Tab.GRAMMAR,
        lambda: run_grammar(lm, grammar, root, inputs.system_prompt, inputs.user_text, inputs.max_new_tokens),
        source=grammar, spinner="Generating without and with the grammar…")
    notebook_snippet("transformers-cfg", TRANSFORMERS_CFG_SNIPPET)

# ---- tab ②: regex constraint ----------------------------------------------------------

with regex_tab:
    st.header("② Regex constraint")
    st.markdown(
        "A regex constraint is a grammar too. The regex is converted into a GBNF grammar with a "
        "single rule, `root`, and then decoded with exactly the same engine as tab ①: the whole "
        "output must match the regex. Anchors, lookarounds and backreferences are not supported."
    )
    inputs = tab_inputs(Tab.REGEX, REGEX_PRESETS, "1")
    regex_column, grammar_column = st.columns(2)
    with regex_column:
        regex = st.text_input("Regex", inputs.preset.regex, key=f"{Tab.REGEX}_regex_{inputs.preset.name}")
    regex_preview = preview_regex(regex)
    with grammar_column:
        st.markdown("**Generated grammar (GBNF)**")
        show_generated_grammar(regex_preview, regex)

    generate_and_show(
        Tab.REGEX,
        lambda: run_regex(lm, regex, inputs.system_prompt, inputs.user_text, inputs.max_new_tokens),
        source=regex, spinner="Converting the regex and generating without and with it…",
        disabled=bool(regex_preview.error))
    notebook_snippet("Outlines", OUTLINES_SNIPPET,
                     caption="The notebook's `convert_message_to_prompt` has no `return`; `return prompt` is added "
                             "here. Its cell has no saved output.")

# ---- tab ③: JSON mode ---------------------------------------------------------------------

with json_mode_tab:
    st.header("③ JSON mode")
    st.markdown(
        "JSON mode is a grammar too: one that allows **any** valid JSON object. The output is "
        "guaranteed to parse, but the grammar says nothing about which keys appear or what type each "
        "value has. Only the prompt can ask for particular keys, such as `author`, `title` and `year`."
    )
    inputs = tab_inputs(Tab.JSON_MODE, JSON_PRESETS, "2a", default_max_tokens=96)
    with inputs.right:
        json_limit_whitespace = st.toggle(
            "Limit whitespace", value=True, key=f"{Tab.JSON_MODE}_limit_ws",
            help="On: each gap allows at most one space, or one newline plus a short indent. Off: unlimited "
                 "whitespace, so a model that keeps choosing newlines or spaces can loop until the token limit.")
    json_preview = preview_json_mode(json_limit_whitespace)
    with st.expander("Generated JSON grammar (GBNF)"):
        st.code(json_preview.grammar_text, language=None, wrap_lines=True)
    st.info(
        "**Keys and value types are not controlled.** Inside an object the grammar allows any string as a "
        "key and any JSON value after it. Only the prompt asks for particular keys: with the preset whose "
        "prompt doesn't name them, Qwen 0.5B writes `publication_year` instead of `year`. Value types are "
        "up to the model too: on the Tirukkural paragraph it writes `\"year\": \"450-500 CE\"`, a string "
        "(the notebook's GPT-4o-mini wrote the string `\"NULL\"` where JSON's `null` was meant). JSON strings allow any "
        "character, so accents such as \"García Márquez\" survive here, unlike tab ①'s `[a-zA-Z ]`. "
        "Whitespace is optional everywhere in JSON, so **Limit whitespace** caps each gap. A whitespace loop "
        "happens when the model wants a token the grammar disallows and whitespace is the best allowed "
        "token left. JSON mode rarely forces that: in the app's smoke test Qwen 0.5B never looped on "
        "these presets, with the limit on or off. To see a real loop, turn the limit off in tab ④ and "
        "run \"Clients (€43)\", where a required field does the forcing."
    )

    generate_and_show(
        Tab.JSON_MODE,
        lambda: run_json_mode(lm, inputs.system_prompt, inputs.user_text, inputs.max_new_tokens,
                              json_limit_whitespace),
        source=json_preview.grammar_text, spinner="Generating without and with the JSON grammar…")
    notebook_snippet("OpenAI JSON mode", OPENAI_JSON_MODE_SNIPPET,
                     caption="The notebook sends the system prompt with role `developer`, OpenAI's newer name for "
                             "`system`; this app sends it as `system`.")

# ---- tab ④: structured output ---------------------------------------------------------------

with structured_tab:
    st.header("④ Structured output")
    st.markdown(
        "Structured output is a grammar built from a schema. Each object's keys are spelled out in "
        "schema order and each value must have the field's type, so the model only fills in the "
        "values. Enums become literal alternatives: only their values can be written."
    )
    inputs = tab_inputs(Tab.STRUCTURED, SCHEMA_PRESETS, "2b", text_height=100, system_height=110)
    with inputs.right:
        schema_limit_whitespace = st.toggle(
            "Limit whitespace", value=True, key=f"{Tab.STRUCTURED}_limit_ws",
            help="As in tab ③. Turn it off and run \"Clients (€43)\": the model wants to close the object "
                 "before the required `currency`, every `}` is disallowed, and it writes newlines instead.")

    schema_column, view_column = st.columns(2)
    with schema_column:
        schema = st.text_area(
            "JSON Schema", inputs.preset.schema, height=380, key=f"{Tab.STRUCTURED}_schema_{inputs.preset.name}",
            help="Supported: type object, string, number, integer or boolean; properties; required; enum; "
                 "nested objects; and title, description and default.")
    schema_preview = preview_schema(schema, schema_limit_whitespace)
    with view_column:
        if schema_preview.error:
            st.markdown("**Schema error**")
            show_error(schema_preview.error, schema)
        else:
            view_tab, gbnf_tab = st.tabs(["Python dataclass view", "Generated grammar (GBNF)"])
            with view_tab:
                st.code(schema_preview.dataclass_view, language="python")
                st.caption("Read-only: the schema as notebook 2b's kind of dataclass.")
            with gbnf_tab:
                st.code(schema_preview.grammar_text, language=None, wrap_lines=True)
    st.info(
        "**Why is a field with a default still required, and why are fields in schema order?** A grammar "
        "either allows a key at a position or it doesn't. Writing every field, in the order the schema "
        "lists them, gives one simple sequence of keys. Letting fields be left out, or come in any order, "
        "would multiply the alternatives the grammar has to spell out. So `currency` must be written even "
        "though it defaults to `USD`, and `purpose` always comes first. The model has to infer the "
        "currency, since the system prompt never mentions it. Field order also matters for quality: "
        "the model writes earlier fields first and can use them when choosing later ones.\n\n"
        "**A real whitespace loop.** On \"Clients (€43)\" and \"Dinner (₹820)\" Qwen 0.5B wants to "
        "end the object right after `amount`, but `}` is disallowed until `currency` is written. The most "
        "likely token that is still allowed is whitespace. With **Limit whitespace** on, it uses up the "
        "allowed indent (hence the odd `43⏎            ,`) and is then pushed on to `currency`. Turn it "
        "off and the same run writes newlines until the token limit: ⚠️ incomplete output."
    )

    generate_and_show(
        Tab.STRUCTURED,
        lambda: run_structured(lm, schema, inputs.system_prompt, inputs.user_text, inputs.max_new_tokens,
                               schema_limit_whitespace),
        source=schema, spinner="Converting the schema and generating without and with it…",
        disabled=bool(schema_preview.error))
    notebook_snippet("pydantic-ai + Gemini", PYDANTIC_AI_SNIPPET,
                     caption="pydantic-ai 0.0.53 as in the notebook: later versions renamed `result_type=` to "
                             "`output_type=` and `response.data` to `response.output`.")
