"""The step trace view: a reusable Streamlit component for any constrained run (tabs ①–④).

Call ``show_trace(run, model, key="t1")`` with a constrained ``Run`` and the model it ran on.
``key`` must be unique per tab so the widgets don't clash.
"""

import pandas as pd
import streamlit as st

from grammar_lab.decoding import TOP_K, Run, Step
from grammar_lab.model_boundary import ModelBoundary
from grammar_lab.pipelines import exact_count


def _before_table(step: Step) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "token": [c.shown for c in step.before],
            "prob": [c.prob for c in step.before],
            "allowed": ["✅" if c.allowed else "🚫" for c in step.before],
        }
    )


def _after_table(step: Step) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "token": [("▶ " if i == 0 else "") + c.shown for i, c in enumerate(step.after)],
            "prob": [c.prob for c in step.after],
        }
    )


PROB = st.column_config.ProgressColumn("probability", format="%.3f", min_value=0.0, max_value=1.0)


def _show_position(step: Step):
    position = step.grammar_position
    st.markdown("**Grammar position** (before this step's token)")
    rules = " · ".join(f"`{r}`" for r in position.open_rules) or "none"
    st.markdown(f"Open rules: {rules}")
    next_chars = [f"`{c}`" for c in position.next_chars]
    if position.can_end:
        next_chars.append("**end of output** (the root rule is fully matched)")
    st.markdown("May come next: " + (" or ".join(next_chars) if next_chars else "nothing"))


def show_trace(run: Run, model: ModelBoundary, key: str):
    """Step slider, before/after tables, grammar position, counts and the mask-changed steps."""
    if not run.steps:
        st.info("No steps were recorded.")
        return
    st.subheader("Step trace")
    key = f"{key}_{id(run)}"  # a new run gets fresh widget state (slider position, exact counts)
    slider_key = f"{key}_step"
    changed = run.mask_changed_steps
    if changed:
        st.markdown("**Steps where the mask changed the model's top choice** (click to jump):")
        columns = st.columns(min(len(changed), 10))
        for n, index in enumerate(changed):
            step = run.steps[index]
            label = f"{index}: {step.before[0].shown} → {step.chosen.shown if step.chosen else '∅'}"
            if columns[n % len(columns)].button(label, key=f"{key}_jump_{index}"):
                st.session_state[slider_key] = index
    else:
        st.caption("The mask never changed the model's top choice in this run.")

    last = len(run.steps) - 1
    index = st.slider("Step", 0, last, key=slider_key) if last > 0 else 0
    step = run.steps[index]
    written = model.tokenizer.text_of(run.token_ids[:index])  # bytes joined, so split characters come out whole
    st.markdown("Output so far: " + (f"`{written.replace(chr(10), '⏎')}`" if written else "(nothing yet)"))

    st.caption(
        f"Checked **{step.checked}** tokens (highest logit first) to find {TOP_K} allowed ones: "
        f"**{step.disallowed}** disallowed."
    )
    before, after = st.columns(2)
    with before:
        st.markdown(f"**Top {TOP_K} before the mask** (🚫 = disallowed)")
        st.dataframe(_before_table(step), hide_index=True, column_config={"prob": PROB}, width="stretch")
    with after:
        st.markdown(f"**Top {TOP_K} allowed after the mask** (renormalised over these)")
        st.dataframe(_after_table(step), hide_index=True, column_config={"prob": PROB}, width="stretch")
    _show_position(step)

    count_key = f"{key}_exact_{index}"
    if st.button("Exact count: check the full vocabulary at this step", key=f"{key}_exact_go"):
        with st.spinner("Checking every token in the vocabulary…"):
            st.session_state[count_key] = exact_count(model, step)
    count = st.session_state.get(count_key)
    if count:
        total = count.allowed + count.disallowed
        st.info(
            f"Across all {total:,} tokens: **{count.allowed:,} allowed**, **{count.disallowed:,} disallowed** "
            f"({count.disallowed / total:.1%} of the vocabulary ruled out by the grammar)."
        )
