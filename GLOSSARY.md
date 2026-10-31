# Grammar-Constrained Explorer

A learning tool for seeing how a formal grammar limits which tokens a language model may produce, step by step.

## Language

### Grammars

**Grammar**:
A set of formal rules (written in GBNF) that defines exactly which strings are valid output. Not English grammar.
_Avoid_: Format, template, syntax

**Rule**:
One named definition in a grammar, such as `year ::= [1-2][0-9][0-9][0-9] | unk`.
_Avoid_: Production, line

**Root rule**:
The rule that the whole output must match.
_Avoid_: Start symbol, entry point

**Character class**:
A part of a rule written in square brackets, such as `[a-zA-Z ]` or `[^"]`, that matches one character from a set (or, with `^`, any character not in it).
_Avoid_: Charset, range

**Grammar position**:
Where the output generated so far sits within the grammar: which rules are still open and what may come next. The step trace shows it as "Open rules" and "May come next".
_Avoid_: Parser state, cursor

**Open rule**:
A rule that the output has started but not yet finished at the current grammar position.
_Avoid_: Active rule, stack frame

### Model output

**Token**:
The unit the model writes in, one per step: often a word with its leading space (`␣Garcia`), sometimes part of a word, a digit or a newline. The app shows spaces as `␣` and newlines as `⏎`.
_Avoid_: Word, character

**Logit**:
The model's raw score for one token at one step. Softmax turns the logits into probabilities.
_Avoid_: Score, weight

**Greedy decoding**:
Choosing the most probable allowed token at every step, so the same input always gives the same output.
_Avoid_: Argmax sampling

**End-of-sequence**:
The special token (`<|im_end|>` for Qwen) with which the model says it has finished. Under a grammar it is allowed only once the output fully matches the root rule.
_Avoid_: EOS token, stop token

### Decoding

**Grammar-constrained decoding**:
Generating text where, at every step, any token that would make the output no longer match the grammar is masked out.
_Avoid_: Grammar filtering, guided generation

**Allowed token**:
A token that, appended to the output so far, still leaves a valid prefix of the grammar.
_Avoid_: Legal token, valid token

**Disallowed token**:
A token whose logit is masked to negative infinity because it would break the grammar.
_Avoid_: Banned token (that term belongs to logits masking with phrase lists)

**Mask**:
Setting the logits of disallowed tokens to negative infinity before the token is chosen, so their probability becomes zero and the rest are renormalised. Grammar-constrained decoding is logits masking where the grammar decides what to mask.
_Avoid_: Filter, block

**Complete output**:
Constrained output that fully matches the root rule (shown as ✅). It can still have stopped at the token limit, if the grammar allows more text and the model never chose end-of-sequence.
_Avoid_: Valid output, success

**Incomplete output**:
Constrained output that hit the token limit before fully matching the root rule (shown as ⚠️).
_Avoid_: Truncated, failed

**Unconstrained run**:
The same prompt generated with no grammar, shown for comparison.
_Avoid_: Baseline, free run

**Step trace**:
The per-step record of a constrained run: the top 10 tokens before the mask, the top 10 allowed tokens after it, the grammar position, and how many tokens were checked and disallowed.
_Avoid_: Log, debug view

**Checked tokens**:
The tokens, taken from the highest logit down, that were tested against the grammar at one step before 10 allowed tokens were found. Most of the vocabulary is never checked.
_Avoid_: Scanned, evaluated

**First-character pre-filter**:
A shortcut while checking tokens: a token whose first character the grammar cannot accept next is disallowed without walking it through the grammar. It changes speed, not results.
_Avoid_: Index, cache

**Exact count**:
The number of allowed and disallowed tokens across the whole vocabulary at one step, computed on demand by checking every token.
_Avoid_: Full mask, vocabulary scan

**Whitespace loop**:
A constrained run that keeps writing spaces or newlines until the token limit, because the model wants a token the grammar disallows and whitespace is the most likely token that is still allowed.
_Avoid_: Infinite loop, hang

### Constraint forms

**Regex constraint**:
A constraint written as a regular expression and converted into a grammar before decoding.
_Avoid_: Pattern

**JSON mode**:
A constraint that allows any valid JSON but does not control its keys or value types.
_Avoid_: JSON output

**Limit whitespace**:
A JSON grammar setting that lets each gap between JSON tokens hold at most one space, or one newline plus up to 12 spaces or tabs of indent, so a whitespace loop cannot go on for long.
_Avoid_: Compact mode, minify

**Schema**:
A description of the exact fields, types and allowed values the output must have.
_Avoid_: Model, type, dataclass

**Structured output**:
A constraint built from a schema, so the output is JSON with exactly the fields and value types the schema describes.
_Avoid_: Typed output, function calling

**Enum**:
A schema field that may hold only one of a listed set of values, such as `currency` in `USD`, `UKP`, `INR` or `EUR`. In the grammar it becomes a choice between literal strings.
_Avoid_: Choice, options list

**Preset**:
A ready-made set of inputs for a tab (system prompt, input texts and the grammar, regex or schema), taken from the book's notebooks and stored as data in `grammar_lab/presets.py`.
_Avoid_: Example, sample
