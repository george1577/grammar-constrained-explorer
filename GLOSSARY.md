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

**Grammar position**:
Where the output generated so far sits within the grammar: which rules are still open and what may come next.
_Avoid_: Parser state, cursor

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

**Incomplete output**:
Constrained output that hit the token limit before fully matching the root rule.
_Avoid_: Truncated, failed

**Unconstrained run**:
The same prompt generated with no grammar, shown for comparison.
_Avoid_: Baseline, free run

### Constraint forms

**Regex constraint**:
A constraint written as a regular expression and converted into a grammar before decoding.
_Avoid_: Pattern

**JSON mode**:
A constraint that allows any valid JSON but does not control its keys or value types.
_Avoid_: JSON output

**Schema**:
A description of the exact fields, types and allowed values the output must have.
_Avoid_: Model, type, dataclass

**Structured output**:
A constraint built from a schema, so the output is JSON with exactly the fields and value types the schema describes.
_Avoid_: Typed output, function calling
