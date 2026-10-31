"""Tab ④ (structured output) through its pipeline entry point, with a fake model."""

import pytest

from grammar_lab.pipelines import run_structured
from grammar_lab.presets import INVOICE
from tests.fakes import EOS, IM_START, FakeModel

INVOICE_VOCAB = [
    EOS, IM_START,
    "{", "}", "\"", "\":", " ", ",", " \"", "\"}",
    "purpose", "amount", "currency", "note",
    "taxi", " ride", "32.30", "thirty",
    "GBP", "USD", "EUR", "UKP", "INR",
]


def invoice_model(script):
    return FakeModel(script, vocab=INVOICE_VOCAB)


def run_invoice(model, schema=INVOICE.schema, invoice="Taxi ($32.30)", max_new_tokens=40):
    return run_structured(model, schema, INVOICE.system_prompt, INVOICE.inputs[invoice], max_new_tokens)


# What the model "wants" at each step: wrong field order, an unknown key, ending before the
# defaulted `currency` field, and a currency that is not in the enum.
WAYWARD_SCRIPT = [
    ["{"], ["\""], ["amount", "purpose"], ["\":"], [" "], ["\""], ["taxi"], [" ride"], ["\""],
    [","], [" \""], ["note", "amount"], ["\":"], [" "], ["thirty", "32.30"], ["}", ","],
    [" \""], ["currency"], ["\":"], [" "], ["\""], ["GBP", "USD"], ["\"}"], [EOS],
]


def test_structured_output_permits_only_the_schemas_fields_in_order_and_its_enum_values():
    result = run_invoice(invoice_model(WAYWARD_SCRIPT))

    assert result.constrained.text == '{"purpose": "taxi ride", "amount": 32.30, "currency": "USD"}'
    assert result.complete
    steps = result.constrained.steps
    assert [(steps[i].before[0].text, steps[i].chosen.text) for i in result.constrained.mask_changed_steps] == [
        ("amount", "purpose"),  # fields come in schema order
        ("note", "amount"),     # keys outside the schema are disallowed
        ("thirty", "32.30"),    # `amount` must be a number
        ("}", ","),             # `currency` has a default but is still required
        ("GBP", "USD"),         # only the enum's values
    ]
    assert result.unconstrained.text.startswith('{"amount')


def test_the_invoice_schema_is_shown_as_the_grammar_it_becomes():
    result = run_invoice(invoice_model(WAYWARD_SCRIPT))

    assert result.grammar_text.startswith(
        'root ::= "{" ws\n'
        '    "\\"purpose\\":" ws string "," ws\n'
        '    "\\"amount\\":" ws number "," ws\n'
        '    "\\"currency\\":" ws currency\n'
        '    "}" ws\n'
        'currency ::= ("\\"USD\\"" | "\\"UKP\\"" | "\\"INR\\"" | "\\"EUR\\"") ws\n'
        'string ::= '
    )
    assert "\nnumber ::= " in result.grammar_text
    assert "integer ::=" not in result.grammar_text  # only the value rules the schema uses


NESTED_SCHEMA = """{
  "type": "object",
  "properties": {
    "paid": {"type": "boolean"},
    "vendor": {
      "type": "object",
      "properties": {"name": {"type": "string"}, "rating": {"type": "integer"}}
    }
  },
  "required": ["paid", "vendor"]
}"""

NESTED_VOCAB = INVOICE_VOCAB + ["paid", "true", "vendor", "name", "rating", "4", "4.5", "Uber"]


def test_nested_objects_integers_and_booleans_are_constrained_too():
    script = [["{"], ["\""], ["paid"], ["\":"], [" "], ["yes", "true"], [","], [" \""], ["vendor"], ["\":"],
              [" "], ["{"], ["\""], ["name"], ["\":"], [" "], ["\""], ["Uber"], ["\""], [","], [" \""],
              ["rating"], ["\":"], [" "], ["4.5", "4"], ["}"], ["}"], [EOS]]
    model = FakeModel(script, vocab=NESTED_VOCAB + ["yes"])
    result = run_structured(model, NESTED_SCHEMA, "", "", 40)

    assert result.constrained.text == '{"paid": true, "vendor": {"name": "Uber", "rating": 4}}'
    assert result.complete
    assert result.grammar_text.startswith(
        'root ::= "{" ws\n'
        '    "\\"paid\\":" ws boolean "," ws\n'
        '    "\\"vendor\\":" ws vendor\n'
        '    "}" ws\n'
        'vendor ::= "{" ws\n'
        '    "\\"name\\":" ws string "," ws\n'
        '    "\\"rating\\":" ws integer\n'
        '    "}" ws\n'
    )


def schema_with(field: str, required: str = '["amount"]') -> str:
    return '{\n  "type": "object",\n  "properties": {\n    "amount": ' + field + '\n  },\n  "required": ' + required + "\n}"


@pytest.mark.parametrize("schema, line, words", [
    ('{\n  "type": "object",\n  "properties": {\n', 4, "not valid JSON"),
    ('["purpose", "amount"]', 1, "must be a JSON object"),
    ('{"type": "string"}', 1, 'top level must be `"type": "object"`'),
    (schema_with('{"type": "array", "items": {"type": "number"}}'), 4, "type `array` is not supported"),
    (schema_with('{"type": ["number", "null"]}'), 4, "a list of types is not supported"),
    (schema_with('{"description": "how much"}'), 4, "needs a `type` or an `enum`"),
    (schema_with('{"type": "number", "minimum": 0}'), 4, "`minimum` is not supported"),
    (schema_with('{"anyOf": [{"type": "number"}]}'), 4, "`anyOf` is not supported"),
    (schema_with('{"type": "string", "enum": []}'), 4, "`enum` needs at least one value"),
    (schema_with('{"type": "string", "enum": ["USD", 1]}'), 4, "`1` is not a string"),
    (schema_with('{"type": "number"}', required='["amount", "price"]'), 6, "`price`, which is not in `properties`"),
    ('{"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": true}', 1,
     "`additionalProperties` can only be false"),
])
def test_invalid_or_unsupported_schemas_give_clear_located_errors_and_no_runs(schema, line, words):
    result = run_invoice(invoice_model(WAYWARD_SCRIPT), schema=schema)

    assert result.constrained is None and result.unconstrained is None
    assert words in str(result.error)
    assert result.error.line == line
    assert str(result.error).startswith(f"Line {line}, column ")


def test_schema_errors_name_the_field_they_are_about():
    result = run_invoice(invoice_model(WAYWARD_SCRIPT), schema=schema_with('{"type": "array"}'))

    assert "`amount`" in str(result.error)
