"""Tab ④ (structured output) through its pipeline entry point, with a fake model."""

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
