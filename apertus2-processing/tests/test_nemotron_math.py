import json

import pytest
from apertus_common import Conversation
from conftest import clean, load_mapper, types
from datasets import Dataset

from apertus2_processing import mapping

TOOL = {
    "type": "function",
    "function": {
        "name": "stateful_python_code_exec",
        "description": "Run Python.",
        "parameters": json.dumps(
            {"properties": {"code": {"type": "string"}}, "required": ["code"], "type": "object"}
        ),
    },
}
PROBLEM = (
    "What is 6 times 7?\nPlease reason step by step, and put your final answer within \\boxed{}."
)
FINAL = "Multiplying gives\n\n\\[\n\\boxed{\\frac{a}{b} + \\{c\\}}\n\\]"


def row(uuid, subset, messages, tools=None):
    """A source row with the columns the mapper reads plus one it must drop."""
    return {
        "uuid": uuid,
        "problem": "What is 6 times 7?",
        "expected_answer": "42",
        "messages": messages,
        "tools": tools or [],
        "source": "AoPS",
        "license": "cc-by-4.0",
        "subset": subset,
    }


def message(role, content, **extra):
    """A message as `datasets` yields it: every struct field present, `None` when unset."""
    fields = {"reasoning_content": None, "tool_calls": None, "name": None, "tool_call_id": None}
    return {"role": role, "content": content, **fields} | extra


def call(number):
    function = {"name": "stateful_python_code_exec", "arguments": '{"code": "print(6*7)"}'}
    return {"id": f"chatcmpl-tool-{number}", "type": "function", "function": function}


def result(number, content):
    return message(
        "tool", content, name="stateful_python_code_exec", tool_call_id=f"chatcmpl-tool-{number}"
    )


COT = row(
    "cot-1",
    "cot",
    [message("user", PROBLEM), message("assistant", FINAL, reasoning_content="6*7 is 42.")],
)
TIR = row(
    "tir-1",
    "tir",
    [
        message("user", PROBLEM),
        message("assistant", "", reasoning_content="Let me compute.", tool_calls=[call(1)]),
        result(1, ""),
        message("assistant", "Retrying.", reasoning_content="Empty output.", tool_calls=[call(2)]),
        result(2, "42"),
        message("assistant", "So \\boxed{42}.", reasoning_content="Done."),
    ],
    tools=[TOOL],
)
NO_BOX = row(
    "cot-2", "cot", [message("user", PROBLEM), message("assistant", "42", reasoning_content="r")]
)
EMPTY_BOX = row(
    "cot-3",
    "cot",
    [message("user", PROBLEM), message("assistant", "\\boxed{}", reasoning_content="r")],
)
NO_REASONING = row("cot-4", "cot", [message("user", PROBLEM), message("assistant", FINAL)])


@pytest.mark.parametrize(
    "text,answer",
    [
        (FINAL, "\\frac{a}{b} + \\{c\\}"),
        ("first \\boxed{1} then \\boxed{ 2 }", "2"),
        ("no answer", None),
        ("unbalanced \\boxed{1", None),
    ],
)
def test_boxed_answer_takes_the_last_balanced_box(text, answer):
    assert load_mapper("nemotron_math").boxed_answer(text) == answer


def test_to_native_inserts_claim_before_final_reply():
    mapper = load_mapper("nemotron_math")
    cot = mapper.to_native(COT)
    assert types(cot) == ["user", "think", "claim", "reply", "wait"]
    assert cot.items[2].payload == "\\frac{a}{b} + \\{c\\}" and cot.items[3].payload == FINAL
    assert cot.items[0].payload == PROBLEM and cot.system.thinking == "high" and clean(cot)

    tir = mapper.to_native(TIR)
    assert types(tir) == [
        *["user", "think", "call", "wait", "result"],
        *["think", "reply", "call", "wait", "result"],
        *["think", "claim", "reply", "wait"],
    ]
    assert [tool.name for tool in tir.system.tools] == ["stateful_python_code_exec"]
    assert tir.system.tools[0].json_schema["required"] == ["code"]
    results = [item.payload for item in tir.items if item.type == "result"]
    assert results == ["", "42"] and tir.items[11].payload == "42" and clean(tir)

    for bad in (NO_BOX, EMPTY_BOX):
        with pytest.raises(ValueError, match="no boxed answer"):
            mapper.to_native(bad)
    # The think-first convention places the empty think before the claim.
    bare = mapping.apply_convention(mapper.to_native(NO_REASONING))
    assert types(bare) == ["user", "think", "claim", "reply", "wait"] and clean(bare)


def test_nemotron_mapper_end_to_end():
    mapper = load_mapper("nemotron_math")
    source = Dataset.from_list([COT, TIR, NO_BOX])
    native, summary = mapping.map_dataset(
        source, mapper.to_native, keep_columns=mapper.KEEP_COLUMNS, check=True
    )
    assert sorted(native.column_names) == sorted([*mapper.KEEP_COLUMNS, "conversation_json"])
    assert summary["written"] == 2 and summary["skipped"] == {"no boxed answer in final reply": 1}
    assert summary["check"]["failed_rows"] == 0 and summary["check"]["issues"] == []
    assert native["expected_answer"] == ["42", "42"]
    last = Conversation.from_json(native["conversation_json"][1])
    assert types(last)[-3:] == ["claim", "reply", "wait"]
