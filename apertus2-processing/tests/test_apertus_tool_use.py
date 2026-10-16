import json

import pytest
from apertus_common import Conversation, checkers_for
from conftest import load_mapper, types

from apertus2_processing import _run, mapping

CHECKERS = checkers_for("full", allow_loss_weights=True, training=True)
SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}
CALL = {
    "id": "call_0",
    "type": "function",
    "function": {"name": "search", "arguments": {"q": "Zürich"}},
}
META = {"source_id": "1", "revision": "r", "license": "MIT"}
TOOL_USE = {
    "id": "toucan/oss:1",
    "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather in Zürich?"},
        {"role": "assistant", "reasoning_content": "Search.", "content": "", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_0", "content": "18 degrees"},
        {"role": "assistant", "reasoning_content": "Answer.", "content": "It is 18 degrees."},
    ],
    "tools": [SEARCH],
    "meta": {"source": "toucan/oss", "teacher": "GPT-OSS-120B", "language": "eng_Latn", **META},
}
TERMINAL = {  # Terminus-2: the answer is JSON text, each terminal output is a user message
    "id": "termgrade/train:2",
    "messages": [
        {"role": "user", "content": "Task: create /app/out.txt"},
        {"role": "assistant", "reasoning_content": "Look first.", "content": '{"commands": []}'},
        {"role": "user", "content": "$ ls"},
        {"role": "assistant", "content": '{"task_complete": true}'},
    ],
    "tools": [],
    "meta": {"source": "termgrade/train", "task": "t1", "teacher": "Kimi-K3", **META},
}
NO_REASONING = {
    **TERMINAL,
    "id": "recursive_task_synthesis/train:3",
    "messages": [
        {k: v for k, v in m.items() if k != "reasoning_content"} for m in TERMINAL["messages"]
    ],
    "meta": {"source": "recursive_task_synthesis/train", **META},
}


def write(path, *records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def clean_with_weights(conversation):
    ok, found = _run.check_conversation(conversation, CHECKERS)
    return ok and not found


def weights(conversation):
    return [item.loss_weight for item in conversation.items]


def read_rows(tmp_path, *records):
    mapper = load_mapper("apertus_tool_use")
    return mapper, mapper.read([write(tmp_path / "mix.jsonl", *records)])


def test_a_conversation_with_one_user_message_keeps_its_thinks_and_tools(tmp_path):
    mapper, rows = read_rows(tmp_path, TOOL_USE)
    assert [row["turn"] for row in rows] == [0]
    native = mapper.to_native(rows[0])
    assert types(native) == ["user", "think", "call", "wait", "result", "think", "reply", "wait"]
    assert native.items[2].payload == '{"q":"Zürich"}'  # compact JSON, as the profile writes it
    assert native.system.behavior == "Be brief." and native.system.thinking == "medium"
    assert [tool.name for tool in native.system.tools] == ["search"]
    assert set(weights(native)) == {None} and clean_with_weights(native)


def test_each_user_message_gives_a_sample_without_earlier_thinks(tmp_path):
    mapper, rows = read_rows(tmp_path, TERMINAL)
    assert [row["turn"] for row in rows] == [0, 1]
    first, second = map(mapper.to_native, rows)
    assert types(first) == ["user", "think", "reply", "wait"]
    # The earlier turn has no think and no loss; the think-first convention fills the trained one.
    assert types(second) == ["user", "reply", "wait", "user", "think", "reply", "wait"]
    assert weights(second) == [None, 0, 0, None, None, None, None]
    assert second.items[1].payload == '{"commands": []}' and second.items[4].payload == ""
    assert clean_with_weights(first) and clean_with_weights(second)


def test_a_conversation_without_reasoning_is_one_sample_with_low_effort(tmp_path):
    mapper, rows = read_rows(tmp_path, NO_REASONING)
    assert len(rows) == 1
    native = mapper.to_native(rows[0])
    assert native.system.thinking == "low"
    assert types(native) == ["user", "reply", "wait", "user", "reply", "wait"]
    assert set(weights(native)) == {None} and clean_with_weights(native)


@pytest.mark.parametrize("content", ["", " \n"])
def test_an_assistant_step_without_text_and_without_calls_is_skipped(tmp_path, content):
    empty = {"role": "assistant", "reasoning_content": "Hm.", "content": content}
    mapper, rows = read_rows(tmp_path, {**TERMINAL, "messages": [TERMINAL["messages"][0], empty]})
    with pytest.raises(ValueError, match="without text"):
        mapper.to_native(rows[0])


def test_a_changed_input_file_stops_the_run(tmp_path):
    mapper, rows = read_rows(tmp_path, TOOL_USE)
    write(tmp_path / "mix.jsonl", TERMINAL)
    with pytest.raises(RuntimeError, match="changed after it was read"):
        mapper.to_native(rows[0])


def test_mapper_end_to_end_keeps_the_columns_of_the_source(tmp_path):
    mapper = load_mapper("apertus_tool_use")
    paths = [
        write(tmp_path / "a.jsonl", TOOL_USE, TERMINAL),
        write(tmp_path / "b.jsonl", NO_REASONING),
    ]
    native, summary = mapping.map_dataset(
        mapper.read(paths),
        mapper.to_native,
        keep_columns=mapper.KEEP_COLUMNS,
        convention="source",
        check=True,
        loss_weights=True,
    )
    assert native.column_names == [*mapper.KEEP_COLUMNS, "conversation_json"]
    assert native["id"] == [TOOL_USE["id"], TERMINAL["id"], TERMINAL["id"], NO_REASONING["id"]]
    assert native["turn"] == [0, 0, 1, 0]
    assert native["teacher"] == ["GPT-OSS-120B", "Kimi-K3", "Kimi-K3", None]
    assert native["task"] == [None, "t1", "t1", None]
    assert native["license"] == ["MIT"] * 4
    assert summary["written"] == 4 and summary["check"]["issues"] == []
    assert Conversation.from_json(native["conversation_json"][2]).items[1].loss_weight == 0


def test_a_sample_longer_than_the_limit_is_skipped(tmp_path, artifact):
    mapper, rows = read_rows(tmp_path, TERMINAL)
    short, long = (mapper.to_native(row) for row in rows)
    encoding = mapper._encoding(artifact)
    limit = len(encoding.encode_conversation(short))
    assert len(encoding.encode_conversation(long)) > limit
    assert mapper.to_native(rows[0], limit, artifact).items == short.items
    with pytest.raises(ValueError, match=f"longer than {limit} tokens"):
        mapper.to_native(rows[1], limit, artifact)
