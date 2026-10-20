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


def written(mapper, row):
    """The sample as the driver writes it, with the default thinking convention."""
    return mapping.apply_convention(mapper.to_native(row))


def read_rows(tmp_path, *records):
    mapper = load_mapper("apertus_tool_use")
    return mapper, mapper.read([write(tmp_path / "mix.jsonl", *records)])


def test_a_conversation_keeps_its_thinks_and_tools(tmp_path):
    mapper, rows = read_rows(tmp_path, TOOL_USE)
    native = written(mapper, rows[0])
    assert types(native) == ["user", "think", "call", "wait", "result", "think", "reply", "wait"]
    assert native.items[2].payload == '{"q":"Zürich"}'  # compact JSON, as the profile writes it
    assert native.system.behavior == "Be brief." and native.system.thinking == "medium"
    assert [tool.name for tool in native.system.tools] == ["search"]
    assert set(weights(native)) == {None} and clean_with_weights(native)


def test_a_multi_turn_conversation_is_one_sample_with_the_thinks_of_every_turn(tmp_path):
    mapper, rows = read_rows(tmp_path, TERMINAL)
    assert len(rows) == 1
    native = written(mapper, rows[0])
    assert types(native) == ["user", "think", "reply", "wait", "user", "think", "reply", "wait"]
    # The source think of the first turn stays; the second turn has none, so it gets an empty one.
    assert native.items[1].payload == "Look first." and native.items[5].payload == ""
    assert native.items[2].payload == '{"commands": []}'
    assert set(weights(native)) == {None} and clean_with_weights(native)


def test_a_conversation_without_reasoning_gets_medium_and_empty_thinks(tmp_path):
    mapper, rows = read_rows(tmp_path, NO_REASONING)
    native = written(mapper, rows[0])
    assert native.system.thinking == "medium"
    assert types(native) == ["user", "think", "reply", "wait", "user", "think", "reply", "wait"]
    assert native.items[1].payload == native.items[5].payload == ""
    assert clean_with_weights(native)


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
        mapper.read(paths), mapper.to_native, keep_columns=mapper.KEEP_COLUMNS, check=True
    )
    assert native.column_names == [*mapper.KEEP_COLUMNS, "conversation_json"]
    assert native["id"] == [TOOL_USE["id"], TERMINAL["id"], NO_REASONING["id"]]
    assert native["teacher"] == ["GPT-OSS-120B", "Kimi-K3", None]
    assert native["task"] == [None, "t1", None]
    assert native["license"] == ["MIT"] * 3
    assert summary["written"] == 3 and summary["check"]["issues"] == []
    assert types(Conversation.from_json(native["conversation_json"][2]))[1] == "think"


def test_the_limit_counts_the_empty_thinks_of_the_written_sample(tmp_path, artifact):
    mapper, rows = read_rows(tmp_path, NO_REASONING)
    encoding = mapper._encoding(artifact)
    limit = len(encoding.encode_conversation(written(mapper, rows[0])))
    assert len(encoding.encode_conversation(mapper.to_native(rows[0]))) < limit
    assert mapper.to_native(rows[0], limit, artifact).items == mapper.to_native(rows[0]).items
    with pytest.raises(ValueError, match=f"longer than {limit - 1} tokens"):
        mapper.to_native(rows[0], limit - 1, artifact)
