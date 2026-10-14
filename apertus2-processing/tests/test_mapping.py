import json

import pytest
from apertus_common import AdapterError, Conversation, SystemPrompt
from conftest import clean, load_mapper, run_all, types
from datasets import Dataset, load_from_disk

from apertus2_processing import mapping

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add one.",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
        },
    }
]
TOOL_CHAT = [
    {"role": "user", "content": "add 1"},
    {
        "role": "assistant",
        "content": None,
        "reasoning_content": "call the tool",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a":1}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "2"},
    {"role": "assistant", "content": "2"},
    {"role": "user", "content": "thanks"},
    {"role": "assistant", "content": "welcome"},
]
TURN = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]


def chats(*extra):
    """Three clean single-turn chats followed by `extra` message lists, with an id column."""
    messages = [TURN] * 3 + list(extra)
    return Dataset.from_dict({"id": list(range(len(messages))), "messages": messages})


def to_native(row):
    if not row["messages"]:
        raise ValueError("empty chat")
    return mapping.from_chat(row["messages"])


@pytest.mark.parametrize("convention", mapping.CONVENTIONS)
def test_apply_convention_inserts_empty_thinks_only_under_think_first(convention):
    chat = mapping.from_chat([*TURN, {"role": "user", "content": "more"}, TURN[1]])
    tools = mapping.from_chat(TOOL_CHAT[:4], tools=TOOLS)
    for before, expected in (
        (chat, ["user", "think", "reply", "wait", "user", "think", "reply", "wait"]),
        (tools, ["user", "think", "call", "wait", "result", "think", "reply", "wait"]),
    ):
        after = mapping.apply_convention(before, convention)
        if convention == "think-first":
            assert types(after) == expected
            inserted = [i for i in after.items if i.type == "think" and i.payload == ""]
            assert len(inserted) == len(expected) - len(before.items)
            assert mapping.apply_convention(after, convention) is after
        else:
            assert after is before
        assert after.system == before.system and clean(after)


def test_from_chat_maps_tools_reasoning_and_system():
    conversation = mapping.from_chat(TOOL_CHAT, tools=TOOLS, system={"thinking": "high"})
    assert types(conversation) == [
        *["user", "think", "call", "wait", "result", "reply", "wait"],
        *["user", "reply", "wait"],
    ]
    call, result = conversation.items[2], conversation.items[4]
    assert call.counter == result.counter == 0 and conversation.items[1].payload == "call the tool"
    assert [tool.name for tool in conversation.system.tools] == ["add"]
    assert conversation.system.thinking == "high" and clean(conversation)
    developer = [{"role": "developer", "content": "Be brief."}, *TURN]
    with pytest.raises(AdapterError):
        mapping.from_chat(developer)
    merged = mapping.from_chat([{"role": "system", "content": "A."}, *developer], merge_system=True)
    assert merged.system.behavior == "A.\n\nBe brief."
    with pytest.raises(AdapterError, match="unsupported role"):
        mapping.from_chat([{"role": "bot", "content": "x"}])


@pytest.mark.parametrize("num_proc", [1, 2])
@pytest.mark.parametrize("convention", mapping.CONVENTIONS)
def test_map_dataset_skips_failing_rows_and_keeps_columns(num_proc, convention):
    source = chats([{"role": "bot", "content": "x"}], [])
    native, summary = mapping.map_dataset(
        source, to_native, num_proc=num_proc, keep_columns=["id"], convention=convention
    )
    assert native.column_names == ["id", "conversation_json"] and native["id"] == [0, 1, 2]
    assert summary == {
        "rows": 5,
        "written": 3,
        "skipped": {"empty chat": 1, "unsupported role 'bot'": 1},
        "convention": convention,
        "check": None,
    }
    expected = ["user", "reply", "wait"]
    if convention == "think-first":
        expected.insert(1, "think")
    for raw in native["conversation_json"]:
        conversation = Conversation.from_json(raw)
        assert types(conversation) == expected and clean(conversation)


def test_map_dataset_check_reports_and_strict_skips():
    def flawed(row):
        builder = Conversation.builder().system(SystemPrompt.build()).user("q")
        if row["id"] == 1:
            return builder.reply("a").build()  # no final wait: warning only
        if row["id"] == 2:
            return builder.call(name="nope", counter=0, arguments={}).wait().build()  # error
        return builder.reply("a").wait().build()

    source = chats()
    native, summary = mapping.map_dataset(source, flawed, check=True)
    assert len(native) == 3 and summary["skipped"] == {}
    assert summary["check"] == {
        "rows_with_issues": 2,
        "failed_rows": 1,
        "issues": [
            {"rule": "arguments/schema", "severity": "unevaluated", "count": 1},
            {"rule": "conversation/declared-tool", "severity": "error", "count": 1},
            {"rule": "training/final-wait", "severity": "warning", "count": 1},
        ],
    }
    native, summary = mapping.map_dataset(source, flawed, strict=True)
    assert len(native) == 2 and summary["written"] == 2
    assert summary["skipped"] == {"check/conversation/declared-tool": 1}
    assert summary["check"]["failed_rows"] == 1


def test_map_dataset_side_columns_survive_a_clean_first_batch():
    """Arrow types are fixed by the first 1000-row batch; later skips and issues must fit."""

    def to_native_or_flawed(row):
        if row["id"] == 1002:  # no final wait: warning only
            return Conversation.builder().system(SystemPrompt.build()).user("q").reply("a").build()
        return to_native(row)

    source = chats(*[TURN] * 998, [], TURN)  # 1003 rows: 1001 clean, one empty, one flawed
    native, summary = mapping.map_dataset(source, to_native_or_flawed, check=True)
    assert (summary["rows"], summary["written"]) == (1003, 1002)
    assert summary["skipped"] == {"empty chat": 1}
    assert summary["check"]["issues"] == [
        {"rule": "training/final-wait", "severity": "warning", "count": 1}
    ]
    assert len(native) == 1002


def test_from_chat_normalises_dataset_quirks():
    """Tool-message names, empty call-turn content, and string parameters as HF stores them."""
    string_tools = [{**TOOLS[0], "function": {**TOOLS[0]["function"], "parameters": "{}"}}]
    messages = [
        TOOL_CHAT[0],
        {**TOOL_CHAT[1], "content": ""},
        {**TOOL_CHAT[2], "name": "add"},
        {**TOOL_CHAT[3], "reasoning_content": None},
    ]
    conversation = mapping.from_chat(messages, tools=string_tools)
    assert types(conversation) == ["user", "think", "call", "wait", "result", "reply", "wait"]
    assert conversation.system.tools[0].json_schema == {}
    with pytest.raises(AdapterError, match="unsupported fields"):
        mapping.from_chat([{"role": "user", "content": "q", "extra": 1}])
    empty_schema = [{**TOOLS[0], "function": {**TOOLS[0]["function"], "parameters": ""}}]
    assert (
        mapping.from_chat(TURN, tools=empty_schema).system.tools[0].json_schema["type"] == "object"
    )


def test_map_dataset_handles_struct_columns_from_datasets():
    """`datasets` fills optional struct fields with None; from_chat must drop them."""
    source = Dataset.from_dict({"id": [0, 1], "messages": [TOOL_CHAT[:4], TURN]})
    assert "tool_calls" in source.features["messages"].feature
    with_tools = lambda row: mapping.from_chat(row["messages"], tools=TOOLS)
    native, summary = mapping.map_dataset(source, with_tools, check=True)
    assert summary["written"] == 2 and summary["check"]["failed_rows"] == 0
    first = types(Conversation.from_json(native["conversation_json"][0]))
    assert first[:3] == ["user", "think", "call"]


def test_map_dataset_handles_empty_input_and_rejects_bad_arguments():
    empty, summary = mapping.map_dataset(chats().select([]), to_native, keep_columns=["id"])
    assert empty.column_names == ["id", "conversation_json"] and len(empty) == 0
    assert summary["rows"] == summary["written"] == 0 and summary["check"] is None
    with pytest.raises(ValueError, match="keep_columns"):
        mapping.map_dataset(chats(), to_native, keep_columns=["nope"])
    with pytest.raises(ValueError, match="reserved"):
        mapping.map_dataset(chats(), to_native, keep_columns=["skip"])
    with pytest.raises(KeyError):  # Mapper bugs propagate instead of being counted as skips.
        mapping.map_dataset(chats(), lambda row: to_native({"messages": row["mesages"]}))
    with pytest.raises(ValueError, match="convention"):
        mapping.map_dataset(chats(), to_native, convention="always")


def test_no_robots_mapper_end_to_end(tmp_path, capsys):
    mapper = load_mapper("no_robots")
    multi_turn = [
        {"role": "system", "content": "Answer in rhymes."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there"},
        {"role": "user", "content": "Fine"},
        {"role": "assistant", "content": "Glad to hear"},
    ]
    single_turn = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    mislabelled = [*single_turn, {"role": "assistant", "content": "Thanks!"}]
    for messages, items, behavior in (
        (multi_turn, ["user", "reply", "wait", "user", "reply", "wait"], "Answer in rhymes."),
        (single_turn, ["user", "reply", "wait"], None),
    ):
        conversation = mapper.to_native({"messages": messages})
        assert conversation.system.thinking == "medium"
        assert conversation.system.behavior == behavior
        assert types(conversation) == items and clean(conversation)
    with pytest.raises(ValueError, match="alternate"):
        mapper.to_native({"messages": mislabelled})

    source = Dataset.from_dict(
        {
            "prompt_id": ["a", "b", "c"],
            "prompt": ["Hi"] * 3,
            "messages": [multi_turn, single_turn, mislabelled],
            "category": ["Chat"] * 3,
        }
    )
    parser = mapping.mapper_parser("test")
    with pytest.raises(SystemExit):
        parser.parse_args([str(tmp_path), "--num-proc", "0"])
    output = tmp_path / "native"
    args = parser.parse_args([str(output), "--num-proc", "2", "--limit", "3", "--check"])
    summary = mapping.run_mapper(source, mapper.to_native, args, keep_columns=["prompt_id"])
    assert summary["written"] == 2 and list(summary["skipped"].values()) == [1]
    assert summary["check"]["failed_rows"] == 0 and summary["check"]["issues"] == []
    assert "2 native conversations written" in capsys.readouterr().out
    native = load_from_disk(str(output))
    assert native.column_names == ["prompt_id", "conversation_json"]
    assert native["prompt_id"] == ["a", "b"]
    assert json.loads((output / mapping.SUMMARY).read_text()) == summary
    first = Conversation.from_json(native["conversation_json"][0])
    assert types(first) == ["user", "think", "reply", "wait", "user", "think", "reply", "wait"]
    assert run_all("check", output, tmp_path / "check") == 0

    args = parser.parse_args([str(tmp_path / "none"), "--limit", "1"])
    with pytest.raises(SystemExit, match="2"):  # Nothing written is a failure.
        mapping.run_mapper(source.select([2]), mapper.to_native, args)
