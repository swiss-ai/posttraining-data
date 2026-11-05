import json

import pytest

from standardise_native import checked_row
from terminus_terminal import apply_to_result, convert_chat, reply_inside_think

ENTRY = dict(
    repo="x/y",
    revision="r",
    category="code_agentic",
    license="unknown",
    license_status="unknown",
    license_notes="",
    synthetic_status="unknown",
    synthetic_notes="",
    metadata_source="test",
    code_policy="code_source",
)
SPEC = dict(path="t", source_config="local", source_split="train")
HARNESS = (
    "You are an AI assistant tasked with solving command-line tasks.\n"
    'Format your response as JSON with "keystrokes".\n\n'
)
TASK = "Task Description:\nFix the bug\n\nCurrent Terminal Screen:\nroot@abc:/app#\n\n\n"
REPLY = '{\n  "analysis": "Look.",\n  "plan": "List.",\n  "commands": [{"keystrokes": "ls\\n", "duration": 0.1}]\n}'
SCREEN = "Previous response had warnings:\nWARNINGS: - Extra text\n\nNew Terminal Output:\nroot@abc:/app# ls\na.py\nroot@abc:/app#\n\n\n"
DONE = '{"analysis": "Done.", "plan": "", "commands": [], "task_complete": true}'


def source(first_reply=REPLY, **extras):
    return [
        {"role": "user", "content": HARNESS + TASK},
        {"role": "assistant", "content": first_reply, **extras},
        {"role": "user", "content": SCREEN},
        {"role": "assistant", "content": DONE},
    ]


def row(messages):
    return checked_row(
        {"messages": messages}, ENTRY, SPEC, 0, {}, "think-first", None, "terminus"
    )


def test_harness_replies_and_screens_are_verbatim():
    converted = row(source())
    assert json.loads(converted["messages_json"]) == source()
    assert json.loads(converted["tools_json"]) in (None, [])


def test_native_items_are_user_think_reply_without_calls():
    conversation = json.loads(row(source())["conversation_json"])
    kinds = [item["type"] for item in conversation["items"]]
    assert "call" not in kinds and "result" not in kinds
    assert kinds[:3] == ["user", "think", "reply"]
    replies = [item["payload"] for item in conversation["items"] if item["type"] == "reply"]
    assert replies == [REPLY, DONE]


def test_the_wrapper_and_its_override_prompt_are_gone():
    converted = row(source())
    assert "terminal_step" not in converted["conversation_json"]
    assert "structured tool_calls" not in converted["conversation_json"]
    assert "Format your response as JSON" in converted["native_text"]
    provenance = json.loads(converted["provenance_json"])
    assert provenance["harness_conversion"] == {"target": "terminus"}
    assert provenance["tool_provenance"] == []
    assert not any("override" in change for change in provenance["changes"])


def test_system_prompt_stays_a_system_prompt():
    messages = [{"role": "system", "content": HARNESS}, {"role": "user", "content": TASK}] + source()[1:]
    conversation = json.loads(row(messages)["conversation_json"])
    assert HARNESS.strip() in conversation["system"]["behavior"]
    assert conversation["items"][0]["payload"] == TASK


def test_think_tags_and_reasoning_field_become_think_items():
    tagged = row(source(first_reply="<think>Inspect first.</think>\n" + REPLY))
    assert json.loads(tagged["conversation_json"])["items"][1]["payload"] == "Inspect first."
    field = row(source(reasoning_content="From the field."))
    assert json.loads(field["conversation_json"])["items"][1]["payload"] == "From the field."


def test_json_inside_think_becomes_the_reply():
    chat, reasoning, _ = convert_chat(source(first_reply=f"<think>\nInspect first.\n{REPLY}\n</think>\n\n"))
    assert chat[1]["content"] == REPLY
    assert reasoning[0]["values"] == ["Inspect first."]


def test_lost_json_opener_starts_the_reply_at_plan():
    text = '<think>\nInspect first.",\n  "plan": "List.",\n  "commands": [{"keystrokes": "ls\\n"}]\n}\n</think>\n\n'
    reply, thinking = reply_inside_think(text)
    assert json.loads(reply) == {"plan": "List.", "commands": [{"keystrokes": "ls\n"}]}
    assert thinking == "Inspect first."


def test_thinking_without_any_reply_is_refused():
    with pytest.raises(ValueError, match="no text outside thinking"):
        convert_chat(source(first_reply="<think>Only thoughts.</think>"))


def test_empty_reply_with_reasoning_field_stays_as_recorded():
    chat, reasoning, _ = convert_chat(source(first_reply="", reasoning_content="Only thoughts."))
    assert chat[1]["content"] == ""
    assert reasoning[0]["values"] == ["Only thoughts."]


def test_trailing_user_turns_move_to_provenance():
    result = {"chat": {}, "changes": []}
    apply_to_result(result, source() + [{"role": "user", "content": SCREEN}])
    assert result["chat"]["messages"][-1]["role"] == "assistant"
    assert result["trailing_context"] == [{"role": "user", "content": SCREEN}]
