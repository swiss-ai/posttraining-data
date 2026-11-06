import json

import pytest

from opencode_terminal import clean_thoughts, read_only
from source_formats import (
    SourceFormatError,
    harmony_reply,
    harmony_terminus,
    openhands_text_calls,
    text_format_residue,
)


def litecoder_row(*turns):
    return {"id": "x", "conversations": [{"from": f, "value": v} for f, v in turns]}


CALL = '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls", "security_risk": "LOW"}}\n</tool_call>'


def test_openhands_text_calls_become_structured_calls_and_results():
    record = openhands_text_calls(litecoder_row(
        ("human", "Do the task"),
        ("gpt", "Let me look.\n" + CALL),
        ("human", "<tool_response>\na\nb\n</tool_response>"),
        ("gpt", '<tool_call>\n{"name": "finish", "arguments": {"message": "Done"}}\n</tool_call>'),
    ))
    messages = record["messages"]
    assert messages[0]["role"] == "system" and messages[0]["content"].startswith("You are OpenHands agent")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[2]["content"] == "Let me look."
    call = messages[2]["tool_calls"][0]
    assert call["function"]["name"] == "execute_bash"
    assert json.loads(call["function"]["arguments"]) == {"command": "ls", "security_risk": "LOW"}
    assert messages[3] == {"role": "tool", "tool_call_id": call["id"], "content": "a\nb"}
    assert [t["function"]["name"] for t in record["tools"]] == [
        "execute_bash", "think", "finish", "execute_ipython_cell", "task_tracker", "str_replace_editor"]
    assert "conversations" not in record


def test_openhands_harness_messages_stay_user_messages():
    record = openhands_text_calls(litecoder_row(
        ("human", "Do the task"),
        ("gpt", CALL),
        ("human", "<tool_response>\nok\n</tool_response>"),
        ("human", "Please continue on whatever approach you think is suitable."),
        ("gpt", "Finished."),
    ))
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant", "tool", "user", "assistant"]


def test_terminus_rows_and_claude_code_rows():
    terminus = litecoder_row(("human", "Format your response as JSON ..."), ("gpt", '{"analysis": ""}'))
    assert openhands_text_calls(terminus) is terminus
    with pytest.raises(SourceFormatError, match="claude_code_text_calls"):
        openhands_text_calls(litecoder_row(
            ("human", "Do it"), ("gpt", '<tool_call>\n{"name": "Bash", "arguments": {"command": "ls"}}\n</tool_call>')))


def test_harmony_channels_split_into_reasoning_and_reply():
    content = ('<|channel|>analysis<|message|>Need to list files.<|end|><|start|>assistant<|channel|>commentary'
               '<|message|>List them.<|end|><|start|>assistant<|channel|>final<|message|>{"analysis": "a", "commands": []}')
    assert harmony_reply(content) == ("Need to list files.\n\nList them.", '{"analysis": "a", "commands": []}')
    record = harmony_terminus({"messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": content}]})
    assert record["messages"][1]["content"] == '<think>Need to list files.\n\nList them.</think>{"analysis": "a", "commands": []}'
    assert harmony_reply("plain text") is None


def test_text_format_residue():
    assert text_format_residue({"messages": [{"role": "assistant", "content": "<|channel|>x"}]}) == "harmony_tokens_in_text"
    assert text_format_residue({"messages": [{"role": "assistant", "content": CALL}]}) == "tool_call_as_text"
    assert text_format_residue({"messages": [{"role": "user", "content": "<tool_response>\nx\n</tool_response>"}]}) == "tool_response_as_text"
    assert text_format_residue({"messages": [{"role": "assistant", "content": "Use `<tool_call>` tags like this"}]}) is None


def test_read_only_final_checks():
    assert read_only("ls -la /output 2>/dev/null || echo missing")
    assert read_only("cat f | grep x")
    assert not read_only("rm -f core.*")
    assert not read_only("mkdir -p /output")
    assert not read_only("echo done > /output/x")
    assert not read_only("python3 run.py")


class Counts:
    def count(self, *args):
        pass


def test_repeated_reasoning_sentences_are_dropped():
    think = "All tests passed. The fix works. Let me clean up."
    analysis = "All tests passed. The fix works. The method now converts pairs."
    assert clean_thoughts([think, analysis], Counts()) == [think, "The method now converts pairs."]


def test_task_tracker_items_are_mapped_onto_the_openhands_schema():
    from source_formats import task_tracker_items
    arguments, changed = task_tracker_items({"command": "plan", "task_list": [
        {"id": 1, "content": "Install packages", "status": "pending", "priority": "high"},
        {"id": "2", "description": "Train", "status": "completed", "ACTIVE_FORM": "Training"},
        {"id": "3", "title": "Report", "status": "in_progress", "notes": "draft"},
    ]})
    assert changed
    assert arguments["task_list"] == [
        {"id": "1", "title": "Install packages", "status": "todo", "notes": "priority: high"},
        {"id": "2", "title": "Train", "status": "done", "notes": "ACTIVE_FORM: Training"},
        {"id": "3", "title": "Report", "status": "in_progress", "notes": "draft"},
    ]
    with pytest.raises(SourceFormatError, match="task_item_not_mappable"):
        task_tracker_items({"command": "plan", "task_list": [{"id": "1", "content": "x", "status": "failed"}]})
