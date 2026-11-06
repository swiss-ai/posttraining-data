import json

import pytest

from opencode_terminal import ConversionError
from swe_opencode import convert_chat, read_output, source_harness, strip_harness_talk

TASK = "<uploaded_files>\n/testbed\n</uploaded_files>\nFix the bug."
SWE_AGENT_TOOLS = [{"type": "function", "function": {"name": n, "parameters": {}}} for n in ("bash", "str_replace_editor", "submit")]


class Chat:
    """A checked chat in the shape openai_chat.convert produces."""

    def __init__(self, system="You are a helpful assistant that can interact with a computer to solve tasks."):
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": TASK}]
        self.number = 0

    def call(self, name, arguments, result, text=None):
        self.number += 1
        cid = f"c{self.number}"
        self.messages.append({
            "role": "assistant", "content": text,
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}],
        })
        if result is not None:
            self.messages.append({"role": "tool", "tool_call_id": cid, "content": result})
        return self


def swe_agent(chat):
    return convert_chat("swe-agent", chat.messages, [])


def calls(messages):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for m in messages for c in m.get("tool_calls") or []]


def results(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


VIEW = "OBSERVATION:\nHere's the result of running `cat -n` on /testbed/a.py:\n     1\tx = 1\n     2\t\n     3\ty = 2\n"
SUBMIT_REVIEW = "OBSERVATION:\nThank you for your work on this issue. Please carefully follow the steps below"


def test_only_the_named_releases_with_their_own_tools_are_converted():
    assert source_harness("allenai/Sera-4.6-Lite-T1", {"tools": SWE_AGENT_TOOLS}) == "swe-agent"
    assert source_harness("SWE-bench/SWE-smith-trajectories", {"tools": SWE_AGENT_TOOLS}) is None
    other = SWE_AGENT_TOOLS + [{"type": "function", "function": {"name": "browser", "parameters": {}}}]
    assert source_harness("allenai/Sera-4.6-Lite-T1", {"tools": other}) is None


def test_session_becomes_opencode_with_read_edit_and_a_closing_summary():
    chat = (
        Chat()
        .call("str_replace_editor", {"command": "view", "path": "/testbed/a.py"}, VIEW, "Look at the file.")
        .call("str_replace_editor", {"command": "str_replace", "path": "/testbed/a.py", "old_str": "x = 1", "new_str": "x = 2"},
              "OBSERVATION:\nThe file /testbed/a.py has been edited. Here's the result of running `cat -n` on a snippet")
        .call("submit", {}, SUBMIT_REVIEW, "Changed x to 2.")
        .call("submit", {}, None, "Submitting again.")
    )
    messages, tools, _, stats = swe_agent(chat)
    assert messages[0]["content"].startswith("You are opencode")
    assert messages[1]["content"] == TASK
    assert calls(messages) == [
        ("read", {"filePath": "/testbed/a.py"}),
        ("edit", {"filePath": "/testbed/a.py", "oldString": "x = 1", "newString": "x = 2"}),
    ]
    assert results(messages) == [
        "<file>\n00001| x = 1\n00002| \n00003| y = 2\n\n(End of file - total 3 lines)\n</file>", ""
    ]
    assert messages[-1] == {"role": "assistant", "content": "Changed x to 2."}
    assert {t["function"]["name"] for t in tools} >= {"bash", "read", "edit", "write"}
    assert stats["submit_dropped"] == 2


def test_line_range_read_has_offset_limit_and_the_more_lines_footer():
    view = "OBSERVATION:\nHere's the result of running `cat -n` on /testbed/a.py:\n    10\tdef f():\n    11\t    pass\n"
    chat = Chat().call("str_replace_editor", {"command": "view", "path": "/testbed/a.py", "view_range": [10, 11]}, view)
    messages, _, _, _ = swe_agent(chat.call("submit", {}, None, "Done."))
    assert calls(messages)[0] == ("read", {"filePath": "/testbed/a.py", "offset": 9, "limit": 2})
    assert results(messages)[0].endswith("(File has more lines. Use 'offset' parameter to read beyond line 11)\n</file>")


def test_directory_view_is_the_find_command_the_editor_runs():
    listing = "OBSERVATION:\nHere's the files and directories up to 2 levels deep in /testbed, excluding hidden items:\n/testbed\n/testbed/a.py\n"
    messages, _, _, _ = swe_agent(Chat().call("str_replace_editor", {"command": "view", "path": "/testbed"}, listing).call("submit", {}, None, "Done."))
    name, arguments = calls(messages)[0]
    assert name == "bash" and arguments["command"] == "find /testbed -maxdepth 2 -not -path '*/.*'"
    assert results(messages)[0] == "/testbed\n/testbed/a.py"


def test_persistent_cd_becomes_workdir_of_later_calls():
    chat = (
        Chat()
        .call("bash", {"command": "cd /testbed/src && ls"}, "OBSERVATION:\na.py")
        .call("bash", {"command": "python a.py"}, "OBSERVATION:\nYour last command ran successfully and did not produce any output.")
        .call("submit", {}, None, "Done.")
    )
    messages, _, _, _ = swe_agent(chat)
    (_, first), (_, second) = calls(messages)
    assert first["command"] == "ls" and first["workdir"] == "/testbed/src"
    assert second["workdir"] == "/testbed/src" and results(messages)[1] == ""


def test_failed_attempts_and_shortened_views_are_dropped_with_their_words_moved_on():
    chat = (
        Chat()
        .call("str_replace_editor", {"command": "view", "path": "/testbed/big.py"},
              "OBSERVATION:\n<NOTE>This file is too large to display entirely. Showing abbreviated version.</NOTE>\n     1\timport os",
              "The file is long.")
        .call("str_replace_editor", {"command": "view", "path": "/testbed/a.py"}, VIEW)
        .call("str_replace_editor", {"command": "str_replace", "path": "/testbed/a.py", "old_str": "z", "new_str": "w"},
              "OBSERVATION:\nNo replacement was performed, old_str `z` did not appear verbatim in /testbed/a.py.")
        .call("submit", {}, None, "Done.")
    )
    messages, _, reasoning, stats = swe_agent(chat)
    assert [c[0] for c in calls(messages)] == ["read"]
    assert stats["shortened_view_dropped"] == 1 and stats["failed_edit_dropped"] == 1
    assert any("The file is long." in r["values"] for r in reasoning)


def test_edit_of_a_file_the_session_never_read_is_refused():
    chat = Chat().call("str_replace_editor", {"command": "str_replace", "path": "/testbed/a.py", "old_str": "x", "new_str": "y"},
                       "OBSERVATION:\nThe file /testbed/a.py has been edited.")
    with pytest.raises(ConversionError) as error:
        swe_agent(chat.call("submit", {}, None, "Done."))
    assert error.value.reason == "edit_without_read"


def test_cancelled_command_gets_opencode_timeout_note():
    chat = Chat().call("bash", {"command": "python slow.py"},
                       "OBSERVATION:\nThe command 'python slow.py' was cancelled because it took more than 300 seconds. Please try")
    messages, _, _, _ = swe_agent(chat.call("submit", {}, None, "Done."))
    assert calls(messages)[0][1]["timeout"] == 300000
    assert "exceeding timeout 300000 ms" in results(messages)[0]


def next_chat():
    return Chat(system="You are a programming agent who is provided a github issue")


def test_swe_next_directory_search_is_grep_count_with_the_same_files():
    found = ('Execution output of [search]:\nFound 3 matches for "class Mac" in /workspace:\n'
             './pkg/a.py (2 matches)\n./b.py (1 matches)\nEnd of matches for "class Mac" in /workspace\n')
    chat = next_chat().call("search", {"search_term": "class Mac", "path": "."}, found)
    chat.call("finish", {"command": "submit", "result": "Fixed."}, None)
    messages, _, _, _ = convert_chat("swe-next", chat.messages, [])
    assert "Working directory: /workspace" in messages[0]["content"]
    assert calls(messages)[0][1]["command"].startswith("grep -rFc --exclude-dir='.*' --exclude='.*' -- 'class Mac' .")
    assert results(messages)[0] == "./pkg/a.py:2\n./b.py:1"
    assert messages[-1]["content"] == "Fixed."


def test_swe_next_file_search_and_bash_output():
    chat = (
        next_chat()
        .call("search", {"search_term": "def f", "path": "a.py"},
              'Execution output of [search]:\nMatches for "def f" in a.py:\n3:def f():\n')
        .call("execute_bash", {"cmd": "python a.py"},
              "Exit code: 1\nExecution output of [execute_bash]:\n[STDOUT]\n\nout\n[STDERR]\n\nTraceback\n")
        .call("finish", {"command": "submit", "result": "Fixed."}, None)
    )
    messages, _, _, _ = convert_chat("swe-next", chat.messages, [])
    assert calls(messages)[0][1]["command"] == "grep -n 'def f' a.py"
    assert results(messages) == ["3:def f():", "out\nTraceback"]


def test_swe_next_nudge_and_the_reply_before_it_are_dropped():
    chat = next_chat()
    chat.messages.append({"role": "assistant", "content": "I think it is fixed."})
    chat.messages.append({"role": "user", "content": "NO TOOL CALL DETECTED.\nReply with EXACTLY ONE tool call."})
    chat.call("execute_bash", {"cmd": "ls"}, "Exit code: 0\nExecution output of [execute_bash]:\n[STDOUT]\n\na.py\n[STDERR]\n\n")
    chat.call("finish", {"command": "submit", "result": "Fixed."}, None)
    messages, _, _, stats = convert_chat("swe-next", chat.messages, [])
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert stats["nudged_reply_dropped"] == 1


def test_session_that_stops_before_the_last_result_ends_on_that_call():
    chat = next_chat().call("execute_bash", {"cmd": "ls"}, "Exit code: 0\nExecution output of [execute_bash]:\n[STDOUT]\n\na.py\n[STDERR]\n\n")
    chat.call("execute_bash", {"cmd": "pytest"}, None)
    messages, _, _, _ = convert_chat("swe-next", chat.messages, [])
    assert messages[-1]["role"] == "assistant" and calls(messages[-1:])[0][1]["command"] == "pytest"


def test_harness_talk_and_helpers():
    assert strip_harness_talk("I will use str_replace_editor now. The bug is in f.") == "The bug is in f."
    assert read_output([(1, "a")], 1) == "<file>\n00001| a\n\n(End of file - total 1 lines)\n</file>"
