import json

import pytest

from opencode_terminal import (
    ConversionError,
    apply_to_result,
    clean_observation,
    convert_chat,
    drop_echo_lines,
    joined_output,
    prepare_commands,
    split_output,
    strip_protocol_talk,
)
from standardise_native import checked_row

P = "root@abc:/app# "
TASK = "Task Description:\nFix the bug\n\nCurrent terminal state:\nCurrent Terminal Screen:\nroot@abc:/app#\n\n\n"


def keys(*values):
    return [{"keystrokes": v, "duration": 0.1} for v in values]


def step(number, commands, observation, complete=False, content=None):
    arguments = {"commands": commands}
    if complete:
        arguments["task_complete"] = True
    call = {
        "id": f"pilot_{number}",
        "type": "function",
        "function": {"name": "terminal_step", "arguments": json.dumps(arguments)},
    }
    return [
        {"role": "assistant", "content": content, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": f"pilot_{number}", "content": observation},
    ]


def chat(*turns, end='{"plan": "All done.", "commands": [], "task_complete": true}'):
    messages = [{"role": "system", "content": "override"}, {"role": "user", "content": TASK}]
    for turn in turns:
        messages += turn
    if end:
        messages.append({"role": "assistant", "content": end})
    return messages


def calls(messages):
    return [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for m in messages
        for c in m.get("tool_calls") or []
    ]


def results(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


def test_batch_splits_into_sequential_calls_with_their_own_output():
    observation = f"New Terminal Output:\n\n{P}ls\na.py\n{P}pwd\n/app\n{P}\n\n\n\n"
    messages, tools, _, stats = convert_chat(chat(step(0, keys("ls\n", "pwd\n"), observation)), [])
    assert [a["command"] for _, a in calls(messages)] == ["ls", "pwd"]
    assert results(messages) == ["a.py", "/app"]
    assert all(len(m.get("tool_calls", [])) <= 1 for m in messages)
    assert stats["split_batches"] == 1
    assert {t["function"]["name"] for t in tools} >= {"bash", "read", "write", "edit"}


def test_final_json_reply_becomes_plain_text_and_nothing_of_the_protocol_remains():
    observation = f"New Terminal Output:\n{P}ls\na.py\n{P}"
    messages, _, _, _ = convert_chat(chat(step(0, keys("ls\n"), observation)), [])
    assert messages[-1] == {"role": "assistant", "content": "All done."}
    assert "terminal_step" not in json.dumps(messages)
    assert messages[0]["role"] == "system" and "You are opencode" in messages[0]["content"]
    assert "Working directory: /app" in messages[0]["content"]
    assert messages[1]["content"] == "Fix the bug"


def test_empty_closing_plan_gives_way_to_the_analysis():
    observation = f"New Terminal Output:\n{P}ls\na\n{P}"
    source = chat(step(0, keys("ls\n"), observation), end='{"plan": "No further commands are needed.", "commands": [], "task_complete": true}')
    final = len(source) - 1
    messages = convert_chat(source, [{"message_index": final, "values": ["Check first.\n\nThe bug is fixed and tests pass."]}])[0]
    assert messages[-1]["content"] == "The bug is fixed and tests pass."


def test_completion_dialog_is_removed():
    dialog = (
        "Current terminal state:\nNew Terminal Output:\n\nroot@abc:/app#\n\n\n"
        "Are you sure you want to mark the task as complete? This will trigger your solution "
        'to be graded. If so, include "task_complete": true in your JSON response again.'
    )
    source = chat(step(0, keys("ls\n"), f"New Terminal Output:\n{P}ls\na.py\n{P}"), step(1, [], dialog, True))
    messages, _, _, stats = convert_chat(source, [])
    assert len(calls(messages)) == 1
    assert "Are you sure" not in json.dumps(messages)
    assert stats["completion_dialogs"] == 1


def test_rejected_reply_and_its_feedback_are_dropped():
    source = chat(
        [
            {"role": "assistant", "content": "oops, not json"},
            {"role": "user", "content": "Previous response had parsing errors:\nERROR: No valid JSON"},
        ],
        step(0, keys("ls\n"), f"Previous response had warnings:\nWARNINGS: - Extra text\n\nNew Terminal Output:\n{P}ls\na.py\n{P}"),
    )
    messages, _, _, stats = convert_chat(source, [])
    assert "oops" not in json.dumps(messages) and "Previous response" not in json.dumps(messages)
    assert results(messages) == ["a.py"]
    assert stats["rejected_turns"] == 1


def test_cd_becomes_workdir_and_export_is_carried():
    first = f"New Terminal Output:\n{P}cd src\nroot@abc:/app/src# export A=1\nroot@abc:/app/src# ls\nm.py\nroot@abc:/app/src# "
    second = "New Terminal Output:\nroot@abc:/app/src# pwd\n/app/src\nroot@abc:/app/src# "
    source = chat(step(0, keys("cd src\n", "export A=1\n", "ls\n"), first), step(1, keys("pwd\n"), second))
    messages, _, _, _ = convert_chat(source, [])
    found = [a for _, a in calls(messages)]
    assert [a["command"] for a in found] == ["export A=1 && ls", "export A=1 && pwd"]
    assert all(a["workdir"] == "/app/src" for a in found)


def test_cat_and_heredoc_use_native_tools():
    write = f"New Terminal Output:\n{P}cat > n.py << 'EOF'\n> x = 1\n> EOF\n{P}"
    read = f"New Terminal Output:\n{P}cat n.py\nx = 1\n{P}"
    source = chat(step(0, keys("cat > n.py << 'EOF'\nx = 1\nEOF\n"), write), step(1, keys("cat n.py\n"), read))
    messages, _, _, _ = convert_chat(source, [])
    assert calls(messages) == [
        ("write", {"content": "x = 1\n", "filePath": "/app/n.py"}),
        ("read", {"filePath": "/app/n.py"}),
    ]
    assert results(messages)[1] == "<file>\n00001| x = 1\n\n(End of file - total 1 lines)\n</file>"


def test_write_stays_bash_when_the_file_exists_and_was_not_read():
    # OpenCode refuses to overwrite a file the session has not read
    listing = f"New Terminal Output:\n{P}ls\nn.py\n{P}"
    write = f"New Terminal Output:\n{P}cat > n.py << 'EOF'\n> x = 1\n> EOF\n{P}"
    source = chat(step(0, keys("ls\n"), listing), step(1, keys("cat > n.py << 'EOF'\nx = 1\nEOF\n"), write))
    assert [name for name, _ in calls(convert_chat(source, [])[0])] == ["bash", "bash"]


def test_heredoc_with_expansion_stays_bash():
    write = f"New Terminal Output:\n{P}cat > n.py << EOF\n> x = $HOME\n> EOF\n{P}"
    source = chat(step(0, keys("cat > n.py << EOF\nx = $HOME\nEOF\n"), write))
    assert calls(convert_chat(source, [])[0])[0][0] == "bash"


def test_wait_turn_merges_into_the_running_call():
    running = f"New Terminal Output:\n{P}make\nbuilding"
    done = f"New Terminal Output:\nbuilt\n{P}"
    long_wait = [{"keystrokes": "", "duration": 200}]
    source = chat(step(0, keys("make\n"), running), step(1, long_wait, done))
    messages, _, _, _ = convert_chat(source, [])
    assert results(messages) == ["building\nbuilt"]
    name, arguments = calls(messages)[0]
    assert arguments["timeout"] == 200100 and "_waited" not in arguments


def test_scrolled_off_echo_runs_the_batch_as_one_call():
    source = chat(step(0, keys("ls\n", "pwd\n"), f"New Terminal Output:\na.py\n{P}pwd\n/app\n{P}"))
    messages, _, _, stats = convert_chat(source, [])
    assert calls(messages)[0][1]["command"] == "ls\npwd"
    assert results(messages) == ["a.py\n/app"]
    assert stats["joined_batches"] == 1


@pytest.mark.parametrize(
    "commands,reason",
    [
        (keys("C-c"), "interactive_keystrokes"),
        (keys("q"), "unsubmitted_keystrokes"),
        (keys("vim a.py\n"), "interactive_program"),
        (keys("python3\n"), "interactive_program"),
    ],
)
def test_interactive_use_is_refused(commands, reason):
    with pytest.raises(ConversionError) as error:
        prepare_commands(commands)
    assert error.value.reason == reason


def test_text_typed_but_never_run_is_refused():
    stuck = f"New Terminal Output:\n{P}python3 m.py << 'EOF'\n> 4\n> EOF"
    source = chat(step(0, keys("python3 m.py << 'EOF'\n4\nEOF"), stuck))
    with pytest.raises(ConversionError) as error:
        convert_chat(source, [])
    assert error.value.reason == "unsubmitted_keystrokes"


def test_scripts_that_mention_heredoc_syntax_are_not_merged():
    assert prepare_commands(keys("python3 -c 'print(1 << 2)'\n", "ls\n")) == ["python3 -c 'print(1 << 2)'", "ls"]


def test_rows_without_a_task_are_refused():
    messages = [
        {"role": "system", "content": "Environment"},
        {"role": "user", "content": "Current terminal state:\nCurrent Terminal Screen:\nroot@abc:/app#\n"},
    ] + step(0, keys("ls\n"), f"New Terminal Output:\n{P}ls\na\n{P}")
    with pytest.raises(ConversionError) as error:
        convert_chat(messages, [])
    assert error.value.reason == "no_task"


def test_output_quoting_a_shell_prompt_is_refused():
    observation = f"New Terminal Output:\n{P}cat log\nx\n{P}ls\ny\n{P}"
    source = chat(step(0, keys("cat log\n"), observation))
    with pytest.raises(ConversionError) as error:
        convert_chat(source, [])
    assert error.value.reason == "output_not_attributable"


def test_last_action_without_result_is_kept_as_target():
    call = step(0, keys("ls\n", "pwd\n"), "")[0]
    messages, _, _, _ = convert_chat(chat([call], end=None), [])
    assert messages[-1]["role"] == "assistant"
    assert calls(messages)[0][1]["command"] == "ls\npwd"


@pytest.mark.parametrize(
    "sentence",
    [
        "We need respond JSON only.",
        "Let's craft JSON.",
        "We should output the final JSON with task_complete true.",
        "The previous response had extra text before the JSON object.",
        "Then we can mark the task complete.",
        "We need act via JSON command batches.",
        "This is ready for grading.",
    ],
)
def test_protocol_talk_is_removed(sentence):
    text, removed = strip_protocol_talk(f"Look at the logs. {sentence} Then fix it.")
    assert text == "Look at the logs. Then fix it."
    assert removed == 1


@pytest.mark.parametrize(
    "sentence",
    [
        "Write a script to process the CSV and output JSON to report.json.",
        "Parse the package.json file.",
        "The terminal output shows a partial metrics json output from a command.",
        "Fix the parsing errors in the config.",
        "Python's json module handles this.",
    ],
)
def test_real_work_with_json_is_kept(sentence):
    assert strip_protocol_talk(sentence) == (sentence, 0)


def test_drafted_json_reply_is_removed_whole():
    draft = 'First check.\n\n{"analysis": "a", "plan": "b", "commands": [{"keystrokes": "ls\\n", "duration": 1}]}\n\nThen run.'
    assert strip_protocol_talk(draft)[0] == "First check.\n\nThen run."
    cut_off = 'First check.\n\n{"plan": "b", "commands": [{"keys'
    assert strip_protocol_talk(cut_off)[0] == "First check."


def test_short_plan_stays_visible_and_long_plan_becomes_reasoning():
    observation = f"New Terminal Output:\n{P}ls\na\n{P}"
    long_plan = "1. First do this\n2. Then do that"
    source = chat(
        step(0, keys("ls\n"), observation, content="Check the files."),
        step(1, keys("ls\n"), observation, content=long_plan),
    )
    messages, _, reasoning, _ = convert_chat(source, [])
    turns = [m for m in messages if m.get("tool_calls")]
    assert turns[0]["content"] == "Check the files."
    assert turns[1]["content"] is None
    assert reasoning == [{"message_index": messages.index(turns[1]), "values": [long_plan]}]


def test_reasoning_stays_with_the_first_call_of_a_batch():
    observation = f"New Terminal Output:\n{P}ls\na\n{P}pwd\n/app\n{P}"
    source = chat(step(0, keys("ls\n", "pwd\n"), observation))
    messages, _, reasoning, _ = convert_chat(source, [{"message_index": 2, "values": ["Look around first."]}])
    assert reasoning[0]["values"] == ["Look around first."]
    target = messages[reasoning[0]["message_index"]]
    assert target["role"] == "assistant" and json.loads(target["tool_calls"][0]["function"]["arguments"])["command"] == "ls"


def test_blocking_suffix_is_removed_from_echoes_also_when_wrapped():
    for suffix in ("; tmux wait -S done", "; tm\nux wait -S done"):
        observation = f"New Terminal Output:\n{P}make{suffix}\nbuilt\n{P}"
        messages = convert_chat(chat(step(0, keys("make\n"), observation)), [])[0]
        assert calls(messages)[0][1]["command"] == "make"
        assert results(messages) == ["built"]


def test_commands_replayed_before_the_batch_are_not_its_output():
    outcome = joined_output(f"{P}old\nstale\n{P}ls\nfresh\n{P}", ["ls", "missing"], False)
    assert outcome["output"] == "fresh" and outcome["matched"]


def test_batch_whose_echoes_are_all_gone_behind_old_commands_is_refused():
    observation = f"New Terminal Output:\n{P}old\nstale\n{P}"
    with pytest.raises(ConversionError) as error:
        convert_chat(chat(step(0, keys("ls\n"), observation)), [])
    assert error.value.reason == "output_not_attributable"


def test_clean_observation_and_helpers():
    text, screen = clean_observation("Current Terminal Screen:\nroot@abc:/app# ls\n")
    assert screen and text == "root@abc:/app# ls"
    assert split_output(f"{P}a\n1\n{P}b\n2\n{P}", ["a", "b"])["pieces"] == [("/app", "1"), ("/app", "2")]
    assert split_output(f"{P}a\n1\n{P}", ["a", "b"]) is None
    # A wrapped echo and a heredoc echo still match
    assert split_output(f"{P}echo aaaa\nbbbb\nout\n{P}", ["echo aaaabbbb"])["pieces"][0][1] == "out"
    assert joined_output(f"> x = 1\n> EOF\n{P}", ["cat > f << EOF\nx = 1\nEOF"], True)["output"] == ""
    assert drop_echo_lines("> not ours\nkept", ["ls"]) == "> not ours\nkept"


HEREDOC_SCRIPT = "cat > t.py << 'EOF'\nimport sys\n\nprint(\"Edge tests passed.\")\nEOF"


def test_blank_heredoc_line_echoed_as_bare_prompt_still_splits_the_batch():
    screen = (
        f"New Terminal Output:\n{P}cat > t.py << 'EOF'\n> import sys\n>\n> print(\"Edge tests passed.\")\n> EOF\n"
        f"{P}python3 t.py\nEdge tests passed.\n{P}"
    )
    source = chat(step(0, keys(HEREDOC_SCRIPT + "\n", "python3 t.py\n"), screen))
    messages, _, _, stats = convert_chat(source, [])
    assert [c[0] for c in calls(messages)] == ["write", "bash"]
    assert results(messages)[-1] == "Edge tests passed."
    assert "joined_batches" not in stats


def test_output_that_repeats_a_printed_string_is_kept_after_an_echo():
    # The joined fallback: output right after a heredoc echo is output, even when the
    # script prints that same text
    body = "> import sys\n>\n> print(\"Edge tests passed.\")\n> EOF\nEdge tests passed.\nAll done"
    assert drop_echo_lines(body, [HEREDOC_SCRIPT, "python3 t.py"]) == "Edge tests passed.\nAll done"


def test_soft_wrapped_echo_continuation_is_still_removed():
    command = "cat > f << EOF\nprint('a very long line that the terminal wrapped')\nEOF"
    body = "> print('a very long line that\n the terminal wrapped')\n> EOF\nreal output"
    assert drop_echo_lines(body, [command]) == "real output"


def terminal_row():
    prompt = (
        "You are an AI assistant tasked with solving command-line tasks in a Linux environment.\n\n"
        'Format your response as JSON with the following structure:\n{"analysis": "", "plan": "", '
        '"commands": [{"keystrokes": "ls\\n", "duration": 0.1}], "task_complete": false}\n\n'
    )
    reply = json.dumps(
        {"analysis": "See files.", "plan": "List them.", "commands": keys("ls\n"), "task_complete": False}
    )
    final = json.dumps({"analysis": "Finished.", "plan": "The files are listed.", "commands": [], "task_complete": True})
    return {
        "conversations": [
            {"from": "human", "value": prompt + TASK},
            {"from": "gpt", "value": reply},
            {"from": "human", "value": f"New Terminal Output:\n\n{P}ls\na.py\n{P}\n"},
            {"from": "gpt", "value": final},
        ]
    }


ENTRY = dict(
    repo="test/terminal",
    revision="abc",
    category="code_agentic",
    license="unknown",
    license_status="unknown",
    license_notes="",
    synthetic_status="unknown",
    synthetic_notes="",
    metadata_source="fixture",
)
SPEC = dict(path="fixture", source_config="local", source_split="train")


def test_checked_row_produces_a_valid_native_opencode_session():
    row = checked_row(terminal_row(), ENTRY, SPEC, 0, {}, "think-first")
    conversation = json.loads(row["conversation_json"])
    kinds = [item["type"] for item in conversation["items"]]
    assert kinds[0] == "user" and kinds[-1] in {"reply", "wait"}
    call = next(item for item in conversation["items"] if item["type"] == "call")
    assert call["name"] == "bash" and json.loads(call["payload"])["command"] == "ls"
    result = next(item for item in conversation["items"] if item["type"] == "result")
    assert result["payload"] == "a.py"
    assert "keystrokes" not in row["native_text"] and "terminal_step" not in row["native_text"]
    assert "You are opencode" in conversation["system"]["behavior"]
    provenance = json.loads(row["provenance_json"])
    assert provenance["harness_conversion"]["target"] == "opencode"
    assert json.loads(row["source_record_json"]) == terminal_row()


def test_apply_to_result_records_tools_as_harness_defined():
    result = {"chat": {"messages": chat(step(0, keys("ls\n"), f"New Terminal Output:\n{P}ls\na\n{P}")), "tools": []}}
    apply_to_result(result)
    assert {"name": "bash", "origin": "producer_harness"} in result["tool_provenance"]


def test_echo_without_its_prompt_is_removed_after_an_echo():
    # The prompt of the next command was lost; its lines still follow the previous echo
    first, second = "cat > m.cpp << 'CPP'\nint main() {}\nCPP", "./main << 'EOF'\n3\nEOF"
    body = "> int main() {}\n> CPP\n./main << 'EOF'\n3\nEOF\nanswer"
    assert drop_echo_lines(body, [first, second]) == "answer"


def test_command_without_newline_before_another_is_refused():
    # Without its newline the first command is typed onto the line of the second
    with pytest.raises(ConversionError) as error:
        prepare_commands(keys("printf 'a' | ./solve", "printf 'b' | ./solve\n"))
    assert error.value.reason == "concatenated_keystrokes"
    # The last command of a batch is checked against the output instead
    assert prepare_commands(keys("ls\n", "pwd")) == ["ls", "pwd"]


@pytest.mark.parametrize("harness", ["opencode", "terminus"])
def test_trajectory_that_stops_after_the_first_reply_is_rejected(harness):
    source = terminal_row()
    source["conversations"] = source["conversations"][:2]
    with pytest.raises(ValueError, match="single_reply_trajectory"):
        checked_row(source, ENTRY, SPEC, 0, {}, "think-first", None, harness)


def test_both_versions_of_a_row_get_the_same_thinking_level():
    levels = {
        harness: json.loads(checked_row(terminal_row(), ENTRY, SPEC, 0, {}, "think-first", None, harness)["conversation_json"])["system"]["thinking"]
        for harness in ("opencode", "terminus")
    }
    # The analysis becomes OpenCode reasoning, but the source has no reasoning of its own
    assert levels == {"opencode": "low", "terminus": "low"}
