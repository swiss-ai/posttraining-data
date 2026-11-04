"""ATIF trajectory tar packages become chat rows; local fixtures only, no network."""

import io
import json
import tarfile

import pytest

from atif_trajectories import convert, iter_tar_rows, steps_to_messages

PROMPT = 'Format your response as JSON ... "keystrokes" ...'


def add(archive, name, data):
    raw = data if isinstance(data, bytes) else data.encode()
    info = tarfile.TarInfo(name)
    info.size = len(raw)
    archive.addfile(info, io.BytesIO(raw))


def add_dir(archive, name):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    archive.addfile(info)


JSON_REPLY = {"analysis": "look", "plan": "ls", "commands": [{"keystrokes": "ls\n", "duration": 0.1}]}
JSON_TRAJECTORY = {
    "schema_version": "ATIF-v1.7",
    "agent": {"name": "terminus-2", "version": "2.0.0", "model_name": "m"},
    "steps": [
        {"step_id": 1, "source": "user", "message": PROMPT},
        {
            "step_id": 2,
            "source": "agent",
            "message": json.dumps(JSON_REPLY),
            "observation": {"results": [{"content": "New Terminal Output:\n$ ls\n"}]},
        },
        {"step_id": 3, "source": "agent", "message": json.dumps({**JSON_REPLY, "task_complete": True})},
    ],
}
TOOL_TRAJECTORY = {
    "schema_version": "ATIF-v1.7",
    "agent": {"name": "terminus-2", "version": "2.0.0", "model_name": "glm"},
    "steps": [
        {"step_id": 1, "source": "user", "message": PROMPT},
        {
            "step_id": 2,
            "source": "agent",
            "message": "Analysis: a\nPlan: b",
            "reasoning_content": "thinking",
            "tool_calls": [
                {"tool_call_id": "c1", "function_name": "bash_command",
                 "arguments": {"keystrokes": "pwd\n", "duration": 0.1}},
                {"tool_call_id": "c2", "function_name": "bash_command",
                 "arguments": {"keystrokes": "ls\n", "duration": 0.5}},
            ],
            "observation": {"results": [{"content": "New Terminal Output:\n/app\n"}]},
        },
        {
            "step_id": 3,
            "source": "agent",
            "message": "Analysis: done\nPlan: none",
            "tool_calls": [{"tool_call_id": "c3", "function_name": "mark_task_complete", "arguments": {}}],
        },
    ],
}


def make_tars(tmp_path):
    """One tar per layout; the second also holds a gold run without a trajectory."""
    json_tar, tool_tar = tmp_path / "trajectories-0.tar", tmp_path / "trials-0.tar"
    with tarfile.open(json_tar, "w") as a:
        add_dir(a, "trajectories/t1")
        add(a, "trajectories/t1/trajectory.json", json.dumps(JSON_TRAJECTORY))
        add(a, "trajectories/t1/verifier/reward.txt", "1\n")
    with tarfile.open(tool_tar, "w") as a:
        add(a, "gold_oracle_0/result.json", "{}")
        add(a, "gold_oracle_0/exception.txt", "setup failed")
        add(a, "run_glm_1/agent/trajectory.json", json.dumps(TOOL_TRAJECTORY))
        add(a, "run_glm_1/verifier/reward.txt", "0\n")
        add(a, "run_glm_2/agent/trajectory.json", json.dumps(TOOL_TRAJECTORY))
    return json_tar, tool_tar


def test_json_reply_is_kept_verbatim_and_observation_becomes_user_turn():
    messages = steps_to_messages(JSON_TRAJECTORY["steps"])
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1]["content"] == json.dumps(JSON_REPLY)
    assert messages[2]["content"] == "New Terminal Output:\n$ ls\n"
    assert "reasoning_content" not in messages[1]


def test_tool_calls_become_ordered_blocks_and_reasoning_stays_a_field():
    messages = steps_to_messages(TOOL_TRAJECTORY["steps"])
    assistant = messages[1]
    assert assistant["reasoning_content"] == "thinking"
    assert assistant["content"].startswith("Analysis: a\nPlan: b\n<tool_call>")
    blocks = [
        json.loads(b.split("</tool_call>")[0].strip())
        for b in assistant["content"].split("<tool_call>")[1:]
    ]
    assert [(b["name"], b["arguments"]["keystrokes"]) for b in blocks] == [
        ("bash_command", "pwd\n"),
        ("bash_command", "ls\n"),
    ]
    assert blocks[0]["arguments"]["duration"] == 0.1
    final = messages[-1]
    assert final["role"] == "assistant"
    assert json.loads(final["content"].split("<tool_call>")[1].split("</tool_call>")[0]) == {
        "name": "mark_task_complete",
        "arguments": {},
    }


def test_unknown_step_source_is_an_error():
    with pytest.raises(ValueError, match="source"):
        steps_to_messages([{"step_id": 1, "source": "tool", "message": "x"}])


def test_both_layouts_are_read_and_trials_without_a_trajectory_are_counted(tmp_path):
    json_tar, tool_tar = make_tars(tmp_path)
    counts = {}
    with open(json_tar, "rb") as f:
        rows = list(iter_tar_rows(f, "trajectories-0.tar", counts))
    assert len(rows) == 1 and rows[0]["trajectory_id"] == "t1"
    assert rows[0]["reward"] == 1.0 and rows[0]["model_name"] == "m"
    assert counts == {"trials": 1, "no_trajectory": 0, "rows": 1}

    counts = {}
    with open(tool_tar, "rb") as f:
        rows = list(iter_tar_rows(f, "trials-0.tar", counts))
    assert [r["trajectory_id"] for r in rows] == ["run_glm_1", "run_glm_2"]
    assert rows[0]["reward"] == 0.0 and rows[1]["reward"] is None
    assert counts == {"trials": 3, "no_trajectory": 1, "rows": 2}


def test_limit_stops_early_and_convert_writes_jsonl_shards(tmp_path):
    json_tar, tool_tar = make_tars(tmp_path)
    with open(tool_tar, "rb") as f:
        assert len(list(iter_tar_rows(f, "t", limit=1))) == 1
    counts = convert([json_tar, tool_tar], tmp_path / "out", rows_per_file=2)
    assert counts["written"] == 3 and counts["files"] == 2 and counts["no_trajectory"] == 1
    rows = [
        json.loads(line)
        for path in sorted((tmp_path / "out").glob("part-*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    assert [r["trajectory_id"] for r in rows] == ["t1", "run_glm_1", "run_glm_2"]
    assert all(r["messages"][0]["content"] == PROMPT for r in rows)
    limited = convert([json_tar, tool_tar], tmp_path / "limited", limit=2)
    assert limited["written"] == 2
