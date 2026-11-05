"""Rehydrate Terminus-2 (ATIF) trajectory tar packages into chat rows.

Some releases (Recursive-Task-Synthesis-Trajectories, the GLM-5.3 rollouts) ship the
agent trajectories inside tar shards, one directory per trial, as ATIF trajectory.json
files, with only a catalog table next to them. The standardiser reads saved datasets,
JSONL and Parquet, so this module turns the trajectories into JSONL rows with a
`messages` array first. The result is the same shape the existing Terminus-2 sources
already use, so `standardise_native.py` maps it with its terminal-protocol handling.

Two layouts are read (both are Terminus-2, ATIF-v1.7):

  trajectories/<id>/trajectory.json         agent message = the raw JSON reply
  <execution>/agent/trajectory.json         agent message = "Analysis:/Plan:" text, the
                                            commands are structured tool_calls
                                            (bash_command, mark_task_complete) and the
                                            reasoning is in reasoning_content. This is
                                            what the harness parsed from the model's JSON
                                            reply; the reply itself is not in the release.

Conversion rules, per ATIF step:
  source user/system   one message with the step text
  source agent         one assistant message; reasoning_content stays a message field.
                       Without tool calls the text is kept as it is (the raw JSON reply, or
                       an unparseable reply the harness rejected). With tool calls the reply
                       is rebuilt as the JSON object the prompt asks for and the harness
                       parsed: {"analysis", "plan", "commands": the bash_command arguments in
                       order, "task_complete": true if mark_task_complete was called}, with
                       2-space indents. The ATIF message and tool calls stay on the message as
                       atif_message / atif_tool_calls.
  observation          one user message with the result contents (the terminal output)
A trial with a step that cannot be rebuilt exactly (other text than "Analysis: ...\nPlan:
...", another tool, mark_task_complete before a command) is counted as `not_rebuilt` and
skipped. Harness feedback turns stay: their warnings are about the commands, which the
rebuilt reply keeps. Nothing else is dropped or reworded. Trials without an agent trajectory (the gold-solution
runs of the rollouts) are counted and skipped. The verifier reward is kept as `reward`
and is not used for filtering here.

Usage:
  python atif_trajectories.py --input data/trajectories-00000.tar --output out/ [--limit 5]
"""

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path

TRAJECTORY_FILES = ("trajectory.json", "agent/trajectory.json")


def trial_prefix(name):
    """Split a tar member name into (trial directory, path inside the trial)."""
    parts = name.strip("/").split("/")
    if parts[0] == "trajectories" and len(parts) >= 2:
        return "/".join(parts[:2]), "/".join(parts[2:])
    return parts[0], "/".join(parts[1:])


ANALYSIS_PLAN = re.compile(r"Analysis: ?(.*?)\nPlan: ?(.*)", re.S)


class NotRebuilt(ValueError):
    """An ATIF agent step whose JSON reply cannot be rebuilt exactly."""


def rebuilt_reply(text, calls):
    """The Terminus-2 JSON reply behind an ATIF agent step with structured tool calls."""
    match = ANALYSIS_PLAN.fullmatch(text or "")
    if not match or text.count("\nPlan:") != 1:
        raise NotRebuilt(f"agent message is not 'Analysis: ...\\nPlan: ...': {str(text)[:80]!r}")
    commands, complete = [], False
    for call in calls:
        name, arguments = call.get("function_name"), call.get("arguments") or {}
        if complete:
            raise NotRebuilt("mark_task_complete is not the last call")
        if name == "mark_task_complete" and arguments == {}:
            complete = True
        elif name == "bash_command" and isinstance(arguments.get("keystrokes"), str):
            commands.append(arguments)
        else:
            raise NotRebuilt(f"unsupported tool call {name!r}")
    reply = {"analysis": match[1], "plan": match[2], "commands": commands}
    if complete:
        reply["task_complete"] = True
    return json.dumps(reply, ensure_ascii=False, indent=2)


def steps_to_messages(steps):
    """ATIF steps -> chat messages (see the module docstring for the rules)."""
    messages = []
    for step in steps:
        source, text = step.get("source"), step.get("message")
        if source in ("user", "system"):
            messages.append({"role": source, "content": text if isinstance(text, str) else ""})
        elif source == "agent":
            calls = step.get("tool_calls") or []
            if calls:
                message = {
                    "role": "assistant",
                    "content": rebuilt_reply(text, calls),
                    "atif_message": text,
                    "atif_tool_calls": calls,
                }
            else:
                message = {"role": "assistant", "content": text if isinstance(text, str) else ""}
            reasoning = step.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                message["reasoning_content"] = reasoning
            messages.append(message)
            observation = step.get("observation") or {}
            results = [
                r["content"]
                for r in observation.get("results") or []
                if isinstance(r, dict) and isinstance(r.get("content"), str)
            ]
            if results:
                messages.append({"role": "user", "content": "\n".join(results)})
        else:
            raise ValueError(f"Unknown ATIF step source {source!r} in step {step.get('step_id')}")
    return messages


def parse_reward(raw):
    try:
        return float(raw.decode().strip())
    except (ValueError, AttributeError):
        return None


def trial_row(prefix, parts, source):
    """One JSONL row for a trial, or None if the trial has no usable agent trajectory."""
    raw = parts.get("trajectory")
    if raw is None:
        return None
    trajectory = json.loads(raw)
    messages = steps_to_messages(trajectory.get("steps") or [])  # may raise NotRebuilt
    if not any(m["role"] == "assistant" for m in messages):
        return None
    agent = trajectory.get("agent") or {}
    return {
        "trajectory_id": prefix.rsplit("/", 1)[-1],
        "member_prefix": prefix,
        "source_tar": source,
        "schema_version": trajectory.get("schema_version"),
        "agent": agent.get("name"),
        "agent_version": agent.get("version"),
        "model_name": agent.get("model_name"),
        "reward": parse_reward(parts.get("reward")),
        "has_exception": bool(parts.get("exception")),
        "n_steps": len(trajectory.get("steps") or []),
        "messages": messages,
    }


def iter_tar_rows(fileobj, source, counts=None, limit=None):
    """Stream a tar once and yield one row per trial that has an agent trajectory.

    Members of one trial are adjacent in the archives; a trial is emitted when the next
    one starts. `counts` collects trials seen, trials without a trajectory and rows.
    """
    counts = counts if counts is not None else {}
    for key in ("trials", "no_trajectory", "not_rebuilt", "rows"):
        counts.setdefault(key, 0)
    current, parts = None, {}
    emitted = 0

    def flush():
        nonlocal emitted
        if current is None:
            return None
        counts["trials"] += 1
        try:
            row = trial_row(current, parts, source)
        except NotRebuilt:
            counts["not_rebuilt"] += 1
            return None
        if row is None:
            counts["no_trajectory"] += 1
            return None
        counts["rows"] += 1
        emitted += 1
        return row

    with tarfile.open(fileobj=fileobj, mode="r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            prefix, inner = trial_prefix(member.name)
            if prefix != current:
                row = flush()
                if row is not None:
                    yield row
                    if limit is not None and emitted >= limit:
                        return
                current, parts = prefix, {}
            if inner in TRAJECTORY_FILES:
                parts["trajectory"] = archive.extractfile(member).read()
            elif inner == "verifier/reward.txt":
                parts["reward"] = archive.extractfile(member).read()
            elif inner == "exception.txt" and member.size:
                parts["exception"] = True
        row = flush()
        if row is not None and (limit is None or emitted <= limit):
            yield row


def convert(tars, output, rows_per_file=20000, limit=None):
    """Convert tar files to JSONL shards in `output`; returns the counts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts = {"tars": 0}
    written, shard, handle = 0, 0, None
    try:
        for tar in tars:
            counts["tars"] += 1
            with open(tar, "rb") as fileobj:
                remaining = None if limit is None else limit - written
                if remaining is not None and remaining <= 0:
                    break
                for row in iter_tar_rows(fileobj, Path(tar).name, counts, remaining):
                    if handle is None or written and written % rows_per_file == 0:
                        if handle:
                            handle.close()
                        handle = open(output / f"part-{shard:05d}.jsonl", "w", encoding="utf-8")
                        shard += 1
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
    finally:
        if handle:
            handle.close()
    counts["files"] = shard
    counts["written"] = written
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, help="tar files, or folders holding *.tar")
    parser.add_argument("--output", required=True, help="folder for the JSONL shards")
    parser.add_argument("--limit", type=int, help="stop after this many trajectories")
    parser.add_argument("--rows-per-file", type=int, default=20000)
    args = parser.parse_args()
    tars = []
    for item in args.input:
        path = Path(item)
        tars += sorted(path.rglob("*.tar")) if path.is_dir() else [path]
    if not tars:
        sys.exit("No tar files found")
    print(json.dumps(convert(tars, args.output, args.rows_per_file, args.limit)))


if __name__ == "__main__":
    main()
