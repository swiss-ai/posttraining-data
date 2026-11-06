"""Rewrite checked SWE-agent (Sera) and SWE-Next trajectories as OpenCode sessions.

Both releases already have structured calls, one per turn, but for tools OpenCode does not
have. This module maps each call onto the OpenCode tool that does the same thing, with the
output OpenCode would show, and gives the session OpenCode's system prompt and tool schemas
(opencode_harness.json). The task prompt is kept: it names no tool except "the bash tool".

  source call                       OpenCode call
  bash / execute_bash               bash (SWE-agent's shell keeps `cd` and `export`, so they
                                    become `workdir` and a prefix of later commands)
  view of a directory               bash `find PATH -maxdepth 2 -not -path '*/.*'`, the
                                    command both editors run for it
  view of a file, or a line range   read, with OpenCode's numbered output
  create                            write
  str_replace                       edit
  search in a directory (SWE-Next)  bash `grep -rFc` over non-hidden files, which counts the
                                    same lines; its output lists the same files and counts
  search in a file (SWE-Next)       bash `grep -n`, the command the tool runs
  submit / finish                   dropped: the session ends in plain text, the summary the
                                    model wrote when it submitted (SWE-Next: the finish result)

A shortened view (SWE-agent's outline of a large file, SWE-Next's condensed view of a Python
file, a clipped view) has no OpenCode form: read always returns the lines themselves. Such a
step is dropped and its reasoning moves to the next call. SWE-agent's review step after the
first submit is dropped too; any work the model did after it is kept.

Failed attempts are dropped and the model's retry is kept: a refused view (a range past the
end of the file), a failed str_replace or create (OpenCode words those errors differently, and
no OpenCode data shows how), a malformed call, and SWE-Next's "NO TOOL CALL DETECTED" nudge
with the reply before it. A command cancelled for its time limit becomes a bash call with that
timeout and OpenCode's timeout note (SWE-agent keeps no output of such a command).

Rows that cannot be mapped honestly raise ConversionError with a reason: edits refused by
SWE-Next's linter (OpenCode has none), insert and undo_edit (no OpenCode tool), an edit of a
file the session never read (OpenCode refuses it), searches with too many matches, user turns
inside the session, and observations of unknown form.
"""

import json
import posixpath
import re
import shlex

from opencode_terminal import (
    CD_ONLY,
    STATE_ONLY,
    ConversionError,
    describe,
    harness_tools,
    load_harness,
    split_plan,
    system_prompt,
)

SOURCES = {
    "allenai/Sera-4.5A-Full-T1": "swe-agent",
    "allenai/Sera-4.5A-Full-T2": "swe-agent",
    "allenai/Sera-4.6-Lite-T1": "swe-agent",
    "allenai/Sera-4.6-Lite-T2": "swe-agent",
    "TIGER-Lab/SWE-Next-SFT-Trajectories": "swe-next",
}
TOOLS = {
    "swe-agent": {"bash", "str_replace_editor", "submit"},
    "swe-next": {"execute_bash", "file_editor", "search", "finish"},
}
READ_LIMIT = 2000
READ_LINE_LIMIT = 2000

NO_OUTPUT = "Your last command ran successfully and did not produce any output."
DIRECTORY_VIEW = re.compile(r"^Here's the files and directories up to 2 levels deep in (.*?), excluding hidden(?: items)?:\n?")
FILE_VIEW = re.compile(r"^Here's the result of running `cat -n` on (?:the file: )?(.*?):\n")
SHORTENED = re.compile(r"too large to display entirely|<response clipped>|Here is a condensed view for file|\.\.\. eliding lines")
ADJUSTED = re.compile(r"^NOTE: adjusted (?:out-of-bounds )?view_range.*?\(file length: (\d+) lines?\)\.?\n", re.M)
ADJUSTED_BEFORE = re.compile(r"^\[note\] adjusted view_range .*?\(file has (\d+) lines?\)\n")
NUMBERED = re.compile(r"^ *(\d+)(?:[\t ](.*))?$")
EDIT_DONE = re.compile(r"^The file (.*?) has been edited\.")
CREATE_DONE = re.compile(r"^File created (?:successfully )?at:? \S")
SWE_NEXT_HEADER = re.compile(r"^(?:Exit code: (-?\d+)\n)?Execution output of \[(\w+)\]:\n")
DIRECTORY_MATCHES = re.compile(r'^Found \d+ matches for ".*" in (.*):\n(.*?)\nEnd of matches for ', re.S)
FILE_MATCHES = re.compile(r'^Matches for ".*" in (.*?):\n')
NO_MATCHES = re.compile(r'^No matches found for ".*" in ')
SWE_AGENT_TIMEOUT = re.compile(r"^The command '.*' was cancelled because it took more than (\d+) seconds", re.S)
SWE_NEXT_TIMEOUT = re.compile(r"Command timed out after (\d+)s")
FAILED = re.compile(
    r"^(?:ERROR:|Invalid `|No replacement was performed|The path .* does not exist|File already exists|Error|"
    r"The `view_range` parameter is not allowed|Parameter `\w+` is required|Ran into \[Errno|"
    r"The parent directory .* does not exist|usage: )"
)
# SWE-Next's release cuts long observations in the middle; the cut is kept as it is
CUT = re.compile(r"\n?-{50}\n<Observation truncated in middle for saving context>\n-{50}\n?")
CUT_TEXT = "\n--------------------------------------------------\n<Observation truncated in middle for saving context>\n--------------------------------------------------\n"
FAILED_SEARCH = re.compile(
    r"^(?:\[search\]|usage: search|/bin/sh: |Directory '.*' not found|More than \d+ files matched|Error executing grep)"
)
NUDGE = re.compile(r"^\s*(?:NO TOOL CALL DETECTED|You forgot to use a function call)")
TIMEOUT_NOTE = "<bash_metadata>\nbash tool terminated command after exceeding timeout {ms} ms\n</bash_metadata>"
LEADING_CD = re.compile(r"^cd\s+(\S+)\s*(?:&&|;|\n)\s*(.+)$", re.S)
# Sentences that are about the source harness rather than the task
HARNESS_TALK = re.compile(
    r"str_replace_editor|file_editor|execute_bash|view_range|\bsubmit(?:ted|ting)?\b|"
    r"\bfinish (?:tool|command|function)\b|condensed view|abbreviated version|OBSERVATION",
    re.I,
)
SENTENCE = re.compile(r"(?<=[.!?])[ \t]+|\n+")


def source_harness(repo, chat):
    kind = SOURCES.get(repo)
    names = {t["function"]["name"] for t in chat.get("tools") or []}
    if kind and names and names <= TOOLS[kind]:
        return kind
    return None


def strip_harness_talk(text):
    """Remove sentences that name the source harness's tools; keep the rest as written."""
    if not text or not HARNESS_TALK.search(text):
        return text
    pieces = re.split(r"((?<=[.!?])[ \t]+|\n+)", text)
    kept = []
    for piece, separator in zip(pieces[::2], pieces[1::2] + [""]):
        if not HARNESS_TALK.search(piece):
            kept.append(piece + separator)
    return re.sub(r"\n{3,}", "\n\n", "".join(kept)).strip()


def read_output(rows, total, cut_after=()):
    """OpenCode's read result for numbered `rows`; total None if unknown. A cut made by the
    release after a row is shown where it was."""
    parts = []
    for n, line in rows:
        parts.append(f"{n:05d}| {line[:READ_LINE_LIMIT] + ('...' if len(line) > READ_LINE_LIMIT else '')}")
        if n in cut_after:
            parts[-1] += CUT_TEXT.rstrip("\n")
    body = "\n".join(parts)
    last = rows[-1][0] if rows else 0
    if total is not None and last >= total:
        footer = f"(End of file - total {total} lines)"
    else:
        footer = f"(File has more lines. Use 'offset' parameter to read beyond line {last})"
    return f"<file>\n{body}\n\n{footer}\n</file>"


def numbered_lines(text):
    """([(number, line)], numbers followed by a cut) of a `cat -n` listing; None if a line is
    not numbered in order (a gap is allowed only at a cut)."""
    rows, cuts = [], set()
    for piece_number, piece in enumerate(CUT.split(text)):
        if piece_number and rows:
            cuts.add(rows[-1][0])
        for raw in piece.split("\n"):
            match = NUMBERED.match(raw)
            if not match:
                return None
            number = int(match[1])
            if rows and number != rows[-1][0] + 1 and rows[-1][0] not in cuts:
                return None
            rows.append((number, match[2] or ""))
    return rows, cuts


class Converter:
    def __init__(self, kind, cwd, harness):
        self.kind, self.root, self.cwd, self.harness = kind, cwd, cwd, harness
        self.env, self.messages, self.reasoning, self.carried = [], [], [], []
        self.lengths, self.read = {}, set()
        self.counter, self.stats, self.summary = 0, {}, None
        self.ended_on_call = False

    def count(self, key):
        self.stats[key] = self.stats.get(key, 0) + 1

    def path(self, path):
        return posixpath.normpath(posixpath.join(self.cwd, path))

    def emit(self, text, thoughts, name, arguments, result):
        """One OpenCode call and its result; a result of None ends the session on the call."""
        self.counter += 1
        call = {
            "id": f"call_{self.counter}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }
        visible, thought = split_plan(strip_harness_talk(text))
        values = [v for v in self.carried + thoughts + [thought] if v]
        self.carried = []
        if values:
            self.reasoning.append({"message_index": len(self.messages), "values": list(dict.fromkeys(values))})
        self.messages.append({"role": "assistant", "content": visible, "tool_calls": [call]})
        if result is not None:
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        self.count(f"{name}_calls")

    def drop(self, text, thoughts, reason):
        """A step with no OpenCode form: its words go with the next call."""
        self.carried += [t for t in thoughts + [strip_harness_talk(text)] if t]
        self.count(reason)

    def bash(self, text, thoughts, command, output, timeout=None):
        command = command.strip("\n")
        if self.kind == "swe-agent":
            # A persistent shell: a leading cd moves every later command
            lead = LEADING_CD.match(command)
            if CD_ONLY.match(command) and not output.strip():
                target = CD_ONLY.match(command)[1] or "~"
                if target.startswith("~"):
                    raise ConversionError("unmapped_shell_state", command)
                self.cwd = self.path(target)
                return self.drop(text, thoughts, "cd_to_workdir")
            if STATE_ONLY.match(command) and not output.strip():
                self.env.append(command)
                return self.drop(text, thoughts, "state_prefixed")
            if lead and not lead[1].startswith("~"):
                self.cwd = self.path(lead[1])
                command = lead[2]
        arguments = {"command": " && ".join(self.env + [command]), "description": describe(command)}
        if self.cwd != self.root:
            arguments["workdir"] = self.cwd
        if timeout:
            arguments["timeout"] = timeout
        self.emit(text, thoughts, "bash", arguments, output)

    def view(self, text, thoughts, arguments, output):
        path = self.path(arguments["path"])
        directory = DIRECTORY_VIEW.match(output)
        if directory:
            listing = output[directory.end():].strip("\n")
            command = f"find {shlex.quote(arguments['path'])} -maxdepth 2 -not -path '*/.*'"
            return self.bash(text, thoughts, command, listing)
        if SHORTENED.search(output) or arguments.get("concise"):
            return self.drop(text, thoughts, "shortened_view_dropped")
        before = ADJUSTED_BEFORE.match(output)
        if before:
            self.lengths[path] = int(before[1])
            output = output[before.end():]
        header = FILE_VIEW.match(output)
        if not header and (FAILED.match(output) or not output.strip() or output.startswith("NOTE: file is empty")):
            # The editor refused the view (a range past the end, a missing path); the model
            # retries, and OpenCode would have shown something else, so the attempt goes
            return self.drop(text, thoughts, "failed_view_dropped")
        if not header:
            raise ConversionError("unknown_observation", output[:200])
        body = output[header.end():]
        adjusted = ADJUSTED.search(body)
        if adjusted:
            self.lengths[path] = int(adjusted[1])
            body = body[: adjusted.start()] + body[adjusted.end():]
        body = body.rstrip("\n")
        parsed = numbered_lines(body) if body else ([], set())
        if parsed is None:
            raise ConversionError("unknown_observation", body[:200])
        rows, cuts = parsed
        if cuts:
            self.count("cut_view_kept")
        span = arguments.get("view_range")
        call = {"filePath": path}
        if span:
            first = rows[0][0] if rows else int(span[0])
            if first > 1:
                call["offset"] = first - 1
            if int(span[1]) != -1:
                call["limit"] = int(span[1]) - int(span[0]) + 1
            else:
                self.lengths[path] = rows[-1][0] if rows else first - 1
            total = self.lengths.get(path)
        else:
            if rows and rows[-1][0] > READ_LIMIT:
                raise ConversionError("file_longer_than_read_limit", path)
            total = rows[-1][0] if rows else 0
            self.lengths[path] = total
        self.read.add(path)
        self.emit(text, thoughts, "read", call, read_output(rows, total, cuts))

    def editor(self, text, thoughts, arguments, output):
        command = arguments.get("command")
        if command == "view":
            return self.view(text, thoughts, arguments, output)
        if "syntax error" in output[:200] and "Your proposed edit" in output:
            raise ConversionError("edit_refused_by_linter", arguments.get("path", ""))
        path = self.path(arguments.get("path", ""))
        if command == "create":
            if not CREATE_DONE.match(output):
                if FAILED.match(output):
                    return self.drop(text, thoughts, "failed_edit_dropped")
                raise ConversionError("unknown_observation", output[:200])
            self.read.add(path)
            self.lengths[path] = len((arguments.get("file_text") or "").rstrip("\n").split("\n"))
            return self.emit(text, thoughts, "write", {"content": arguments.get("file_text") or "", "filePath": path}, "")
        if command == "str_replace":
            if not EDIT_DONE.match(output):
                if FAILED.match(output):
                    # OpenCode words a failed edit differently; the model's retry stays
                    return self.drop(text, thoughts, "failed_edit_dropped")
                raise ConversionError("unknown_observation", output[:200])
            if path not in self.read:
                # OpenCode refuses to edit a file the session has not read
                raise ConversionError("edit_without_read", path)
            self.lengths.pop(path, None)
            new = arguments.get("new_str") or ""
            return self.emit(
                text, thoughts, "edit", {"filePath": path, "oldString": arguments.get("old_str") or "", "newString": new}, ""
            )
        raise ConversionError("unsupported_editor_command", command or "")

    def search(self, text, thoughts, arguments, output):
        term, given = arguments.get("search_term") or "", arguments.get("path") or "."
        found = FILE_MATCHES.match(output)
        if found:
            command = f"grep -n {shlex.quote(term)} {shlex.quote(given)}"
            return self.bash(text, thoughts, command, output[found.end():].strip("\n"))
        command = (
            f"grep -rFc --exclude-dir='.*' --exclude='.*' -- {shlex.quote(term)} {shlex.quote(given)} "
            "| grep -v ':0$'"
        )
        if NO_MATCHES.match(output):
            return self.bash(text, thoughts, command, "")
        found = DIRECTORY_MATCHES.match(output)
        if not found and FAILED_SEARCH.match(output):
            # A malformed call, a missing directory, or too many matches: the model retries
            return self.drop(text, thoughts, "failed_call_dropped")
        if not found:
            raise ConversionError("unmapped_search_output", output[:200])
        base = posixpath.normpath(posixpath.join(self.cwd, given))
        lines = []
        for entry in CUT.sub("\n\0CUT\n", found[2]).split("\n"):
            if entry == "\0CUT":
                lines.append(CUT_TEXT.strip("\n"))
                continue
            if not entry:
                continue
            match = re.match(r"^(.*) \((\d+) matches\)$", entry)
            if not match:
                raise ConversionError("unmapped_search_output", entry)
            absolute = posixpath.normpath(posixpath.join(self.cwd, match[1]))
            lines.append(f"{posixpath.join(given, posixpath.relpath(absolute, base))}:{match[2]}")
        self.bash(text, thoughts, command, "\n".join(lines))

    def last_call(self, text, thoughts, name, arguments):
        """The final call of a session that stops before its result: the training target."""
        self.ended_on_call = True
        self.count("ended_on_last_call")
        if name in ("bash", "execute_bash"):
            command = arguments.get("command") if name == "bash" else arguments.get("cmd")
            return self.bash(text, thoughts, command or "", None)
        command, path = arguments.get("command"), self.path(arguments.get("path") or "")
        if name in ("str_replace_editor", "file_editor"):
            if command == "create":
                return self.emit(text, thoughts, "write", {"content": arguments.get("file_text") or "", "filePath": path}, None)
            if command == "str_replace" and path in self.read:
                return self.emit(text, thoughts, "edit", {
                    "filePath": path, "oldString": arguments.get("old_str") or "", "newString": arguments.get("new_str") or ""
                }, None)
            if command == "view" and not arguments.get("concise"):
                call = {"filePath": path}
                span = arguments.get("view_range")
                if span and int(span[0]) > 1:
                    call["offset"] = int(span[0]) - 1
                if span and int(span[1]) != -1:
                    call["limit"] = int(span[1]) - int(span[0]) + 1
                return self.emit(text, thoughts, "read", call, None)
        raise ConversionError("unmapped_final_call", f"{name} {command or ''}")

    def observation(self, content):
        content = content or ""
        if self.kind == "swe-agent":
            content = content.removeprefix("OBSERVATION:\n").removeprefix("OBSERVATION:")
            if content.strip() == NO_OUTPUT:
                return "", None
            timeout = SWE_AGENT_TIMEOUT.match(content)
            if timeout:
                # SWE-agent keeps no output of a cancelled command
                return TIMEOUT_NOTE.format(ms=int(timeout[1]) * 1000), int(timeout[1]) * 1000
            return content, None
        header = SWE_NEXT_HEADER.match(content)
        if not header:
            raise ConversionError("unknown_observation", content[:200])
        body = content[header.end():]
        # Editor errors start on a new line ("\nERROR: ERROR: ...")
        return (body.lstrip("\n") if FAILED.match(body.lstrip("\n")) else body), header[2]

    def bash_output(self, output):
        """SWE-Next prints stdout and stderr under headers; OpenCode shows them together."""
        match = re.match(r"^\[STDOUT\]\n\n(.*?)\n\[STDERR\]\n\n(.*)$", output, re.S)
        if not match:
            return output.strip("\n"), None
        out, err = match[1].strip("\n"), match[2].strip("\n")
        timeout = SWE_NEXT_TIMEOUT.search(err)
        if timeout:
            err = err[: timeout.start()].strip("\n")
        text = "\n".join(part for part in (out, err) if part)
        if timeout:
            note = TIMEOUT_NOTE.format(ms=int(timeout[1]) * 1000)
            return (text + "\n\n\n" + note) if text else note, int(timeout[1]) * 1000
        return text, None

    def step(self, text, thoughts, name, arguments, result):
        if name in ("submit", "finish"):
            if name == "finish":
                self.summary = (arguments.get("result") or "").strip() or self.summary
            elif self.summary is None:
                self.summary = (text or "").strip() or None
            # The review prompt and the answer to it are about the harness, not the task
            self.carried = []
            return self.count("submit_dropped")
        if result is None:
            return self.last_call(text, thoughts, name, arguments)
        output, timeout = self.observation(result)
        if name in ("bash", "execute_bash"):
            command = arguments.get("command") if name == "bash" else arguments.get("cmd")
            if name == "execute_bash":
                output, timeout = self.bash_output(output)
            if timeout:
                self.count("timeout_mapped")
            return self.bash(text, thoughts, command or "", output, timeout)
        if name in ("str_replace_editor", "file_editor"):
            return self.editor(text, thoughts, arguments, output)
        if name == "search":
            return self.search(text, thoughts, arguments, output)
        raise ConversionError("unknown_tool", name)


def working_directory(kind, messages):
    """SWE-agent works in /testbed. SWE-Next's shell starts where its searches say it is."""
    if kind == "swe-agent":
        return "/testbed"
    calls = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            calls[call["id"]] = call
        if message.get("role") == "tool":
            call = calls.get(message.get("tool_call_id"))
            if call and call["function"]["name"] == "search":
                arguments = json.loads(call["function"]["arguments"])
                found = re.search(r'^(?:Found \d+ matches|No matches found) for ".*" in (/\S*?):?$', message.get("content") or "", re.M)
                if found and (arguments.get("path") or ".") in (".", "./"):
                    return found[1]
    return "/testbed"


def convert_chat(kind, messages, reasoning_by_message, harness=None):
    harness = harness or load_harness()
    thoughts = {}
    for record in reasoning_by_message or []:
        thoughts.setdefault(record["message_index"], []).extend(v for v in record.get("values", []) if v)
    users = [m for m in messages if m["role"] == "user"]
    if not users or not (users[0].get("content") or "").strip():
        raise ConversionError("no_task")
    nudges = {i for i, m in enumerate(messages) if m["role"] == "user" and NUDGE.match(m.get("content") or "")}
    others = [m for m in users[1:] if not NUDGE.match(m.get("content") or "")]
    if others:
        raise ConversionError("user_turn_inside_session", (others[0].get("content") or "")[:120])
    converter = Converter(kind, working_directory(kind, messages), harness)
    results = {m.get("tool_call_id"): m.get("content") for m in messages if m["role"] == "tool"}
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        own = [strip_harness_talk(t) for t in thoughts.get(index, [])]
        calls = message.get("tool_calls") or []
        if not calls:
            if index + 1 in nudges:
                # A reply without a call that the harness sent back: not a turn OpenCode has
                converter.count("nudged_reply_dropped")
                continue
            converter.summary = (message.get("content") or "").strip() or converter.summary
            continue
        for number, call in enumerate(calls):
            arguments = json.loads(call["function"]["arguments"] or "{}")
            text = message.get("content") if number == 0 else None
            converter.step(text, own if number == 0 else [], call["function"]["name"], arguments, results.get(call["id"]))
    if not any(m["role"] == "tool" for m in converter.messages):
        raise ConversionError("no_executed_command")
    final = strip_harness_talk(converter.summary)
    if converter.ended_on_call:
        pass
    elif final:
        values = list(dict.fromkeys(converter.carried))
        if values:
            converter.reasoning.append({"message_index": len(converter.messages), "values": values})
        converter.messages.append({"role": "assistant", "content": final})
    else:
        # No closing words: end on the last call as the training target
        converter.messages.pop()
        converter.count("ended_on_last_call_no_summary")
    seen = json.dumps(converter.messages)
    is_git = kind == "swe-agent" or bool(re.search(r"\bgit (?:diff|status|log|show|stash|checkout)\b", seen))
    cwd = converter.root
    output = [{"role": "system", "content": system_prompt(harness, cwd, is_git)}, dict(users[0])]
    output += converter.messages
    # The session follows the system prompt and the task
    reasoning = [dict(r, message_index=r["message_index"] + 2) for r in converter.reasoning]
    return output, harness_tools(harness, cwd), reasoning, converter.stats


def apply_to_result(result, kind, harness=None):
    harness = harness or load_harness()
    messages, tools, reasoning, stats = convert_chat(
        kind, result["chat"]["messages"], result.get("reasoning_by_message"), harness
    )
    result["chat"] = {"messages": messages, "tools": tools}
    result["reasoning_by_message"] = reasoning
    result["tool_provenance"] = [{"name": t["function"]["name"], "origin": "producer_harness"} for t in tools]
    result["ends_with_tool_call"] = bool(messages[-1].get("tool_calls"))
    result["harness_conversion"] = {"target": "opencode", "source": kind, "snapshot": harness["source"], "statistics": stats}
    return result
