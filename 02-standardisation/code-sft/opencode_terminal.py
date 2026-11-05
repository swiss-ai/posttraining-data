"""Rewrite checked Terminus-2 terminal trajectories as OpenCode sessions.

`openai_chat.convert` maps Terminus-2 replies onto one `terminal_step` wrapper tool whose
arguments are tmux keystrokes and whose results are terminal screens. No harness we train
for has that tool. This module turns such a checked chat into what OpenCode shows a model:
its system prompt and tool schemas (opencode_harness.json), `bash` / `read` / `write` calls,
plain tool output, and a final turn of plain text with no tool call.

What changes, per Terminus-2 feature:

  keystroke batch       one sequential call per command, when the echoed commands in the
                        terminal output show which output belongs to which command. If they
                        do not, the batch becomes one multi-line bash call with all output.
  terminal screen       harness notices, prompts, echoed commands and blank padding removed
  duration / waiting    no argument: a bash call blocks. Wait turns are merged into the
                        running call, whose `timeout` covers them when over the 2 min default
  `cd`, prompt path     `workdir` argument (OpenCode bash calls do not share a shell)
  `export`, `source`    prefixed to later commands, for the same reason
  `cat FILE`            read call with OpenCode's numbered output, if the output is complete
  `cat > F << EOF`      write call, if the body is literal and F was not seen before (OpenCode
                        refuses to overwrite a file the session has not read)
  task_complete         dropped, with its confirmation dialog. The final plan becomes the text
  analysis / plan       analysis is reasoning. A short plan stays as text with the call, a
                        long one becomes reasoning, as OpenCode text next to a call is short
  rejected JSON reply   the failed turn and the harness feedback are dropped
  format talk           JSON reply drafts and sentences about the protocol are removed from
                        reasoning and text

Rows that cannot be mapped honestly raise ConversionError and are quarantined with the
reason: interactive programs and control keys, keystrokes never submitted or typed onto the
next command's line, handoff prompts,
rows without a task, and output that belongs to no known command. Such rows stay in the
rejected.jsonl ledger, and every converted row keeps its source in source_record_json.

Terminal output soft-wraps long lines; that cannot be undone, so read is not used when a
line looks wrapped. Exit codes are not in the source and OpenCode's output has none here.
"""

import json
from functools import lru_cache
from pathlib import Path
import posixpath
import re

TERMINAL_TOOL = "terminal_step"
DEFAULT_TIMEOUT_SECONDS = 120
VISIBLE_TEXT_LIMIT = 240


class ConversionError(ValueError):
    def __init__(self, reason, detail=""):
        self.reason = reason
        super().__init__(
            json.dumps(
                {"status": "terminal_not_convertible", "reason": reason, "detail": detail[:200]}
            )
        )


@lru_cache(maxsize=1)
def load_harness():
    return json.loads(Path(__file__).with_name("opencode_harness.json").read_text())


# ---------------------------------------------------------------- terminal text

PROMPT = re.compile(
    r"^(?:\([^()\n]*\) ?)*"
    r"(?:(?P<user>[\w.-]+)@[\w.-]+:(?P<cwd>[^\n#$]*)[#$]|(?:ba)?sh-[\d.]+[#$])[ ]?",
    re.M,
)
NOTICE_START = re.compile(
    r"^(?:Previous response had warnings:|WARNINGS:|\[SYSTEM WARNING\]|"
    r"AUTO-CORRECTED:)[^\n]*(?:\n|$)",
    re.M,
)
SCREEN_MARKER = re.compile(r"^(?:Current Terminal Screen:|Current terminal state:)[ \t]*\n?", re.M)
OUTPUT_MARKER = re.compile(r"^New Terminal Output:[ \t]*\n?", re.M)
CONFIRM_DIALOG = re.compile(r"\n*Are you sure you want to mark the task as complete\?.*\Z", re.S)
# The harness appends this to a blocking command, and the terminal echoes it, wrapped or not
BLOCKING_SUFFIX = re.compile(r"\n?".join(re.escape(c) for c in "; tmux wait -S done"))
REJECTED = "Previous response had parsing errors:"
HANDOFF = re.compile(r"You are about to hand off your work|provide a comprehensive summary", re.I)


def clean_observation(text):
    """Remove what the Terminus-2 harness adds. Returns (terminal text, is a full screen)."""
    text = CONFIRM_DIALOG.sub("", (text or "").replace("\r", ""))
    text = NOTICE_START.sub("", text)
    screen = bool(SCREEN_MARKER.search(text)) and not OUTPUT_MARKER.search(text)
    text = SCREEN_MARKER.sub("", text)
    text = OUTPUT_MARKER.sub("", text)
    text = BLOCKING_SUFFIX.sub("", text)
    return text.strip("\n"), screen


def match_echo(text, position, command):
    """End of `command` echoed at `position`, allowing soft wraps and `> ` continuations."""
    for character in command:
        if character == "\n":
            if text[position:position + 1] != "\n":
                return None
            position += 1
            if text[position:position + 2] == "> ":
                position += 2
            elif text[position:position + 1] == ">" and text[position + 1:position + 2] in ("\n", ""):
                # An empty continuation line is echoed as a bare ">"
                position += 1
            else:
                prompt = PROMPT.match(text, position)
                if prompt:
                    position = prompt.end()
            continue
        while text[position:position + 1] == "\n" and character != "\n":
            position += 1
        if text[position:position + 1] != character:
            return None
        position += 1
    # The echo has to reach the end of its line
    end = text.find("\n", position)
    end = len(text) if end < 0 else end
    return end if not text[position:end].strip() else None


def prompt_cwd(match):
    if not match.groupdict().get("cwd"):
        return None
    cwd, user = match["cwd"].strip(), match["user"]
    home = "/root" if user == "root" else f"/home/{user}"
    if cwd == "~":
        return home
    if cwd.startswith("~/"):
        return home + cwd[1:]
    return cwd if cwd.startswith("/") else None


def split_output(text, commands):
    """Output per command, or None when the echoes do not account for the whole batch."""
    prompts = list(PROMPT.finditer(text))
    found, index = [], 0
    for command in commands:
        while index < len(prompts):
            match = prompts[index]
            index += 1
            end = match_echo(text, match.end(), command)
            if end is not None:
                found.append((match, end))
                break
        else:
            return None
    pieces = []
    for number, (match, end) in enumerate(found):
        stop = found[number + 1][0].start() if number + 1 < len(found) else len(text)
        pieces.append((prompt_cwd(match), text[end:stop].strip("\n")))
    # What follows the last echo may only be its output and the idle prompt
    cwd, last = pieces[-1]
    tail = list(PROMPT.finditer(last))
    running = True
    if tail:
        final = tail[-1]
        if last[final.end():].strip() or len(tail) > 1:
            return None
        running = False
        pieces[-1] = (cwd, last[: final.start()].strip("\n"))
    return {"pieces": pieces, "running": running, "next_cwd": prompt_cwd(tail[-1]) if tail else None}


def drop_echo_lines(body, commands):
    """Remove echoed lines of a command whose first line scrolled off the screen.

    A `> ` line goes only if it repeats a line of the batch, so quoted output stays. Right
    after such an echo, a line without the `> ` also goes if it is a whole line of the batch
    (an echo whose prompt was lost) or completes the echo line before it into one (an echo
    soft-wrapped at the terminal width). Output that only contains text of the batch, such
    as a string the script prints, stays."""
    lines = {line.strip() for command in commands for line in command.split("\n")}
    kept, in_echo, partial = [], False, None
    for line in body.split("\n"):
        text = line[2:] if line.startswith("> ") else "" if line == ">" else None
        if text is not None and (
            text.strip() in lines or any(known.startswith(text.strip()) for known in lines if text.strip())
        ):
            in_echo, partial = True, text if text.strip() not in lines else None
            continue
        if in_echo and text is None and line.strip():
            if line.strip() in lines:
                partial = None
                continue
            if partial is not None:
                joined = (partial + line).strip()
                if any(known == joined or known.startswith(joined) for known in lines):
                    partial = joined if joined not in lines else None
                    continue
        in_echo, partial = False, None
        kept.append(line)
    return "\n".join(kept).strip("\n")


def echoes_batch_line(typed, commands):
    """Is this text after a prompt a line of a batch command? (wrapped echoes are cut short)"""
    return any(
        line.strip().startswith(typed)
        for command in commands
        for line in command.split("\n")
    )


def joined_output(text, commands, screen):
    """Output of a batch that runs as one call: everything visible minus prompts and echoes."""
    output, cursor, matched, stale = [], 0, False, False
    next_cwd, ended = None, False
    for match in PROMPT.finditer(text):
        if match.start() < cursor:
            continue
        before = text[cursor:match.start()]
        ends = [e for e in (match_echo(text, match.end(), c) for c in commands) if e is not None]
        line_end = text.find("\n", match.end())
        line_end = len(text) if line_end < 0 else line_end
        typed = text[match.end():line_end].strip()
        if ends:
            # After a replay of old commands, text before the first echo is not this batch's
            output = [] if stale and not matched else output + [before]
            matched, cursor, ended = True, max(ends), False
        elif not typed:
            output.append(before)
            cursor, ended = line_end, True
            next_cwd = prompt_cwd(match) or next_cwd
        elif echoes_batch_line(typed, commands):
            output = [] if stale and not matched else output + [before]
            matched, cursor, ended = True, line_end, False
        elif not matched:
            # Earlier commands shown again before this batch starts: not its output
            output, cursor, ended = [], line_end, False
            stale = True
        else:
            # A command this batch did not type: what follows belongs to something else
            raise ConversionError("output_not_attributable", typed)
    output.append(text[cursor:])
    body = "\n".join(p.strip("\n") for p in output if p.strip("\n"))
    return {
        "output": drop_echo_lines(body, commands),
        "matched": matched,
        "stale": stale,
        "running": not ended,
        "next_cwd": next_cwd,
    }


# ---------------------------------------------------------------- commands

CONTROL_KEY = re.compile(r"^(?:C-[a-zA-Z\\]|M-[a-zA-Z]|Escape|Enter|Up|Down|Left|Right|Tab|BSpace)(?:\s|$)")
INTERACTIVE = re.compile(
    r"^(?:sudo\s+)?(?:vim?|nvim|nano|emacs|less|more|man|top|htop|watch|tmux|screen|ssh|ftp|gdb|pdb)(?:\s|$)"
    r"|^(?:sudo\s+)?(?:python[\d.]*|ipython|node|irb|mysql|psql|sqlite3|bash|sh|R)\s*$"
)
HEREDOC = re.compile(r"<<-?[ \t]*(['\"]?)([\w.-]+)\1")
STATE_ONLY = re.compile(
    r"^(?:export\s+\w+=[^;&|]*|\w+=[^\s;&|]*|source\s+[^;&|]+|\.\s+[^;&|]+|(?:conda|mamba)\s+activate\s+[^;&|]+|set\s+[-+]\w+)$"
)
CD_ONLY = re.compile(r"^cd(?:\s+(\S+))?\s*$")
CD_PREFIX = re.compile(r"^cd\s+(/[^\s;&|]+)\s*&&\s*(.+)$", re.S)
CAT_FILE = re.compile(r"^cat\s+(['\"]?)([\w./~+@-]+)\1$")
CAT_WRITE = re.compile(r"^cat\s*>\s*(['\"]?)([\w./~+@-]+)\1\s*<<-?\s*(['\"]?)([\w.-]+)\3\n(.*\n)?\4$", re.S)
WRAPPED_WIDTHS = {79, 80, 100, 120, 160, 200}

DESCRIPTIONS = {
    "ls": "Lists directory contents",
    "cat": "Shows file contents",
    "head": "Shows the start of a file",
    "tail": "Shows the end of a file",
    "sed": "Edits or prints file text with sed",
    "awk": "Processes text with awk",
    "grep": "Searches text for a pattern",
    "rg": "Searches files for a pattern",
    "find": "Finds files by name or type",
    "wc": "Counts lines and words",
    "mkdir": "Creates a directory",
    "rm": "Removes files",
    "cp": "Copies files",
    "mv": "Moves files",
    "chmod": "Changes file permissions",
    "pwd": "Prints the working directory",
    "echo": "Prints text or writes it to a file",
    "printf": "Prints formatted text",
    "git": "Runs a git command",
    "pip": "Runs pip to manage Python packages",
    "pip3": "Runs pip to manage Python packages",
    "python": "Runs a Python script",
    "python3": "Runs a Python script",
    "pytest": "Runs the test suite with pytest",
    "npm": "Runs an npm command",
    "node": "Runs a Node.js script",
    "make": "Builds with make",
    "cargo": "Runs a cargo command",
    "go": "Runs a Go command",
    "gcc": "Compiles C code with gcc",
    "g++": "Compiles C++ code with g++",
    "apt-get": "Manages system packages with apt-get",
    "apt": "Manages system packages with apt",
    "curl": "Sends an HTTP request with curl",
    "wget": "Downloads a file with wget",
    "tar": "Packs or unpacks an archive",
    "sleep": "Waits for a moment",
    "bash": "Runs a shell script",
    "sh": "Runs a shell script",
    "which": "Locates a program",
    "diff": "Compares two files",
    "sort": "Sorts lines of text",
    "jq": "Processes JSON with jq",
}


def describe(command):
    """OpenCode requires a short description of every bash call; the source has none."""
    if "\n" in command.strip() and not HEREDOC.search(command):
        return "Runs a sequence of shell commands"
    words = re.sub(r"^(?:sudo\s+|\w+=\S+\s+)+", "", command.strip()).split()
    first = posixpath.basename(words[0]) if words else ""
    if first == "cat" and ">" in command:
        return "Writes a file from the shell"
    return DESCRIPTIONS.get(first, f"Runs {first}" if first else "Runs a shell command")


def prepare_commands(raw):
    """Keystroke entries -> shell commands. Raises on anything that is not a shell command."""
    commands, pending, terminator = [], None, None
    for position, entry in enumerate(raw):
        keys = entry.get("keystrokes", "")
        if pending is not None:
            pending += keys
            if keys.strip() == terminator:
                commands.append(pending.rstrip())
                pending = None
            continue
        if not keys.strip():
            continue
        stripped = keys.strip()
        if CONTROL_KEY.match(stripped) or re.search(r"[\x00-\x08\x0b-\x1f\x7f]", keys):
            raise ConversionError("interactive_keystrokes", stripped)
        if not keys.endswith("\n") and len(stripped) <= 2:
            # Single keys answer a running program. Longer text without a newline ran only
            # if a prompt came back after it, which the callers check against the output.
            raise ConversionError("unsubmitted_keystrokes", stripped)
        if not keys.endswith("\n") and any(later.get("keystrokes", "").strip() for later in raw[position + 1 :]):
            # tmux types the next entry onto the same line, so the shell runs the two joined
            raise ConversionError("concatenated_keystrokes", stripped)
        if INTERACTIVE.search(stripped.split("\n", 1)[0]):
            raise ConversionError("interactive_program", stripped)
        heredoc = HEREDOC.search(keys)
        if (
            heredoc
            and not re.search(rf"^\s*{re.escape(heredoc[2])}\s*$", keys, re.M)
            and any(later.get("keystrokes", "").strip() == heredoc[2] for later in raw[position + 1 :])
        ):
            pending, terminator = keys, heredoc[2]
            continue
        commands.append(keys.rstrip())
    return commands


def resolve(path, cwd):
    if path.startswith("~"):
        return None
    return posixpath.normpath(posixpath.join(cwd, path))


def looks_wrapped(text):
    return any(len(line) in WRAPPED_WIDTHS for line in text.split("\n"))


def numbered_file(text):
    """The read tool's output format in the harness snapshot."""
    lines = text.split("\n")
    body = "\n".join(f"{n:05d}| {line}" for n, line in enumerate(lines, 1))
    return f"<file>\n{body}\n\n(End of file - total {len(lines)} lines)\n</file>"


# ---------------------------------------------------------------- reasoning and text

# File formats are real work in these tasks ("parse the JSON report"), so JSON only counts
# as protocol talk when it is about the reply, and never next to a file or a program.
JSON_SUBJECT = re.compile(r"\.json|\bjq\b|\bfile|pars(?:e|ing)|script|endpoint|\bapi\b|schema of|config", re.I)
JSON_PROTOCOL = re.compile(
    r"\b(?:respond\w*|response|reply|replies|answer|output|format\w*|return|provide|produce|send|"
    r"craft|emit|wrap\w*|act|wants?|need)\b(?:\s+\w+){0,3}?\s+json\b"
    r"|\bjson\s+(?:response|reply|format|only|object|commands?|batch\w*|action\w*)\b"
    r"|\bjson with commands\b",
    re.I,
)
PROTOCOL = re.compile(
    r"task_complete|keystrokes?\b|terminal_step|mark_task_complete|bash_command|"
    r"previous (?:response|reply|answer)\b[^.\n]{0,40}(?:error|warning|invalid|json|format|parse)|"
    r"extra text (?:detected )?before|required (?:json|response) format|"
    r"new terminal output|current terminal screen|\"duration\"|command with duration|"
    r"\"(?:analysis|plan|commands)\"|"
    r"\b(?:commands? (?:array|list|field|empty)|empty commands)\b|"
    r"\bmark(?:ing|ed)?\s+(?:the\s+|this\s+)?task\s+(?:as\s+)?complete|"
    r"\bgrad(?:ed|ing|er)\b|are you sure you want|\btrue in json\b",
    re.I,
)
DRAFT_KEYS = {"commands", "task_complete", "keystrokes", "analysis", "plan"}
DRAFT_OPENING = re.compile(r"\{\s*\"(?:analysis|plan|commands|task_complete)\"\s*:.*?(?:\n\s*\n|\Z)", re.S)
DRAFT_SUFFIX = re.compile(r"\"\s*,\s*\"(?:plan|commands|task_complete)\"\s*:.*?(?:\n\s*\n|\Z)", re.S)
EMPTY_FENCE = re.compile(r"```(?:json)?\s*```")
ONLY_PUNCTUATION = re.compile(r"^[\s\[\]{},\"]+$", re.M)
SENTENCE = re.compile(r"(?<=[.!?:])[ \t]+|(\n+)")


# Output from the terminal is padded to the screen height
BLANK_PADDING = re.compile(r"\n{5,}")


def is_protocol_talk(sentence):
    if PROTOCOL.search(sentence):
        return True
    return bool(JSON_PROTOCOL.search(sentence)) and not JSON_SUBJECT.search(sentence)


def strip_json_drafts(text):
    """Models often draft the whole JSON reply while thinking. Remove such a draft whole."""
    decoder, parts, position, removed = json.JSONDecoder(), [], 0, 0
    for match in re.finditer(r"\{", text):
        if match.start() < position:
            continue
        try:
            payload, length = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(payload, dict) and DRAFT_KEYS & set(payload):
            parts.append(text[position:match.start()])
            position = match.start() + length
            removed += 1
    text = "".join(parts) + text[position:]
    # Drafts that are cut off or not valid JSON end with their paragraph
    for pattern in (DRAFT_OPENING, DRAFT_SUFFIX):
        text, count = pattern.subn("\n\n", text)
        removed += count
    if removed:
        text = ONLY_PUNCTUATION.sub("", EMPTY_FENCE.sub("", text))
    return text, removed


def strip_protocol_talk(text):
    """Drop JSON reply drafts and whole sentences that talk about the Terminus-2 reply format."""
    if not text:
        return "", 0
    text, removed = strip_json_drafts(text)
    kept, position = [], 0
    for match in SENTENCE.finditer(text):
        sentence = text[position:match.start()]
        separator = match.group(0)
        position = match.end()
        if is_protocol_talk(sentence):
            removed += 1
            if "\n" in separator:
                kept.append(separator)
        else:
            kept.append(sentence + separator)
    sentence = text[position:]
    if is_protocol_talk(sentence):
        removed += 1
    else:
        kept.append(sentence)
    return re.sub(r"\n{3,}", "\n\n", "".join(kept)).strip(), removed


# The required plan field of a last turn often says only this: no message to end a session on
NOTHING_TO_SAY = re.compile(
    r"^\W*(?:(?:there (?:is|are) )?(?:no|none)\b[^.\n]{0,40}\b(?:commands?|actions?|steps?|work|changes?)\b[^.\n]{0,40}"
    r"|none|n/?a|done|(?:the )?task (?:is )?(?:now )?(?:complete|completed|done|finished))\W*$",
    re.I,
)


def split_plan(text):
    """Text next to an OpenCode call is short (median 77 characters, 90% under 241 in
    Nemotron-SFT-OpenCode-v1). A Terminus-2 plan is a required field and often a long list,
    so anything longer is reasoning, not something said to the user. Returns (text, thought)."""
    if not text:
        return None, None
    if len(text) <= VISIBLE_TEXT_LIMIT and "\n" not in text.strip():
        return text, None
    return None, text


def final_payload(content):
    """A Terminus-2 JSON reply inside text: (text before it, the payload) or (content, None)."""
    if not isinstance(content, str):
        return content, None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            payload, end = decoder.raw_decode(content[match.start():])
        except ValueError:
            continue
        if isinstance(payload, dict) and ("commands" in payload or "task_complete" in payload):
            if not content[match.start() + end:].strip():
                return content[: match.start()].strip(), payload
    return content, None


# ---------------------------------------------------------------- the session


class Session:
    def __init__(self, cwd):
        self.root = cwd
        self.cwd = cwd
        self.env = []
        self.seen = ""
        self.read = set()
        self.messages = []
        self.reasoning = []
        self.counter = 0
        self.carried = []
        self.stats = {}
        self.last_call = None
        self.last_running = False

    def count(self, key, by=1):
        if by:
            self.stats[key] = self.stats.get(key, 0) + by

    def assistant(self, text, thoughts, call=None):
        thoughts = self.carried + thoughts
        self.carried = []
        message = {"role": "assistant", "content": text or (None if call else "")}
        if call:
            message["tool_calls"] = [call]
        values = list(dict.fromkeys(t for t in thoughts if t))
        if values:
            self.reasoning.append(
                {"message_index": len(self.messages) + 1, "values": values}
            )
        self.messages.append(message)

    def call(self, name, arguments):
        self.counter += 1
        return {
            "id": f"call_{self.counter}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }

    def bash(self, command, cwd=None):
        command = command.strip("\n")
        cwd = cwd or self.cwd
        prefix = CD_PREFIX.match(command)
        if prefix and "\n" not in prefix[1]:
            cwd, command = prefix[1], prefix[2]
        full = " && ".join(self.env + [command])
        arguments = {"command": full, "description": describe(command)}
        if cwd != self.root:
            arguments["workdir"] = cwd
        return self.call("bash", arguments)

    def native_call(self, command, output, complete, cwd):
        """read / write for the shell commands that have a faithful native form."""
        cat = CAT_FILE.match(command)
        if cat and complete and output and not looks_wrapped(output):
            if "No such file" not in output and "Is a directory" not in output:
                path = resolve(cat[2], cwd)
                if path:
                    self.read.add(path)
                    self.count("read_calls")
                    return self.call("read", {"filePath": path}), numbered_file(output)
        write = CAT_WRITE.match(command)
        if write and not output.strip():
            body = write[5] or ""
            literal = write[3] or not re.search(r"[$`\\]", body)
            path = resolve(write[2], cwd)
            known = path and (path in self.read or posixpath.basename(path) not in self.seen)
            if literal and known:
                self.read.add(path)
                self.count("write_calls")
                return self.call("write", {"content": body, "filePath": path}), ""
        return None

    def run(self, command, output, complete, cwd, text, thoughts):
        output = BLANK_PADDING.sub("\n\n", output)
        if PROMPT.search(output):
            raise ConversionError("output_not_attributable", command)
        native = None if self.env else self.native_call(command, output, complete, cwd)
        call, result = native or (self.bash(command, cwd), output)
        self.assistant(text, thoughts, call)
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        self.seen += "\n" + command + "\n" + output
        self.last_call = (call, len(self.messages) - 1)

    def extend_last_result(self, extra, seconds):
        """Output from wait turns belongs to the call that was still running."""
        call, index = self.last_call
        message = self.messages[index]
        if extra:
            extra = BLANK_PADDING.sub("\n\n", extra)
            if PROMPT.search(extra):
                raise ConversionError("output_not_attributable", extra)
            message["content"] = (message["content"] + "\n" + extra).strip("\n")
            self.seen += "\n" + extra
        arguments = json.loads(call["function"]["arguments"])
        arguments["_waited"] = arguments.get("_waited", 0) + seconds
        if call["function"]["name"] == "bash" and arguments["_waited"] > DEFAULT_TIMEOUT_SECONDS:
            arguments["timeout"] = int(arguments["_waited"] * 1000)
        call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)

    def step(self, commands_raw, observation, text, thoughts):
        seconds = sum(float(c.get("duration") or 0) for c in commands_raw)
        commands = prepare_commands(commands_raw)
        terminal, screen = clean_observation(observation)

        if not commands:
            # A wait: the result continues the running call, otherwise it was a no-op turn
            self.count("wait_turns")
            if self.last_call and self.last_running:
                outcome = joined_output(terminal, [], screen)
                extra = outcome["output"]
                previous = self.messages[self.last_call[1]]["content"]
                if screen and previous:
                    tail = previous.rsplit("\n", 1)[-1]
                    if tail and tail in extra:
                        extra = extra.split(tail, 1)[1].strip("\n")
                    elif extra in previous:
                        extra = ""
                self.extend_last_result(extra, seconds)
                self.last_running = outcome["running"]
                self.cwd = outcome["next_cwd"] or self.cwd
            return

        if self.last_call and self.last_running and not screen:
            first_prompt = PROMPT.search(terminal)
            if first_prompt and terminal[: first_prompt.start()].strip():
                self.extend_last_result(terminal[: first_prompt.start()].strip("\n"), 0)
                terminal = terminal[first_prompt.start():]
                self.count("late_output_merged")

        split = split_output(terminal, commands)
        if split and not all(
            k.get("keystrokes", "").endswith("\n") for k in commands_raw if k.get("keystrokes", "").strip()
        ) and split["running"]:
            raise ConversionError("unsubmitted_keystrokes", commands[-1])
        if split is None:
            self.joined(commands, commands_raw, terminal, screen, text, thoughts, seconds)
            return

        self.count("split_batches" if len(commands) > 1 else "single_commands")
        total = len(split["pieces"])
        first = True
        for number, (command, (cwd, output)) in enumerate(zip(commands, split["pieces"])):
            cwd = cwd or self.cwd
            complete = number + 1 < total or not split["running"]
            state = self.state_change(command, output, cwd)
            if state:
                continue
            self.run(command, output, complete, cwd, text if first else None, thoughts if first else [])
            first = False
            self.cwd = cwd
        if first:
            # Only cd / export in this batch: its reasoning leads to the next real call
            self.carried += thoughts + ([text] if text else [])
        self.cwd = split["next_cwd"] or self.cwd
        self.last_running = split["running"]
        if split["running"] and self.last_call:
            self.extend_last_result("", seconds)

    def state_change(self, command, output, cwd):
        """cd and export change state only: no call, later calls carry the state."""
        if output.strip():
            return False
        target = CD_ONLY.match(command)
        if target:
            self.count("cd_to_workdir")
            return True
        if STATE_ONLY.match(command):
            self.env.append(command)
            self.count("state_prefixed")
            return True
        return False

    def joined(self, commands, raw, terminal, screen, text, thoughts, seconds):
        outcome = joined_output(terminal, commands, screen)
        if not outcome["matched"] and outcome["stale"]:
            # Only earlier commands are visible, so the output of this batch is unknown
            raise ConversionError("output_not_attributable", commands[0])
        if not all(k.get("keystrokes", "").endswith("\n") for k in raw if k.get("keystrokes", "").strip()):
            if outcome["running"] or not outcome["matched"]:
                raise ConversionError("unsubmitted_keystrokes", commands[-1])
        self.count("joined_batches")
        if not outcome["matched"]:
            self.count("no_echo_batches")
        self.run("\n".join(commands), outcome["output"], False, self.cwd, text, thoughts)
        self.cwd = outcome["next_cwd"] or self.cwd
        self.last_running = outcome["running"]
        if outcome["running"]:
            self.extend_last_result("", seconds)


def split_task(content):
    """(task text, the initial screen) from the first user message."""
    content = content or ""
    if "Format your response as JSON" in content or '"keystrokes"' in content:
        if "Task Description:" not in content:
            raise ConversionError("protocol_in_task")
        content = content.split("Task Description:", 1)[1]
    content = re.sub(r"^\s*Task Description:\s*", "", content)
    marker = re.search(r"(?:^|\n)\s*(?:Current terminal state:|Current Terminal Screen:)", content)
    if marker:
        return content[: marker.start()].strip(), content[marker.end():]
    return content.strip(), ""


def initial_cwd(screen, messages):
    for match in PROMPT.finditer(screen):
        if prompt_cwd(match):
            return prompt_cwd(match)
    for message in messages:
        if message.get("role") == "tool":
            match = next((m for m in PROMPT.finditer(message.get("content") or "") if prompt_cwd(m)), None)
            if match:
                return prompt_cwd(match)
    return "/app"


def system_prompt(harness, cwd, is_git):
    return (
        harness["system_prompt"]
        + "Here is some useful information about the environment you are running in:\n<env>\n"
        + f"  Working directory: {cwd}\n"
        + f"  Is directory a git repo: {'yes' if is_git else 'no'}\n"
        + "  Platform: linux\n"
        + f"  Today's date: {harness['date']}\n"
        + "</env>\n<files>\n  \n</files>"
    )


def harness_tools(harness, cwd):
    return json.loads(json.dumps(harness["tools"]).replace("{workdir}", cwd))


def clean_thoughts(values, session):
    cleaned = []
    for value in values:
        text, removed = strip_protocol_talk(value)
        session.count("protocol_sentences_removed", removed)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def uses_terminal_protocol(chat):
    return any(
        call["function"]["name"] == TERMINAL_TOOL
        for message in chat["messages"]
        for call in message.get("tool_calls") or []
    )


def convert_chat(messages, reasoning_by_message, harness=None):
    """A checked terminal chat -> (OpenCode messages, tools, reasoning records, statistics)."""
    harness = harness or load_harness()
    user_index = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
    if user_index is None:
        raise ConversionError("no_task")
    task, screen = split_task(messages[user_index]["content"])
    if not task:
        # LiteCoder-Terminal-World-Model: only an environment and a screen, nothing to do
        raise ConversionError("no_task")
    cwd = initial_cwd(screen, messages)
    session = Session(cwd)
    thoughts_by_index = {}
    for record in reasoning_by_message or []:
        thoughts_by_index.setdefault(record["message_index"], []).extend(record.get("values", []))

    final_text, final_thoughts, ended_with_call = None, [], False
    index = user_index + 1
    session.messages.append({"role": "user", "content": task})
    while index < len(messages):
        message = messages[index]
        role = message["role"]
        if role == "user":
            content = message.get("content") or ""
            if HANDOFF.search(content):
                raise ConversionError("handoff_prompt")
            if content.startswith(REJECTED) or OUTPUT_MARKER.search(content) or SCREEN_MARKER.search(content):
                # Feedback to a reply that was not an action: never happens with real tool calls
                session.count("rejected_turns")
                index += 1
                continue
            session.messages.append({"role": "user", "content": content})
            index += 1
            continue
        if role == "tool":
            raise ConversionError("unexpected_tool_message")
        if role != "assistant":
            raise ConversionError("mid_conversation_instructions")
        thoughts = clean_thoughts(thoughts_by_index.get(index, []), session)
        calls = message.get("tool_calls") or []
        if not calls:
            before, payload = final_payload(message.get("content"))
            following = messages[index + 1] if index + 1 < len(messages) else None
            if following and following["role"] == "user" and following["content"].startswith(REJECTED):
                session.count("rejected_turns")
                index += 2
                continue
            if payload is not None and payload.get("commands") and following:
                raise ConversionError("unparsed_action")
            text = "\n\n".join(v for v in (before, (payload or {}).get("plan")) if v)
            text, removed = strip_protocol_talk(text)
            session.count("protocol_sentences_removed", removed)
            if following is None:
                final_text, final_thoughts = text, thoughts
            elif text:
                session.assistant(text, thoughts)
            index += 1
            continue
        if len(calls) != 1 or calls[0]["function"]["name"] != TERMINAL_TOOL:
            raise ConversionError("unsupported_call")
        arguments = json.loads(calls[0]["function"]["arguments"])
        text, removed = strip_protocol_talk(message.get("content") or "")
        session.count("protocol_sentences_removed", removed)
        following = messages[index + 1] if index + 1 < len(messages) else None
        plan_text = text
        text, moved = split_plan(text)
        if moved:
            thoughts = thoughts + [moved]
            session.count("plans_moved_to_reasoning")
        observed = following is not None and following["role"] == "tool"
        observation = following["content"] if observed else None
        raw = arguments.get("commands") or []
        if observed and observation.startswith(REJECTED):
            session.count("rejected_turns")
            index += 2
            continue
        if not observed:
            if prepare_commands(raw):
                # The last action of a trajectory has no result: a legal training target
                session.assistant(text, thoughts, session.bash("\n".join(prepare_commands(raw))))
                ended_with_call = True
            else:
                final_text, final_thoughts = plan_text, [t for t in thoughts if t != plan_text]
            index += 1
            if following is not None:
                raise ConversionError("unexpected_message_after_call")
            continue
        if arguments.get("task_complete") and not prepare_commands(raw):
            # The "are you sure" dialog; the trajectory goes on or ends with the confirmation
            session.count("completion_dialogs")
            if index + 2 >= len(messages):
                final_text, final_thoughts = plan_text, [t for t in thoughts if t != plan_text]
            index += 2
            continue
        session.step(raw, observation, text, thoughts)
        index += 2

    if not ended_with_call:
        if not any(m["role"] == "tool" for m in session.messages):
            raise ConversionError("no_executed_command")
        if (not final_text or NOTHING_TO_SAY.match(final_text)) and final_thoughts:
            final_text = final_thoughts[-1].rsplit("\n\n", 1)[-1]
            session.count("final_text_from_reasoning")
        if final_text:
            session.assistant(final_text, final_thoughts)
        elif session.messages[-1]["role"] == "tool":
            # No closing words in the source: end on the last call as the target
            popped = session.messages.pop()
            assert popped["role"] == "tool"
            session.count("ended_on_last_call")

    for message in session.messages:
        for call in message.get("tool_calls") or []:
            arguments = json.loads(call["function"]["arguments"])
            if arguments.pop("_waited", None) is not None:
                call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)

    for message in session.messages:
        if message["role"] == "assistant" and PROTOCOL.search(message.get("content") or ""):
            raise ConversionError("protocol_residue", message["content"])

    commands_text = session.seen
    is_git = bool(re.search(r"(^|\s)\.git(\s|$)|^(?:\S*@\S*[#$] )?git (?!init|clone)", commands_text, re.M))
    output = [{"role": "system", "content": system_prompt(harness, cwd, is_git)}] + session.messages
    return output, harness_tools(harness, cwd), session.reasoning, session.stats


def apply_to_result(result, harness=None):
    """Replace the chat of a checked terminal result; provenance records what was done."""
    harness = harness or load_harness()
    messages, tools, reasoning, stats = convert_chat(
        result["chat"]["messages"], result.get("reasoning_by_message"), harness
    )
    result["chat"] = {"messages": messages, "tools": tools}
    result["reasoning_by_message"] = reasoning
    result["tool_provenance"] = [
        {"name": t["function"]["name"], "origin": "producer_harness"} for t in tools
    ]
    result["harness_conversion"] = {
        "target": "opencode",
        "snapshot": harness["source"],
        "statistics": stats,
    }
    return result
