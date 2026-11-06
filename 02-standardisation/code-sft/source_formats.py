"""Source records whose tool calls or reasoning are serialized inside message text.

Some exports flattened structured turns into text with a chat template. Left as they are,
they pass as plain chat: no declared tools, calls written as text, results as user
messages. The adapters here restore the structure from that text, before normalization,
so the usual checks and harness conversions see ordinary OpenAI-format turns.

  openhands_text_calls  LiteCoder-Terminal-SFT: OpenHands CodeActAgent trajectories whose
                        calls are "<tool_call>{json}</tool_call>" text in the assistant turn
                        and whose results are "<tool_response>...</tool_response>" user turns.
                        The export has no system prompt and no tool schemas; both come from
                        OpenHands itself (openhands_harness.json).
  harmony_terminus      gpt-oss Terminus-2 runs saved with the harmony channels in the reply:
                        "<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>
                        final<|message|>{json}". Terminus-2 parsed only the JSON object; the
                        analysis and commentary channels are the model's reasoning.

`text_format_residue` names what is still serialized in a converted chat, so such rows
are rejected instead of being kept as plain chat.
"""

import copy
import json
import re
from functools import lru_cache
from pathlib import Path

TOOL_CALL = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>")
TOOL_RESPONSE = re.compile(r"^\s*<tool_response>\n?([\s\S]*?)\n?</tool_response>\s*$")
TERMINUS_PROMPT = "Format your response as JSON"
CLAUDE_CODE_TOOLS = {"Bash", "Read", "Write", "Edit", "TodoWrite", "Glob", "Grep", "KillShell",
                     "TaskOutput", "WebSearch", "WebFetch", "MultiEdit", "NotebookEdit", "Task"}

HARMONY_TOKEN = re.compile(r"<\|(?:channel|message|start|end|return|call|constrain)\|>")
HARMONY_SEGMENT = re.compile(
    r"^<\|channel\|>(?P<channel>\w+)(?P<recipient>[^<]*)<\|message\|>(?P<text>[\s\S]*?)"
    r"(?:<\|end\|>|<\|return\|>|<\|call\|>)?$"
)


class SourceFormatError(ValueError):
    def __init__(self, reason, detail=""):
        self.reason = reason
        super().__init__(json.dumps({"status": "source_format_not_convertible", "reason": reason,
                                     "detail": str(detail)[:200]}))


@lru_cache(maxsize=1)
def openhands_harness():
    return json.loads((Path(__file__).resolve().parent / "openhands_harness.json").read_text())


def _turns(record):
    turns = record.get("conversations") or record.get("messages") or []
    out = []
    for turn in turns:
        role = turn.get("role") or turn.get("from")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        out.append((role, turn.get("content") if "content" in turn else turn.get("value") or ""))
    return out


def is_openhands_text_calls(record):
    turns = _turns(record)
    if not turns or TERMINUS_PROMPT in (turns[0][1] or ""):
        return False
    return any(role == "assistant" and TOOL_CALL.search(text or "") for role, text in turns)


TASK_STATUS = {"todo": "todo", "pending": "todo", "in_progress": "in_progress", "in-progress": "in_progress",
               "done": "done", "completed": "done", "complete": "done"}
TITLE_KEYS = ("title", "content", "description", "task")


def task_tracker_items(arguments):
    """OpenHands accepted task_tracker items in any shape; its schema allows only
    {id, title, status, notes}. The text becomes the title, the status is mapped onto
    todo / in_progress / done, and every other field is kept as text in notes.
    Returns (arguments, changed)."""
    items = arguments.get("task_list")
    if not isinstance(items, list):
        return arguments, False
    fixed, changed = [], False
    for item in items:
        if not isinstance(item, dict):
            raise SourceFormatError("task_item_not_an_object")
        lower = {str(k).lower(): k for k in item}
        title_key = next((lower[k] for k in TITLE_KEYS if k in lower), None)
        status = str(item.get(lower.get("status"), "")).lower()
        if title_key is None or "id" not in lower or status not in TASK_STATUS:
            raise SourceFormatError("task_item_not_mappable", json.dumps(item)[:120])
        notes = [str(item[lower["notes"]])] if "notes" in lower and item[lower["notes"]] else []
        notes += [f"{k}: {v}" for k, v in item.items()
                  if k not in (title_key, lower["id"], lower["status"], lower.get("notes")) and v not in (None, "")]
        new = {"id": str(item[lower["id"]]), "title": str(item[title_key]), "status": TASK_STATUS[status]}
        if notes:
            new["notes"] = "; ".join(notes)
        changed = changed or new != item
        fixed.append(new)
    return dict(arguments, task_list=fixed), changed


def openhands_text_calls(record):
    """LiteCoder OpenHands text-call record -> record with OpenAI messages and the OpenHands
    tools. Other records are returned unchanged."""
    if not is_openhands_text_calls(record):
        return record
    turns = _turns(record)
    harness = openhands_harness()
    known = {t["function"]["name"] for t in harness["tools"]}
    if any(role == "system" for role, _ in turns):
        raise SourceFormatError("unexpected_system_message")
    messages = [{"role": "system", "content": harness["system_prompt"]}]
    pending, counter, repairs = None, 0, set()
    for position, (role, text) in enumerate(turns):
        text = text or ""
        if role == "assistant":
            if pending:
                raise SourceFormatError("assistant_turn_before_result", pending)
            calls = TOOL_CALL.findall(text)
            if len(calls) > 1:
                raise SourceFormatError("several_text_calls_in_one_turn")
            reply = TOOL_CALL.sub("", text).strip()
            message = {"role": "assistant", "content": reply or None}
            if calls:
                try:
                    call = json.loads(calls[0])
                except ValueError as error:
                    raise SourceFormatError("unparsable_text_call", calls[0]) from error
                name, arguments = call.get("name"), call.get("arguments")
                if name in CLAUDE_CODE_TOOLS:
                    # Claude Code's tool schemas are not published; they cannot be declared
                    raise SourceFormatError("claude_code_text_calls", name)
                if name not in known or not isinstance(arguments, dict):
                    raise SourceFormatError("unknown_text_call", name)
                if name == "task_tracker":
                    arguments, changed = task_tracker_items(arguments)
                    if changed:
                        repairs.add("task_tracker items mapped onto the OpenHands schema "
                                    "(text -> title, status -> todo/in_progress/done, other fields -> notes)")
                pending = f"call_{counter}"
                counter += 1
                message["tool_calls"] = [{
                    "id": pending, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }]
            elif not reply:
                raise SourceFormatError("empty_assistant_turn")
            messages.append(message)
        elif role == "user":
            result = TOOL_RESPONSE.match(text)
            if result:
                if not pending:
                    raise SourceFormatError("result_without_call", text[:80])
                messages.append({"role": "tool", "tool_call_id": pending, "content": result[1]})
                pending = None
            else:
                if pending:
                    raise SourceFormatError("user_turn_before_result", text[:80])
                if "<tool_response>" in text or "<tool_call>" in text:
                    raise SourceFormatError("partial_text_call_markup", text[:80])
                # The task, OpenHands' "Please continue" and its call validation errors
                messages.append({"role": "user", "content": text})
        else:
            raise SourceFormatError("unexpected_role", role)
    record = {k: v for k, v in record.items() if k not in ("conversations", "messages")}
    record["messages"] = messages
    record["tools"] = copy.deepcopy(harness["tools"])
    # Kept in the row's metadata; the unchanged source stays in source_record_json
    record["source_format_repairs"] = ["OpenHands text calls parsed into calls and results; "
                                       "system prompt and tools from openhands_harness.json", *sorted(repairs)]
    return record


def harmony_reply(content):
    """'<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>X'
    -> (reasoning, reply X), or None when the text is not a complete harmony message."""
    if not isinstance(content, str) or not content.lstrip().startswith("<|channel|>"):
        return None
    pieces = content.strip().split("<|start|>assistant")
    segments = []
    for piece in pieces:
        match = HARMONY_SEGMENT.match(piece.strip())
        if not match or HARMONY_TOKEN.search(match["text"]):
            return None
        segments.append(match)
    *thinking, last = segments
    reasoning = "\n\n".join(s["text"].strip() for s in thinking if s["text"].strip())
    return reasoning, last["text"].strip()


def harmony_terminus(record):
    """Rewrite harmony assistant turns as '<think>reasoning</think>reply', the form of the
    other Terminus-2 runs with reasoning. Turns that do not parse are left as they are."""
    turns = record.get("messages")
    if not isinstance(turns, list) or not any(
        isinstance(t, dict) and t.get("role") == "assistant" and HARMONY_TOKEN.search(t.get("content") or "")
        for t in turns
    ):
        return record
    record = dict(record)
    record["messages"] = rewritten = []
    for turn in turns:
        turn = dict(turn)
        if turn.get("role") == "assistant":
            parsed = harmony_reply(turn.get("content"))
            if parsed is not None:
                reasoning, reply = parsed
                turn["content"] = f"<think>{reasoning}</think>{reply}" if reasoning else reply
        rewritten.append(turn)
    return record


def text_format_residue(chat):
    """What is still serialized as text in a converted chat: harmony tokens anywhere, or
    calls and results written as <tool_call> / <tool_response> text."""
    for message in chat.get("messages") or []:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if HARMONY_TOKEN.search(content):
            return "harmony_tokens_in_text"
        if message.get("role") == "assistant" and content.lstrip().startswith("<tool_call>"):
            return "tool_call_as_text"
        if message.get("role") in ("user", "tool") and content.lstrip().startswith("<tool_response>"):
            return "tool_response_as_text"
    return None
