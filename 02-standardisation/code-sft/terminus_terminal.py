"""Port checked Terminus-2 terminal trajectories to Apertus 2 as they were recorded.

This is the counterpart of opencode_terminal.py: the model sees the Terminus-2 harness, not
another one. Only the chat format changes:

  harness description   kept verbatim where the source has it (system or first user turn)
  JSON reply            kept verbatim as the assistant reply, so it still follows the
                        description; there are no tool calls and no tool schemas
  terminal output       kept verbatim as the next user turn, with harness notices and padding
  <think> block         moved to the think item, the same as reasoning_content
  JSON inside <think>   the reply is the JSON, the think item what precedes it. Some Nemotron
                        replies also lost the JSON opener with `analysis` inside the block,
                        so the reply starts at "plan", as openai_chat.py reads them
  trailing user turns   moved to `trailing_context`: a training row ends on an assistant

`openai_chat.convert` still runs first, so a row is accepted only if its replies can be read
as commands. Its `terminal_step` chat is replaced, which also drops its override prompt and
the removal of the harness description. Other repairs made for that parse do not reach the reply.
"""

import json
import re

from openai_chat import reasoning_parts, text_content
from openai_reasoning import explicit_reasoning_value

OVERRIDE_CHANGE = "Added explicit structured-call protocol override"
LOST_OPENER = re.compile(r'",\s*"plan"\s*:')


def is_final_batch(text):
    """A command batch that runs to the end of the text, or to the end of the think block."""
    try:
        value, end = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return False
    return (
        isinstance(value, dict)
        and isinstance(value.get("commands"), list)
        and text[end:].strip() in ("", "</think>")
    )


def reply_inside_think(text):
    """Split a reply whose JSON the source enclosed in think tags: (reply, thinking)."""
    for match in re.finditer(r"\{", text):
        if is_final_batch(text[match.start() :]):
            reply = text[match.start() :]
            thinking = text[: match.start()].strip().removesuffix("```json")
            break
    else:
        boundary = LOST_OPENER.search(text)
        reply = '{"plan":' + text[boundary.end() :] if boundary else ""
        if not is_final_batch(reply):
            return None, None
        thinking = text[: boundary.start()]
    return reply.strip().removesuffix("</think>").strip(), re.sub(r"^\s*<think>", "", thinking).strip()


def convert_chat(messages):
    """The source messages as a plain chat, and the thinking of each assistant turn."""
    chat, reasoning = [], []
    for source_index, message in enumerate(messages):
        role = message["role"]
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"Message {source_index}: unexpected role in a Terminus-2 chat: {role}")
        content = text_content(message.get("content"))
        if role == "assistant":
            source, (content, values) = content, reasoning_parts(content)
            if not content.strip() and source.strip():
                content, thinking = reply_inside_think(source)
                if content is None:
                    raise ValueError(f"Message {source_index}: Terminus-2 reply has no text outside thinking")
                values = [thinking] if thinking else []
            if message.get("reasoning_content"):
                values = explicit_reasoning_value(message["reasoning_content"]) + values
            if values:
                reasoning.append(
                    {
                        "message_index": len(chat),
                        "source_message_index": source_index,
                        "values": list(dict.fromkeys(values)),
                    }
                )
        chat.append({"role": role, "content": content})
    trailing = []
    while chat and chat[-1]["role"] != "assistant":
        trailing.insert(0, chat.pop())
    return chat, reasoning, trailing


def apply_to_result(result, messages):
    """Replace the chat of a checked terminal result with its source messages."""
    chat, reasoning, trailing = convert_chat(messages)
    result["chat"] = {"messages": chat, "tools": []}
    result["reasoning_by_message"] = reasoning
    result["tool_provenance"] = []
    result["trailing_observations"] = []
    result["trailing_context"] = trailing
    result["ends_with_tool_call"] = False
    result["changes"] = [c for c in result.get("changes", []) if not c.startswith(OVERRIDE_CHANGE)]
    result["harness_conversion"] = {"target": "terminus"}
    return result
