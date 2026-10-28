"""Loss-preserving OpenAI Chat Completions text/tool conversion.

Original snapshots stay separate from API-shaped chat payloads. Generated tool
schemas describe observed argument keys only, never claim to be original specs.
Unsupported or ambiguous traces are rejected instead of inventing observations.
"""

import copy
import functools
import json
from pathlib import Path
import re
from openai_source_repairs import (
    group_unlinked_batches,
    recover_observed_nested_terminal,
)

ROLES = {"system", "developer", "user", "assistant", "tool"}
NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@functools.lru_cache(maxsize=1)
def source_tool_registry():
    path = Path(__file__).with_name("openai_tool_registry.json")
    return json.loads(path.read_text()) if path.exists() else {}


@functools.lru_cache(maxsize=1)
def producer_harness_registry():
    path = Path(__file__).with_name("openai_harness_registry.json")
    return json.loads(path.read_text()) if path.exists() else {}


@functools.lru_cache(maxsize=1)
def execution_protocol_registry():
    path = Path(__file__).with_name("openai_execution_protocols.json")
    return json.loads(path.read_text()) if path.exists() else {}


def decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def text_content(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list) and all(
        isinstance(p, dict)
        and p.get("type") == "text"
        and isinstance(p.get("text"), str)
        for p in value
    ):
        return "".join(p["text"] for p in value)
    raise ValueError("Unsupported non-text message content")


def reasoning_parts(text, allow_closing_only=False):
    from openai_reasoning import extract_leading

    return extract_leading(text, allow_closing_only)


def canonical_call(name, arguments, cid):
    args = decode(arguments)
    if not isinstance(args, dict):
        raise ValueError(f"{name}: arguments must decode to an object")
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f"Invalid function name: {name!r}")
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": encode(args)},
    }


def iter_function_payloads(messages):
    """Include nested batch actions in schema checks; batching cannot hide them."""
    for i, message in enumerate(messages):
        for call in message.get("tool_calls", []):
            function = call["function"]
            yield i, function
            if function["name"] == "source_tool_batch":
                for child in decode(function["arguments"])["calls"]:
                    yield (
                        i,
                        {
                            "name": child["name"],
                            "arguments": encode(child["arguments"]),
                        },
                    )


def parse_xml_calls(text, prefix):
    """Parse the released OpenHands function/parameter text syntax, not XML entities."""
    calls = []
    pattern = r"<function=([A-Za-z0-9_-]+)>\s*(.*?)</function>"
    for j, match in enumerate(re.finditer(pattern, text, re.S)):
        body = match[2]
        args = {}
        # The producer parser accepts arbitrary nonempty parameter names and
        # both equals/attribute syntax. JSON object keys need not be identifiers.
        # Preserve parameter values exactly, including code whitespace.
        parameter_pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
        attribute_pattern = r"<parameter\b[^>]*\bname\s*=\s*[\"\']?([^\"\'>\s]+)[\"\']?[^>]*>(.*?)</parameter>"
        for param_pattern in (parameter_pattern, attribute_pattern):
            for p in re.finditer(param_pattern, body, re.S):
                key = p[1].strip()
                if not key or key in args:
                    raise ValueError("Empty or duplicate function parameter name")
                args[key] = p[2]
        residue = re.sub(parameter_pattern, "", body, flags=re.S)
        residue = re.sub(attribute_pattern, "", residue, flags=re.S)
        if residue.strip():
            raise ValueError("Unparsed text inside function call")
        calls.append(canonical_call(match[1], args, f"{prefix}_{j}"))
    remaining = re.sub(pattern, "", text, flags=re.S).strip()
    if "<function=" in remaining or "<parameter=" in remaining:
        raise ValueError("Incomplete function/parameter markup")
    return remaining, calls


def find_terminal_payload(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, end = decoder.raw_decode(text[match.start() :])
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("commands"), list):
            tail = text[match.start() + end :].strip()
            if tail not in ("", "```", "</think>"):
                continue
            return value, text[: match.start()].strip().removesuffix("```json").strip()
    return None, None


def prompt_tool_definitions(messages):
    """Recover explicitly declared typed parameters from released system prompts.

    This preserves the prose and exports only machine constraints actually
    stated by the numbered parameter declaration and its Allowed values line.
    """
    tools = []
    for message in messages:
        if message.get("role") != "system" or not isinstance(
            message.get("content"), str
        ):
            continue
        text = message["content"]
        for block in re.finditer(
            r"---- BEGIN FUNCTION #\d+: ([\w-]+) ----(.*?)---- END FUNCTION #\d+ ----",
            text,
            re.S,
        ):
            if "Parameters:" not in block[2]:
                continue
            description, parameters = block[2].rsplit("Parameters:", 1)
            props, required = {}, []
            pattern = r"\(\d+\)\s+(\w+)\s+\((string|number|integer|boolean|array|object),\s*(required|optional)\):\s*(.*?)(?=\n\s*\(\d+\)|\Z)"
            for p in re.finditer(pattern, parameters, re.S):
                prop = {"type": p[2], "description": p[4].strip()}
                allowed = re.search(r"Allowed values:\s*\[(.*?)\]", p[4], re.S)
                if allowed and p[2] == "string":
                    values = re.findall(r"`([^`]+)`", allowed[1])
                    if values:
                        prop["enum"] = values
                props[p[1]] = prop
                if p[3] == "required":
                    required.append(p[1])
            if props:
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": block[1],
                            "description": description.strip(),
                            "parameters": {
                                "type": "object",
                                "properties": props,
                                "required": required,
                                "additionalProperties": True,
                            },
                        },
                    }
                )
    return tools


def validate_chat(chat):
    """Validate our documented text/function subset plus cross-message links.

    A final assistant call is a legal supervised target, not a completed API input.
    No result is invented for that final call. Internal pending calls are errors.
    """
    errors, pending, seen = [], set(), set()
    if not isinstance(chat, dict):
        return ["Chat payload must be an object"]
    if set(chat) - {"messages", "tools"}:
        errors.append("Unexpected top-level chat field")
    messages = chat.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["messages must be a nonempty array"]
    tool_names = set()
    tools = chat.get("tools", [])
    if not isinstance(tools, list):
        errors.append("tools must be an array")
        tools = []
    for t in tools:
        if (
            not isinstance(t, dict)
            or set(t) != {"type", "function"}
            or t["type"] != "function"
        ):
            errors.append("Invalid function tool definition")
            continue
        f = t["function"]
        if (
            not isinstance(f, dict)
            or not isinstance(f.get("name"), str)
            or not NAME.fullmatch(f["name"])
        ):
            errors.append("Invalid tool definition name")
            continue
        if f["name"] in tool_names:
            errors.append("Duplicate tool definition")
        tool_names.add(f["name"])
        if (
            not isinstance(f.get("parameters"), dict)
            or f["parameters"].get("type") != "object"
        ):
            errors.append("Tool parameters must be an object schema")
        if set(f) - {"name", "description", "parameters", "strict"}:
            errors.append("Unexpected function definition field")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            errors.append(f"{i}: invalid message role")
            continue
        role = m["role"]
        allowed = {"role", "content"} | (
            {"tool_calls"}
            if role == "assistant"
            else {"tool_call_id"}
            if role == "tool"
            else set()
        )
        if set(m) - allowed:
            errors.append(f"{i}: unsupported message field")
        if not isinstance(m.get("content"), str) and not (
            role == "assistant" and m.get("content") is None and m.get("tool_calls")
        ):
            errors.append(f"{i}: invalid content")
        if role == "tool":
            cid = m.get("tool_call_id")
            if not isinstance(cid, str) or cid not in pending:
                errors.append(f"{i}: orphan or duplicate tool result")
            else:
                pending.remove(cid)
        elif pending:
            errors.append(f"{i}: unanswered calls before non-tool message")
        calls = m.get("tool_calls", [])
        if not isinstance(calls, list):
            errors.append(f"{i}: tool_calls must be an array")
            continue
        for c in calls:
            if (
                not isinstance(c, dict)
                or set(c) != {"id", "type", "function"}
                or c.get("type") != "function"
            ):
                errors.append(f"{i}: invalid tool call structure")
                continue
            cid = c.get("id")
            if not isinstance(cid, str) or not cid or cid in seen:
                errors.append(f"{i}: missing or duplicate tool call ID")
                continue
            seen.add(cid)
            pending.add(cid)
            f = c.get("function")
            if not isinstance(f, dict) or set(f) != {"name", "arguments"}:
                errors.append(f"{i}: invalid call function")
                continue
            if f.get("name") not in tool_names:
                errors.append(f"{i}: undefined tool")
            if not isinstance(f.get("arguments"), str) or not isinstance(
                decode(f["arguments"]), dict
            ):
                errors.append(f"{i}: invalid argument JSON")
    if not any(
        isinstance(m, dict) and m.get("role") == "user" and m.get("content")
        for m in messages
    ):
        errors.append("Missing user prompt")
    if messages[-1].get("role") != "assistant":
        errors.append("Training sample must end with an assistant target")
    if not any(
        isinstance(m, dict)
        and m.get("role") == "assistant"
        and (m.get("content") or m.get("tool_calls"))
        for m in messages
    ):
        errors.append("No assistant training target")
    return errors


def convert(snapshot):
    sample = snapshot["stored_sample"]
    repo = snapshot["dataset"]
    raw = decode(sample.get("source_record_json")) or {}
    messages = copy.deepcopy(snapshot.get("decoded_messages"))
    changes, warnings, issues, reasons, extras = [], [], [], [], []
    execution_provenance = []
    if raw.get("chat_rounds"):
        messages = copy.deepcopy(raw["chat_rounds"])
        changes.append("Recovered source chat_rounds")
    if not isinstance(messages, list):
        messages = [
            {"role": role, "content": sample[key]}
            for role, key in [("user", "prompt"), ("assistant", "response")]
            if sample.get(key)
        ]
    if repo == "PrimeIntellect/SYNTHETIC-1" and raw.get("llm_response"):
        if not any(m.get("role") == "assistant" for m in messages):
            messages.append({"role": "assistant", "content": raw["llm_response"]})
            changes.append("Recovered source llm_response")
    if repo == "hkust-nlp/CodeIO-PyEdu-Reasoning":
        messages = [{"role": "user", "content": raw["prompt"]}]
        for n in (1, 2):
            if raw.get(f"turn_{n}"):
                messages.append({"role": "assistant", "content": raw[f"turn_{n}"]})
                # Only intermediate feedback is conversation context; final judge
                # feedback remains metadata, not a fabricated model target.
                if n == 1 and raw.get("turn_2"):
                    if not raw.get("feedback_1"):
                        issues.append("Second CodeIO turn has no intervening feedback")
                    else:
                        messages.append({"role": "user", "content": raw["feedback_1"]})
        changes.append(
            "Recovered CodeIO answer turns; retained judge feedback in original metadata"
        )
    from openai_reference_answers import recover_reference
    from openai_reasoning import tag_errors, IMPLICIT_OPEN_REPOS

    messages, reference_replacements = recover_reference(
        messages, repo, sample.get("source_revision"), raw
    )
    reasoning_integrity = []
    for source_index, source_message in enumerate(messages):
        if not isinstance(source_message, dict):
            continue
        if str(
            source_message.get("role") or source_message.get("from") or ""
        ).lower() in {"assistant", "gpt", "bot"}:
            for error in tag_errors(
                source_message.get("content"), repo in IMPLICIT_OPEN_REPOS
            ):
                reasoning_integrity.append(
                    {"source_message_index": source_index, "issue": error}
                )
    original_messages = copy.deepcopy(messages)
    messages, message_origins, source_repairs = group_unlinked_batches(messages)
    if source_repairs:
        changes.append(
            "Preserved unlinked parallel calls/results as explicit batches; no per-call result pairing inferred"
        )
    terminal = any(
        isinstance(m.get("content"), str)
        and "keystrokes" in m["content"]
        and "Format your response as JSON" in m["content"]
        for m in messages[:2]
    )
    world_model = repo == "Lite-Coder/LiteCoder-Terminal-World-Model-SFT"
    # Follow the released role labels, not assumptions based on the repo name.
    # Its sampled gpt turns are command batches and human turns observations.
    if world_model:
        assistant_turns = [m for m in messages if m.get("role") == "assistant"]
        if assistant_turns and all(
            isinstance(decode(m.get("content")), dict)
            and isinstance(decode(m.get("content")).get("commands"), list)
            for m in assistant_turns
        ):
            terminal = True
            changes.append(
                "Preserved released World-Model role direction: assistant command batches, human terminal observations; no role inversion"
            )
        else:
            warnings.append(
                "World-model source task direction cannot be established from command/observation roles"
            )
    xml_source = repo.startswith(("Lego-X/SWE-Lego-", "TIGER-Lab/SWE-Next-"))
    shell_source = repo.startswith(("cocoa-org/", "SWE-bench/SWE-smith"))
    codeact = repo == "xingyaoww/code-act"
    feedback_source = repo == "m-a-p/Code-Feedback"
    rejected_xml_actions = set()
    if repo == "TIGER-Lab/SWE-Next-SFT-Trajectories":
        for index in range(len(messages) - 1):
            action, feedback = messages[index : index + 2]
            if (
                action.get("role") == "assistant"
                and not decode(action.get("tool_calls"))
                and str(feedback.get("content", ""))
                .lstrip()
                .startswith(
                    (
                        "NO TOOL CALL DETECTED.",
                        "You forgot to use a function call in your response.",
                    )
                )
            ):
                rejected_xml_actions.add(index)
                source_repairs.append(
                    {
                        "kind": "explicit_harness_nonexecution",
                        "assistant_source_index": message_origins[index][0],
                        "feedback_source_index": message_origins[index + 1][0],
                        "execution": "none_detected_by_source_harness",
                    }
                )
    output, pending, used = [], [], set()
    observed = {}
    protocol_changed = bool(source_repairs)
    for i, src in enumerate(messages):
        if not isinstance(src, dict):
            issues.append(f"Message {i}: not an object")
            continue
        source_indices = message_origins[i]
        extras.append(
            {
                "source_message_index": source_indices[0],
                "source_message_indices": source_indices,
                "fields": copy.deepcopy(original_messages[source_indices[0]]),
            }
        )
        role = str(src.get("role", src.get("from", ""))).lower()
        role = {
            "human": "user",
            "gpt": "assistant",
            "bot": "assistant",
            "tool_response": "tool",
        }.get(role, role)
        try:
            # OpenCode embeds explicit tool-result blocks carrying their own IDs.
            value = src.get("content", src.get("value", src.get("text")))
            if (
                role == "tool"
                and isinstance(value, list)
                and all(
                    isinstance(v, dict) and v.get("type") == "tool-result"
                    for v in value
                )
            ):
                for block in value:
                    payload = block.get("output")
                    content = (
                        payload.get("value")
                        if isinstance(payload, dict) and payload.get("type") == "text"
                        else encode(payload)
                    )
                    cid = block.get("toolCallId")
                    if cid not in [c["id"] for c in pending]:
                        raise ValueError("Embedded tool result has unmatched ID")
                    pending = [c for c in pending if c["id"] != cid]
                    output.append(
                        {"role": "tool", "tool_call_id": cid, "content": content}
                    )
                changes.append("Unwrapped OpenCode tool-result content blocks")
                continue
            content = text_content(value)
            if role == "tool_call":
                payload = decode(content)
                if not isinstance(payload, dict):
                    raise ValueError("Tool call event is not JSON")
                call = canonical_call(
                    payload.get("name"), payload.get("arguments"), f"pilot_{i}_0"
                )
                # Keep preceding assistant prose as an independent message; the
                # event has no prose of its own.
                role, content = "assistant", None
                src["tool_calls"] = [call]
                changes.append("Mapped separate tool_call/tool_response events")
                protocol_changed = True
            if role not in ROLES:
                raise ValueError(f"Unknown role {role}")
            msg = {"role": role, "content": content}
            extracted = []
            if role == "assistant":
                if pending:
                    raise ValueError("New assistant turn before all tool results")
                closing_only = repo in {
                    "microsoft/rStar-Coder",
                    "Modotte/CodeX-7M-Non-Thinking",
                }
                content, extracted = reasoning_parts(content, closing_only)
                # Sera thought often duplicates the ENTIRE assistant content,
                # not just its reasoning. Compare after the same tag parser.
                for key in (
                    "reasoning_content",
                    "reasoning",
                    "thought",
                    "think",
                    "analysis",
                ):
                    val = src.get(key)
                    if isinstance(val, str) and val.strip():
                        from openai_reasoning import explicit_reasoning_value

                        for part in explicit_reasoning_value(val):
                            if part not in extracted:
                                extracted.append(part)
                calls = decode(src.get("tool_calls")) or []
                if not isinstance(calls, list):
                    raise ValueError("tool_calls is not an array")
                normalized = []
                for j, call in enumerate(calls):
                    if not isinstance(call, dict) or not isinstance(
                        call.get("function"), dict
                    ):
                        raise ValueError("Unsupported structured call")
                    f = call["function"]
                    cid = call.get("id") or f"pilot_{i}_{j}"
                    normalized.append(
                        canonical_call(f.get("name"), f.get("arguments"), cid)
                    )
                if (
                    xml_source
                    and i not in rejected_xml_actions
                    and isinstance(content, str)
                    and "<function=" in content
                ):
                    content, parsed = parse_xml_calls(content, f"pilot_{i}")
                    normalized.extend(parsed)
                    protocol_changed = True
                    changes.append("Parsed OpenHands function/parameter calls")
                if shell_source and isinstance(content, str):
                    blocks = list(re.finditer(r"```bash\s*\n(.*?)\n```", content, re.S))
                    if len(blocks) == 1:
                        command = blocks[0][1]
                        normalized.append(
                            canonical_call(
                                "execute_bash", {"command": command}, f"pilot_{i}_0"
                            )
                        )
                        content = (
                            content[: blocks[0].start()] + content[blocks[0].end() :]
                        ).strip()
                        protocol_changed = True
                        changes.append(
                            "Mapped shell-agent command and execution output"
                        )
                    elif len(blocks) > 1:
                        raise ValueError(
                            "Multiple bash blocks: execution order not explicit"
                        )
                if codeact and isinstance(content, str):
                    blocks = list(
                        re.finditer(r"<execute>(.*?)</execute>", content, re.S)
                    )
                    if len(blocks) == 1:
                        normalized.append(
                            canonical_call(
                                "execute_python", {"code": blocks[0][1]}, f"pilot_{i}_0"
                            )
                        )
                        content = (
                            content[: blocks[0].start()] + content[blocks[0].end() :]
                        ).strip()
                        protocol_changed = True
                        changes.append("Mapped CodeAct execute/observation protocol")
                    elif len(blocks) > 1:
                        raise ValueError(
                            "Multiple execute blocks require explicit observation mapping"
                        )
                if terminal and isinstance(content, str):
                    payload, prefix = find_terminal_payload(content)
                    following = (
                        str(messages[i + 1].get("content", ""))
                        if i + 1 < len(messages)
                        else ""
                    )
                    if following.startswith(
                        "Here are the answers the other agent provided."
                    ):
                        msg["content"] = content
                        output.append(msg)
                        if extracted:
                            reasons.append(
                                {
                                    "message_index": len(output) - 1,
                                    "source_message_index": i,
                                    "values": list(dict.fromkeys(extracted)),
                                }
                            )
                        changes.append(
                            "Preserved unexecuted handoff questions/plans as assistant dialogue"
                        )
                        continue
                    if payload is None and "<tool_call>" in content:
                        blocks = list(
                            re.finditer(
                                r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.S
                            )
                        )
                        batch = []
                        complete = False
                        for position, block in enumerate(blocks):
                            tool = decode(block[1])
                            args = (
                                decode(tool.get("arguments"))
                                if isinstance(tool, dict)
                                else None
                            )
                            if (
                                isinstance(tool, dict)
                                and tool.get("name") == "mark_task_complete"
                                and args == {}
                            ):
                                if position != len(blocks) - 1:
                                    raise ValueError(
                                        "Completion marker precedes terminal commands"
                                    )
                                complete = True
                                continue
                            if (
                                not isinstance(tool, dict)
                                or tool.get("name") != "bash_command"
                                or not isinstance(args, dict)
                                or not isinstance(args.get("keystrokes"), str)
                            ):
                                raise ValueError("Unsupported terminal tool_call block")
                            batch.append(args)
                        if batch or complete:
                            payload = {"commands": batch}
                            if complete:
                                payload["task_complete"] = True
                            prefix = re.sub(
                                r"<tool_call>.*?</tool_call>", "", content, flags=re.S
                            ).strip()
                            changes.append(
                                "Mapped terminal bash_command blocks into one ordered command batch"
                            )
                    if (
                        payload is None
                        and "AUTO-CORRECTED: Fixed incomplete JSON by adding missing closing brace"
                        in following
                    ):
                        payload, prefix = find_terminal_payload(content.rstrip() + "}")
                        if payload is not None:
                            changes.append(
                                "Reproduced source harness documented repair of missing closing JSON brace"
                            )
                    if payload is None and extracted:
                        # Some released trajectories enclose the action JSON in
                        # the think block even though the harness executes it.
                        # Recover that explicit JSON, not a guessed command.
                        payload, prefix = find_terminal_payload(text_content(value))
                        if payload is not None:
                            prefix = re.sub(r"^\s*<think>\s*", "", prefix)
                            extracted = [prefix] if prefix else []
                            prefix = ""
                            warnings.append(
                                "Terminal action JSON was inside source think tags; explicit command payload recovered"
                            )
                    if payload is None and repo == "nvidia/Nemotron-Terminal-Corpus":
                        # Observed source corruption: the analysis JSON opener
                        # is missing, but the plan/commands object suffix is
                        # intact. Decode that suffix without editing commands.
                        original = text_content(value)
                        boundary = re.search(r'",\s*"plan"\s*:', original)
                        if boundary:
                            suffix = original[boundary.end() :]
                            candidate = '{"plan":' + suffix
                            try:
                                recovered, end = json.JSONDecoder().raw_decode(
                                    candidate
                                )
                                if (
                                    isinstance(recovered.get("commands"), list)
                                    and candidate[end:].strip() in ("", "</think>")
                                    and set(recovered)
                                    <= {"plan", "commands", "task_complete"}
                                ):
                                    payload, prefix = recovered, ""
                                    extracted = [
                                        re.sub(
                                            r"^\s*<think>\s*",
                                            "",
                                            original[: boundary.start()],
                                        )
                                    ]
                                    warnings.append(
                                        "Repaired missing terminal analysis JSON opener; intact plan/commands suffix decoded without command edits"
                                    )
                            except ValueError:
                                pass
                    if payload is None:
                        previous = str(messages[i - 1].get("content", "")) if i else ""
                        rejected = following.startswith(
                            "Previous response had parsing errors:"
                        )
                        handoff = (
                            "You are about to hand off your work" in previous
                            or "provide a comprehensive summary" in previous.lower()
                        )
                        final_prose = (
                            i + 1 == len(messages)
                            and isinstance(content, str)
                            and not content.lstrip().startswith("{")
                            and '"commands"' not in content
                        )
                        plain_dialogue = (
                            isinstance(content, str)
                            and not content.lstrip().startswith("{")
                            and '"commands"' not in content
                            and "<tool_call>" not in content
                        )
                        if rejected or handoff or final_prose or plain_dialogue:
                            # No command was executed for a harness rejection;
                            # preserve the failed response and feedback as chat.
                            # Handoff summaries and final prose are chat answers.
                            if rejected:
                                content = text_content(value)
                                warnings.append(
                                    "Unparseable terminal response rejected by source harness; preserved as assistant text and user feedback, not an executed action"
                                )
                            else:
                                changes.append(
                                    "Preserved terminal handoff summary/final prose as assistant content"
                                )
                            msg["content"] = content if content is not None else ""
                            if extracted:
                                reasons.append(
                                    {
                                        "message_index": len(output),
                                        "source_message_index": i,
                                        "values": list(dict.fromkeys(extracted)),
                                    }
                                )
                            output.append(msg)
                            continue
                        raise ValueError("Cannot parse terminal commands JSON")
                    args = {
                        k: v
                        for k, v in payload.items()
                        if k not in {"analysis", "plan"}
                    }
                    if not all(
                        isinstance(c, dict) and isinstance(c.get("keystrokes"), str)
                        for c in args["commands"]
                    ):
                        recovered = recover_observed_nested_terminal(
                            text_content(value), following
                        )
                        if recovered is None:
                            raise ValueError("Invalid terminal command entries")
                        payload = recovered
                        args = {
                            k: v
                            for k, v in payload.items()
                            if k not in {"analysis", "plan"}
                        }
                        changes.append(
                            "Selected unique nested command payload matching the complete recorded terminal echo sequence"
                        )
                    if payload.get("analysis"):
                        extracted.append(payload["analysis"])
                    content = "\n\n".join(v for v in (prefix, payload.get("plan")) if v)
                    # Keep an empty wait batch as an action when an observation
                    # follows; final completion is a final assistant payload.
                    if i + 1 < len(messages) or args["commands"]:
                        normalized.append(
                            canonical_call("terminal_step", args, f"pilot_{i}_0")
                        )
                    else:
                        final = encode(
                            {k: v for k, v in payload.items() if k != "analysis"}
                        )
                        content = "\n\n".join(v for v in (prefix, final) if v)
                    protocol_changed = True
                    changes.append(
                        "Mapped terminal command batches; preserved keystrokes and duration exactly"
                    )
                # CodeFeedback publishes code blocks as the executed action. Only
                # adapt when the next turn explicitly says Execution result.
                if (
                    feedback_source
                    and i + 1 < len(messages)
                    and str(messages[i + 1].get("content", "")).startswith(
                        "Execution result:"
                    )
                ):
                    blocks = re.findall(
                        r"```python\s*\n(.*?)\n```", content or "", re.S
                    )
                    if not blocks:
                        raise ValueError(
                            "Execution feedback has no preceding Python block"
                        )
                    protocol = execution_protocol_registry().get(repo)
                    documented = (
                        protocol
                        and protocol["dataset_revision"]
                        == sample.get("source_revision")
                        and protocol["selection_rule"] == "last_python_fence"
                    )
                    if not documented and len(blocks) != 1:
                        raise ValueError(
                            "Multiple Python blocks without a matching documented execution protocol"
                        )
                    normalized.append(
                        canonical_call(
                            "execute_python",
                            {"code": blocks[-1].strip()},
                            f"pilot_{i}_0",
                        )
                    )
                    execution_provenance.append(
                        {
                            "source_message_index": message_origins[i][0],
                            "selection": "producer_runner_last_python_fence"
                            if documented
                            else "single_python_block",
                            "candidate_block_count": len(blocks),
                            "protocol": protocol if documented else None,
                        }
                    )
                    changes.append(
                        "Mapped execution of the final Python cell using the published producer runtime; preserved the full assistant response"
                    )
                    protocol_changed = True
                if normalized:
                    for call in normalized:
                        if call["id"] in used:
                            raise ValueError("Duplicate source call ID")
                        used.add(call["id"])
                        f = call["function"]
                        observed.setdefault(f["name"], []).append(
                            decode(f["arguments"])
                        )
                        if f["name"] == "source_tool_batch":
                            for child in decode(f["arguments"])["calls"]:
                                observed.setdefault(child["name"], []).append(
                                    child["arguments"]
                                )
                    pending = normalized[:]
                    msg["tool_calls"] = normalized
                if isinstance(content, str) and (
                    "<think>" in content or "</think>" in content
                ):
                    if (
                        content.lstrip().startswith("<think>")
                        and "</think>" not in content
                    ):
                        # Truncated source has no answer boundary. Preserve it
                        # verbatim as content; never label a guessed final answer.
                        warnings.append(
                            "Unclosed source think block retained verbatim; truncated/reasoning-only target requires training review"
                        )
                    else:
                        warnings.append(
                            "Non-leading thinking tags retained verbatim in assistant content"
                        )
                msg["content"] = (
                    content if content is not None else (None if normalized else "")
                )
                if extracted:
                    reasons.append(
                        {
                            "message_index": len(output),
                            "source_message_index": i,
                            "values": list(dict.fromkeys(extracted)),
                        }
                    )
            else:
                if i - 1 in rejected_xml_actions and not pending:
                    role = msg["role"] = "user"
                    changes.append(
                        "Retained explicitly rejected SWE-Next response as assistant text and nonexecution feedback as user context"
                    )
                is_observation = role == "tool"
                if role == "user" and pending:
                    marker = content or ""
                    is_observation = (
                        (
                            terminal
                            and (
                                "Terminal Output:" in marker
                                or "terminal state:" in marker
                                or "Current Terminal Screen:" in marker
                                or marker.startswith(
                                    "Previous response had parsing errors:"
                                )
                            )
                        )
                        or (xml_source and marker.startswith("EXECUTION RESULT"))
                        or (shell_source and marker.startswith("<returncode>"))
                        or (codeact and marker.startswith("Observation:"))
                        or (feedback_source and marker.startswith("Execution result:"))
                    )
                if is_observation:
                    if terminal and (content or "").startswith(
                        "Previous response had parsing errors:"
                    ):
                        warnings.append(
                            "Terminal invocation was rejected by source harness; parsing error preserved as tool result, not successful execution"
                        )
                    ids = src.get("tool_call_ids") or []
                    cid = (
                        src.get("tool_call_id")
                        or src.get("id")
                        or (ids[0] if len(ids) == 1 else None)
                    )
                    if len(ids) > 1:
                        raise ValueError("Combined result references multiple call IDs")
                    if not cid:
                        label = re.match(
                            r"(?:Execution output|EXECUTION RESULT) of \[([^\]]+)\]:",
                            content or "",
                        )
                        name = src.get("name") or (label[1] if label else None)
                        matching = (
                            [c for c in pending if c["function"]["name"] == name]
                            if name
                            else []
                        )
                        if len(matching) == 1:
                            cid = matching[0]["id"]
                            changes.append(
                                "Linked tool result by its explicit tool name and unique outstanding call"
                            )
                    if not cid:
                        if len(pending) == 1:
                            cid = pending[0]["id"]
                        else:
                            raise ValueError("Ambiguous tool result association")
                    if cid not in [c["id"] for c in pending]:
                        raise ValueError("Unmatched tool result ID")
                    pending = [c for c in pending if c["id"] != cid]
                    msg = {
                        "role": "tool",
                        "tool_call_id": cid,
                        "content": content if content is not None else "",
                    }
                elif pending:
                    raise ValueError(
                        "User/system message interrupts pending tool calls"
                    )
                elif role == "user" and (
                    (xml_source and (content or "").startswith("EXECUTION RESULT"))
                    or (shell_source and (content or "").startswith("<returncode>"))
                    or (codeact and (content or "").startswith("Observation:"))
                    or (
                        feedback_source
                        and (content or "").startswith("Execution result:")
                    )
                ):
                    if codeact and "I don't understand your input" in (content or ""):
                        warnings.append(
                            "CodeAct harness rejected a non-action response; feedback retained as user message"
                        )
                    else:
                        raise ValueError(
                            "Execution observation without a parsed preceding action"
                        )
            output.append(msg)
        except (ValueError, TypeError, KeyError) as exc:
            issues.append(f"Message {i}: {exc}")
    if protocol_changed:
        changes.append(
            "Added explicit structured-call protocol override; original messages retained in sidecar"
        )
        # Terminal initial user embeds the old protocol before an explicit task
        # delimiter. Replace only that prefix, preserving task/environment text.
        if (
            terminal
            and output
            and output[0]["role"] == "user"
            and "Task Description:" in output[0]["content"]
        ):
            _, task = output[0]["content"].split("Task Description:", 1)
            output[0]["content"] = "Task Description:" + task
        override = (
            "Use the provided function tools through structured tool_calls. "
            "Tool observations arrive in tool messages. This replaces legacy instructions "
            "to emit XML function tags, execute tags, bash blocks, or terminal JSON as plain text. "
            "Preserve all task, environment, command-order, keystroke, and duration requirements."
        )
        if output and output[0]["role"] == "system":
            output[0]["content"] = (output[0]["content"] or "") + "\n\n" + override
        else:
            output.insert(0, {"role": "system", "content": override})
            for r in reasons:
                r["message_index"] += 1
    tools, tool_provenance = [], []
    original_tools = decode(snapshot.get("decoded_tools")) or []
    if not isinstance(original_tools, list):
        issues.append("Tools are not a list")
        original_tools = []
    for source in original_tools:
        t = decode(source)
        if not isinstance(t, dict):
            issues.append("Tool is not an object")
            continue
        if isinstance(t.get("function"), dict):
            f = {
                k: v
                for k, v in t["function"].items()
                if k in {"name", "description", "parameters", "strict"}
            }
        elif isinstance(t.get("name") or t.get("id"), str):
            schema = (
                t.get("parameters") or t.get("input_schema") or t.get("inputSchema")
            )
            if isinstance(schema, dict) and "jsonSchema" in schema:
                schema = schema["jsonSchema"]
            f = {
                "name": t.get("name") or t.get("id"),
                "description": t.get("description", ""),
                "parameters": schema,
            }
        else:
            issues.append("Unsupported tool definition")
            continue
        if not isinstance(f.get("parameters"), dict):
            issues.append(f"Tool {f.get('name')}: missing source parameter schema")
            continue
        tools.append({"type": "function", "function": f})
        tool_provenance.append({"name": f.get("name"), "origin": "source"})
    defined = {t["function"].get("name") for t in tools}
    for tool in prompt_tool_definitions(messages):
        name = tool["function"]["name"]
        if name not in defined:
            tools.append(tool)
            defined.add(name)
            tool_provenance.append(
                {
                    "name": name,
                    "origin": "source_prompt",
                    "constraints": "explicit types, required parameters, and allowed values",
                }
            )
            changes.append(
                f"{name}: recovered explicit parameter declarations from source system prompt"
            )
    for entry in source_tool_registry().get(repo, []):
        tool = entry["tool"]
        name = tool["function"]["name"]
        if (
            entry["source_revision"] == sample.get("source_revision")
            and name in observed
            and name not in defined
        ):
            function = {
                k: v
                for k, v in tool["function"].items()
                if k in {"name", "description", "parameters", "strict"}
            }
            tools.append({"type": "function", "function": function})
            defined.add(name)
            tool_provenance.append(
                {
                    "name": name,
                    "origin": "source_artifact",
                    "source_file": entry["source_file"],
                    "source_revision": entry["source_revision"],
                    "source_record_id": entry["record_id"],
                    "source_shard": entry["source_shard"],
                }
            )
            changes.append(
                f"{name}: recovered schema from same-repository tools.json at the exact source revision"
            )
    producer = producer_harness_registry().get(repo)
    if producer and producer["dataset_revision"] == sample.get("source_revision"):
        system_messages = [
            m.get("content") for m in original_messages if m.get("role") == "system"
        ]
        if any(
            isinstance(s, str) and s.startswith(producer["system_prefix"])
            for s in system_messages
        ):
            for tool in producer["tools"]:
                name = tool["function"]["name"]
                if name in observed and name not in defined:
                    tools.append(copy.deepcopy(tool))
                    defined.add(name)
                    tool_provenance.append(
                        {
                            "name": name,
                            "origin": "producer_harness",
                            **producer["provenance"],
                        }
                    )
                    changes.append(
                        f"{name}: recovered published producer harness schema, guarded by dataset revision and system prompt"
                    )
    adapter_schemas = {}
    if source_repairs:
        adapter_schemas["source_tool_batch"] = {
            "type": "object",
            "required": ["calls"],
            "additionalProperties": False,
            "properties": {
                "calls": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "required": ["name", "arguments"],
                        "additionalProperties": False,
                        "properties": {
                            "name": {
                                "type": "string",
                                "pattern": "^[A-Za-z0-9_-]{1,64}$",
                            },
                            "arguments": {"type": "object"},
                            "source_call_id": {"type": "string"},
                        },
                    },
                }
            },
        }
    if terminal:
        adapter_schemas["terminal_step"] = {
            "type": "object",
            "required": ["commands"],
            "properties": {
                "commands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["keystrokes"],
                        "properties": {
                            "keystrokes": {"type": "string"},
                            "duration": {"type": "number", "minimum": 0},
                        },
                        "additionalProperties": True,
                    },
                },
                "task_complete": {"type": "boolean"},
            },
            "additionalProperties": True,
        }
    if shell_source:
        adapter_schemas["execute_bash"] = {
            "type": "object",
            "required": ["command"],
            "properties": {"command": {"type": "string"}},
            "additionalProperties": False,
        }
    if codeact or feedback_source:
        adapter_schemas["execute_python"] = {
            "type": "object",
            "required": ["code"],
            "properties": {"code": {"type": "string"}},
            "additionalProperties": False,
        }
    for name, argument_sets in observed.items():
        if name in defined:
            continue
        if name in adapter_schemas:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "Dataset protocol wrapper defined by the normalization adapter; arguments preserve the released action payload.",
                        "parameters": adapter_schemas[name],
                    },
                }
            )
            tool_provenance.append(
                {
                    "name": name,
                    "origin": "adapter_defined",
                    "implementation": "openai_chat.py",
                }
            )
            continue
        keys = sorted({k for args in argument_sets for k in args})
        schema = {
            "type": "object",
            "properties": {k: {} for k in keys},
            "additionalProperties": True,
        }
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "Historical dataset action. Argument keys observed in the preserved source trajectory; original tool specification unavailable.",
                    "parameters": schema,
                },
            }
        )
        tool_provenance.append(
            {
                "name": name,
                "origin": "observed_signature_only",
                "observed_argument_keys": keys,
            }
        )
        warnings.append(
            f"{name}: original tool specification unavailable; permissive observed-signature schema exported"
        )
    # Sera includes the terminal submit observation after the last assistant
    # target. Export a training prefix through that target; retain the exact
    # trailing observations separately rather than inventing a final answer.
    trailing_observations = []
    while output and output[-1]["role"] == "tool":
        trailing_observations.insert(0, output.pop())
    if trailing_observations:
        changes.append(
            "Exported assistant-ending training prefix; trailing tool observations retained in sidecar"
        )
    trailing_context = []
    if any(m["role"] == "assistant" for m in output):
        while output and output[-1]["role"] != "assistant":
            trailing_context.insert(0, output.pop())
        if trailing_context:
            warnings.append(
                "Source ends with unanswered user/harness context; exported preceding assistant target and retained trailing context in sidecar"
            )
    # Some releases combine camelCase calls with snake_case declarations.
    # Rename only a unique spelling-equivalent source property; never guess a
    # different tool or rename conflicting arguments.
    source_definitions = {
        t["function"]["name"]: t["function"]["parameters"]
        for t in tools
        if any(
            p["name"] == t["function"]["name"]
            and p["origin"]
            in {"source", "source_prompt", "source_artifact", "producer_harness"}
            for p in tool_provenance
        )
    }

    def normalize_function(f):
        schema = source_definitions.get(f["name"], {})
        properties = schema.get("properties", {})
        args = decode(f["arguments"])
        for key in list(args):
            if key in properties:
                continue
            aliases = [
                p
                for p in properties
                if p.replace("_", "").lower() == key.replace("_", "").lower()
            ]
            if len(aliases) == 1:
                dest = aliases[0]
                if dest in args:
                    issues.append(
                        f"{f['name']}: conflicting argument aliases {key}/{dest}"
                    )
                    continue
                args[dest] = args.pop(key)
                changes.append(
                    f"{f['name']}: mapped source argument alias {key} to {dest}"
                )
        if schema:
            for key, value in list(args.items()):
                expected = properties.get(key, {}).get("type")
                enum = properties.get(key, {}).get("enum", [])
                if isinstance(value, str) and enum and value not in enum:
                    choices = [
                        v
                        for v in enum
                        if isinstance(v, str)
                        and v.casefold() == value.strip().casefold()
                    ]
                    if len(choices) == 1:
                        args[key] = choices[0]
                        changes.append(
                            f"{f['name']}.{key}: normalized serialized enum spelling to its declared value"
                        )
                if isinstance(value, str) and expected in {
                    "number",
                    "integer",
                    "boolean",
                    "array",
                    "object",
                }:
                    parsed = decode(value.strip())
                    if expected == "boolean" and value.strip().lower() in {
                        "true",
                        "false",
                    }:
                        parsed = value.strip().lower() == "true"
                    matches = (
                        (expected == "integer" and type(parsed) is int)
                        or (expected == "number" and type(parsed) in (int, float))
                        or (expected == "boolean" and type(parsed) is bool)
                        or (expected == "array" and isinstance(parsed, list))
                        or (expected == "object" and isinstance(parsed, dict))
                    )
                    if matches:
                        args[key] = parsed
                        changes.append(
                            f"{f['name']}.{key}: decoded serialized parameter using declared {expected} type"
                        )
        f["arguments"] = encode(args)

    for msg in output:
        for call in msg.get("tool_calls", []):
            f = call["function"]
            normalize_function(f)
            if f["name"] == "source_tool_batch":
                args = decode(f["arguments"])
                for child in args["calls"]:
                    nested = {
                        "name": child["name"],
                        "arguments": encode(child["arguments"]),
                    }
                    normalize_function(nested)
                    child["arguments"] = decode(nested["arguments"])
                f["arguments"] = encode(args)
    chat = {"messages": output, "tools": tools}
    errors = validate_chat(chat)
    issues.extend(errors)
    if any(len(r["values"]) > 1 for r in reasons):
        warnings.append(
            "Distinct reasoning sources retained as separate values; review their relationship before training"
        )
    for reason in reasons:
        source_indices = message_origins[reason["source_message_index"]]
        reason["source_message_index"] = source_indices[0]
        reason["source_message_indices"] = source_indices
    # A response-free source is not repaired by manufacturing an answer.
    no_target = not any(
        m["role"] == "assistant" and (m.get("content") or m.get("tool_calls"))
        for m in output
    )
    status = (
        "source_missing_target"
        if no_target and not any(m.get("role") == "assistant" for m in messages)
        else "needs_adapter"
        if issues
        else "passed_sample"
    )
    record_kind = "conversation"
    if (
        no_target
        and raw.get("type") == "function"
        and isinstance(raw.get("function"), dict)
        and sample.get("source_file") == "tools.json"
    ):
        status, record_kind = "source_artifact", "tool_definition"
        issues = [
            "Standalone tool definition; excluded from conversation training export"
        ]
    elif (
        no_target
        and Path(sample.get("source_file") or "").name == "manifest.json"
        and "prepared_at_utc" in raw
        and "splits" in raw
    ):
        status, record_kind = "source_artifact", "manifest"
        issues = ["Source dataset manifest; excluded from conversation training export"]
    result = {
        "dataset": repo,
        "status": status,
        "format_valid": not errors,
        "record_kind": record_kind,
        "reasoning_integrity": reasoning_integrity,
        "reference_replacements": reference_replacements,
        "issues": list(dict.fromkeys(issues)),
        "warnings": list(dict.fromkeys(warnings)),
        "changes": list(dict.fromkeys(changes)),
        "chat": chat,
        "reasoning_by_message": reasons,
        "unassigned_reasoning": sample.get("reasoning"),
        "source_message_extras": extras,
        "tool_provenance": tool_provenance,
        "source_repairs": source_repairs,
        "execution_provenance": execution_provenance,
        "trailing_observations": trailing_observations,
        "trailing_context": trailing_context,
        "source_snapshot": str(snapshot.get("_path")),
        "sample_id": sample.get("id"),
        "source_shard": snapshot["source_shard"],
        "row_offset_in_shard": snapshot["row_offset_in_shard"],
        "ends_with_tool_call": bool(output and output[-1].get("tool_calls")),
        "training_review_required": bool(warnings),
    }

    from openai_reasoning import result_reasoning_errors

    normalized_reasoning_errors = result_reasoning_errors(result)
    if reasoning_integrity or normalized_reasoning_errors:
        result["issues"].extend(
            "Source reasoning: " + entry["issue"] for entry in reasoning_integrity
        )
        result["issues"].extend(normalized_reasoning_errors)
        result["issues"] = list(dict.fromkeys(result["issues"]))
        result["status"] = "source_incomplete_reasoning"
        result["format_valid"] = False
    if reference_replacements:
        result["changes"].append(
            "Replaced incomplete generation with a language-matching source gold reference; no reasoning fabricated"
        )

    from openai_legacy_dispatch import repair_missing_tool_interface

    result = repair_missing_tool_interface(result, sample.get("source_revision"))
    post_errors = validate_chat(result["chat"]) if result.get("legacy_dispatch") else []
    if post_errors:
        result["issues"].extend(post_errors)
        result["format_valid"] = False
        if result["status"] == "passed_sample":
            result["status"] = "needs_adapter"
    if result["status"] == "needs_adapter" and not snapshot.get(
        "_swe_v3_terminal_retry"
    ):
        from openai_swe_v3_protocol import repair_terminal_messages

        original_source_messages = snapshot.get("decoded_messages")
        if isinstance(original_source_messages, list):
            repaired_messages, protocol_repairs = repair_terminal_messages(
                original_source_messages,
                repo,
                sample.get("source_revision"),
                find_terminal_payload,
            )
            if protocol_repairs:
                retry_snapshot = dict(
                    snapshot,
                    decoded_messages=repaired_messages,
                    _swe_v3_terminal_retry=True,
                )
                repaired_result = convert(retry_snapshot)
                for proof in protocol_repairs:
                    proof["assistant_source_index"] = proof.pop(
                        "assistant_message_index"
                    )
                    proof["observation_source_index"] = proof.pop(
                        "observation_message_index"
                    )
                repaired_result["source_repairs"].extend(protocol_repairs)
                repaired_result["initial_conversion_issues"] = result["issues"]
                return repaired_result
    return result
