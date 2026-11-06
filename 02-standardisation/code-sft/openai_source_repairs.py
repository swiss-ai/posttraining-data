"""Evidence-preserving repairs for source exports that omit linkage information."""

import copy
import json
import re


def decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def explicit_result_id(message):
    if message.get("tool_call_id") or message.get("id") or message.get("tool_call_ids"):
        return True
    value = message.get("content")
    return isinstance(value, list) and any(
        isinstance(v, dict) and v.get("toolCallId") for v in value
    )


VALIDATION_FAILURE = re.compile(
    r'Validation failed for tool \\?"([^"\\]+)\\?"[\s\S]*?Received arguments:\s*([\s\S]*)'
)


def names_another_call(call, observation):
    """True when an observation identifies its call, and that call is not `call`.

    Some harnesses answer an invalid call with "Validation failed for tool "X" ...
    Received arguments: {...}" and return such errors ahead of the other results
    (UltraData-SFT-Agent-2609 General-Agent), so source order is wrong there.
    """
    content = decode(observation.get("content"))
    if isinstance(content, dict):  # {"result": "..."} wrappers
        content = "\n".join(v for v in content.values() if isinstance(v, str))
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    match = VALIDATION_FAILURE.search(text or "")
    if not match:
        return False
    function = call["function"]
    if match[1] != function.get("name"):
        return True
    received = decode(match[2].strip())
    arguments = decode(function.get("arguments"))
    return isinstance(received, dict) and isinstance(arguments, dict) and received != arguments


def link_parallel_results_by_position(messages):
    """Link the results of a parallel call block to their calls by position.

    Some exports drop the call IDs: an assistant message with N calls is followed by
    exactly N tool observations, none of them naming its call. Agent harnesses
    (OpenCode, the OpenAI and Anthropic tool APIs) return parallel results in call
    order, so observation k answers call k. Each call keeps its source ID or gets
    "source_call_<message>_<k>", and observation k gets that ID as tool_call_id.
    The calls then stay ordinary calls of the harness's own tools.

    Blocks whose observation count differs from the call count, or where an
    observation names a different call than the one at its position, are left
    unchanged; the conversion rejects them as ambiguous instead of guessing.
    """
    result, origins, repairs = [], [], []
    i = 0
    while i < len(messages):
        message = messages[i]
        calls = decode(message.get("tool_calls")) if isinstance(message, dict) else None
        end = i + 1
        while (
            end < len(messages)
            and isinstance(messages[end], dict)
            and messages[end].get("role") in {"tool", "tool_response"}
        ):
            end += 1
        observations = messages[i + 1 : end]
        eligible = (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(calls, list)
            and len(calls) > 1
            and len(observations) == len(calls)
            and not any(explicit_result_id(m) for m in observations)
            and all(
                isinstance(c, dict) and isinstance(c.get("function"), dict)
                for c in calls
            )
            and not any(
                names_another_call(c, o) for c, o in zip(calls, observations)
            )
        )
        if not eligible:
            result.append(copy.deepcopy(message))
            origins.append([i])
            i += 1
            continue
        ids = [c.get("id") or f"source_call_{i}_{k}" for k, c in enumerate(calls)]
        if len(set(ids)) != len(ids):
            ids = [f"source_call_{i}_{k}" for k in range(len(calls))]
        assistant = copy.deepcopy(message)
        assistant["tool_calls"] = [
            {**copy.deepcopy(c), "id": call_id} for c, call_id in zip(calls, ids)
        ]
        result.append(assistant)
        origins.append([i])
        for k, (observation, call_id) in enumerate(zip(observations, ids)):
            linked = copy.deepcopy(observation)
            linked["tool_call_id"] = call_id
            result.append(linked)
            origins.append([i + 1 + k])
        repairs.append(
            {
                "kind": "parallel_results_linked_by_position",
                "assistant_source_index": i,
                "observation_source_indices": list(range(i + 1, end)),
                "call_ids": ids,
                "per_call_association": "source_order",
            }
        )
        i = end
    return result, origins, repairs


def recover_observed_nested_terminal(text, observation):
    """Recover a valid nested command payload only when terminal echoes prove it.

    No flattening or deduplication of commands is permitted. The unique selected
    payload must match the complete echoed single-line command sequence exactly.
    """
    if "New Terminal Output:" not in observation:
        return None
    echoes = [
        m[1] for m in re.finditer(r"(?m)^[^\n]*@[^\n]*[#$] (.*)$", observation) if m[1]
    ]
    if not echoes:
        return None
    candidates = {}
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except ValueError:
            continue
        if not isinstance(payload, dict) or not isinstance(
            payload.get("commands"), list
        ):
            continue
        commands = payload["commands"]
        if not commands or not all(
            isinstance(c, dict) and isinstance(c.get("keystrokes"), str)
            for c in commands
        ):
            continue
        keys = [c["keystrokes"].rstrip("\r\n") for c in commands]
        if any("\n" in k or "\r" in k for k in keys):
            continue
        if keys == echoes:
            candidates[json.dumps(payload, sort_keys=True)] = payload
    return next(iter(candidates.values())) if len(candidates) == 1 else None
