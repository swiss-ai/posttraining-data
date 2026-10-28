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


def group_unlinked_batches(messages):
    """Group a complete contiguous response block without inventing per-call IDs.

    The batch boundary is explicit in the source: one assistant call list,
    immediately followed by the same number of tool observations. Inputs and
    observations retain their own source order, with NO cross-list pairing.
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
        )
        if eligible:
            children = []
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                args = (
                    decode(function.get("arguments"))
                    if isinstance(function, dict)
                    else None
                )
                if not isinstance(function, dict) or not isinstance(args, dict):
                    eligible = False
                    break
                child = {"name": function["name"], "arguments": args}
                if call.get("id"):
                    child["source_call_id"] = call["id"]
                children.append(child)
        if eligible:
            batch_id = f"source_batch_{i}"
            assistant = copy.deepcopy(message)
            assistant["tool_calls"] = [
                {
                    "id": batch_id,
                    "type": "function",
                    "function": {
                        "name": "source_tool_batch",
                        "arguments": json.dumps(
                            {"calls": children}, ensure_ascii=False
                        ),
                    },
                }
            ]
            result.append(assistant)
            origins.append([i])
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": batch_id,
                    "content": json.dumps(
                        {
                            "association": "batch_only",
                            "source_observations": observations,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            origins.append(list(range(i + 1, end)))
            repairs.append(
                {
                    "kind": "unlinked_parallel_batch",
                    "batch_id": batch_id,
                    "assistant_source_index": i,
                    "observation_source_indices": list(range(i + 1, end)),
                    "per_call_association": "not_recorded_and_not_inferred",
                }
            )
            i = end
        else:
            result.append(copy.deepcopy(message))
            origins.append([i])
            i += 1
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
