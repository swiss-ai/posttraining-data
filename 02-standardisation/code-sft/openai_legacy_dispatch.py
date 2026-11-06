"""Explicit adapter interface for a pinned release that omits native tool APIs.

This defines a NEW transport contract; it does not reconstruct or validate the
unreleased native tool contracts. Source names/arguments/results remain unchanged.
"""

import copy
import json

REPO = "nvidia/Nemotron-SFT-SWE-v3"
REVISION = "3f73de64c1fe928a8f538fe45ccc10c228cc4c6a"
DISPATCH_NAME = "source_tool_call"


def repair_missing_tool_interface(result, revision):
    if result.get("dataset") != REPO or revision != REVISION or not result.get("chat"):
        return result
    unknown = {
        p["name"]
        for p in result.get("tool_provenance", [])
        if p["origin"] == "observed_signature_only"
    }
    if not unknown:
        return result
    chat = result["chat"]
    if any(t["function"]["name"] == DISPATCH_NAME for t in chat["tools"]):
        raise ValueError("Native/adapter dispatch-name collision")
    originals = []
    wrapped = 0
    for index, message in enumerate(chat["messages"]):
        for call in message.get("tool_calls", []):
            function = call["function"]
            originals.append(
                {
                    "message_index": index,
                    "call_id": call["id"],
                    "function": copy.deepcopy(function),
                }
            )
            if function["name"] == "source_tool_batch":
                payload = json.loads(function["arguments"])
                for child in payload["calls"]:
                    if child["name"] in unknown:
                        native = {
                            "name": child["name"],
                            "arguments": copy.deepcopy(child["arguments"]),
                        }
                        child.update(name=DISPATCH_NAME, arguments=native)
                        wrapped += 1
                function["arguments"] = json.dumps(
                    payload, ensure_ascii=False, allow_nan=False
                )
            elif function["name"] in unknown:
                native = {
                    "name": function["name"],
                    "arguments": json.loads(function["arguments"]),
                }
                function.update(
                    name=DISPATCH_NAME,
                    arguments=json.dumps(native, ensure_ascii=False, allow_nan=False),
                )
                wrapped += 1
    if not wrapped:
        return result
    chat["tools"] = [t for t in chat["tools"] if t["function"]["name"] not in unknown]
    chat["tools"].append(
        {
            "type": "function",
            "function": {
                "name": DISPATCH_NAME,
                "description": (
                    "Dispatch an original recorded tool invocation by its native name and exact arguments. "
                    "The source release omitted native tool schemas. This adapter does not prevalidate native arguments. "
                    "Return the native observation verbatim. Names available in this record: "
                    + ", ".join(sorted(unknown))
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": sorted(unknown)},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                },
            },
        }
    )
    native_provenance = [p for p in result["tool_provenance"] if p["name"] in unknown]
    result["tool_provenance"] = [
        p for p in result["tool_provenance"] if p["name"] not in unknown
    ]
    result["tool_provenance"].append(
        {
            "name": DISPATCH_NAME,
            "origin": "adapter_defined",
            "native_contract_status": "not_released_and_not_reconstructed",
            "contract": "Dispatch(name, arguments) to the original native executor; return the recorded/native observation unchanged.",
        }
    )
    result["legacy_dispatch"] = {
        "policy": "explicit-native-dispatch-v1",
        "source_revision": revision,
        "wrapped_calls": wrapped,
        "native_names": sorted(unknown),
        "native_tool_provenance": native_provenance,
        "original_exported_calls": originals,
        "limits": [
            "Native API requirements cannot be checked because the release omitted them.",
            "Live execution requires source_tool_call and source_tool_batch adapter support.",
        ],
    }
    # Install instructions before context without changing existing system text;
    # adjust reasoning indexes to preserve exact message alignment.
    chat["messages"].insert(
        0,
        {
            "role": "system",
            "content": "For tool actions whose original API schema was omitted, use source_tool_call with the original native tool name "
            "and exact argument object. The adapter forwards the invocation without inferring native parameter requirements. "
            "Tool results retain the original observation. source_tool_batch preserves a whole parallel block without "
            "claiming an individual observation-to-call association.",
        },
    )
    for reason in result.get("reasoning_by_message", []):
        reason["message_index"] += 1
    for original in originals:
        original["message_index"] += 1
    result["changes"].append(
        "Mapped unreleased native tool APIs to an explicit lossless dispatch contract; original requirements remain unknown"
    )
    return result


def dispatch_native(arguments, executor):
    """Runtime hook. The caller supplies its own executor; never invoked by audits."""
    if (
        set(arguments) != {"name", "arguments"}
        or not isinstance(arguments["name"], str)
        or not isinstance(arguments["arguments"], dict)
    ):
        raise ValueError("Expected exactly name:string and arguments:object")
    return executor(arguments["name"], copy.deepcopy(arguments["arguments"]))
