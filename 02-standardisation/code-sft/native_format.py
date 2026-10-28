"""Map checked chat semantics to the pinned Apertus 2 reference implementation."""

import copy
from dataclasses import asdict
from functools import cache
import json

from apertus_common import (
    Conversation,
    InputMessage,
    MappingPolicy,
    OutputMessage,
    Think,
    Wait,
    checkers_for,
    from_openai,
    render_str,
)


@cache
def training_checkers():
    return checkers_for("full", training=True)


def native_conversation(result, convention="think-first"):
    messages = copy.deepcopy(result["chat"]["messages"])
    for record in result.get("reasoning_by_message", []):
        values = list(dict.fromkeys(v for v in record.get("values", []) if v))
        if values:
            messages[record["message_index"]]["reasoning_content"] = "\n\n".join(values)
    leading = 0
    for index, message in enumerate(messages):
        if message["role"] in {"system", "developer"}:
            if leading != index:
                raise ValueError(
                    "Mid-dialogue system/developer instructions cannot be mapped safely"
                )
            leading += 1
    # The source messages are retained in provenance before merging leading instructions.
    policy = MappingPolicy(
        merge_system_messages=leading > 1
        or any(m["role"] == "developer" for m in messages)
    )
    clean = []
    for message in messages:
        call = message.get("tool_calls") and not message.get("content")
        clean.append(
            {
                k: v
                for k, v in message.items()
                if v is not None and k != "name" and not (k == "content" and call)
            }
        )
    conversation = from_openai(
        clean,
        tools=result["chat"].get("tools") or None,
        system_defaults={"thinking": "medium"},
        policy=policy,
    ).value
    if convention == "think-first":
        items, previous = [], None
        for item in conversation.items:
            if (
                isinstance(item, OutputMessage)
                and (previous is None or isinstance(previous, InputMessage | Wait))
                and not isinstance(item, Think)
            ):
                items.append(Think(payload=""))
            items.append(item)
            previous = item
        conversation = Conversation(system=conversation.system, items=items)
    report = conversation.check(checkers=training_checkers())
    findings = [
        dict(
            rule=v.rule,
            severity=str(v.severity),
            message=v.message,
            location=asdict(v.location),
        )
        for v in [*report.violations, *report.unevaluated]
    ]
    if (
        not report.ok
        or not report.complete
        or any(f["rule"].startswith("training/") for f in findings)
    ):
        raise ValueError("Native validation failed: " + json.dumps(findings))
    # Check serialization independently before publishing a row.
    Conversation.from_json(conversation.to_json())
    return conversation, render_str(conversation)
