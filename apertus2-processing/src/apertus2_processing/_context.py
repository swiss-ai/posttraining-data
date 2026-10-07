"""Decode explicit caller-supplied history evidence."""

import json
from dataclasses import fields

from apertus_common import (
    CallState,
    CheckContext,
    Conversation,
    HistoryCheckpoint,
    HostClosure,
    SystemPrompt,
)


def object_fields(value, cls):
    if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
        raise ValueError(f"invalid {cls.__name__} fields")
    return dict(value)


def array_value(value, name):
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array or null")
    return value


def decode_context(raw):
    if raw is None:
        return None
    value = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    value = object_fields(value, CheckContext)
    if value.get("prefix") is not None:
        value["prefix"] = Conversation.from_json(json.dumps(value["prefix"]))
    if value.get("checkpoint") is not None:
        checkpoint = object_fields(value["checkpoint"], HistoryCheckpoint)
        if checkpoint.get("system") is not None:
            checkpoint["system"] = SystemPrompt.from_json(json.dumps(checkpoint["system"]))
        if checkpoint.get("calls") is not None:
            checkpoint["calls"] = tuple(
                CallState(**object_fields(c, CallState))
                for c in array_value(checkpoint["calls"], "checkpoint.calls")
            )
        if checkpoint.get("document_ids") is not None:
            checkpoint["document_ids"] = array_value(
                checkpoint["document_ids"], "checkpoint.document_ids"
            )
        value["checkpoint"] = HistoryCheckpoint(**checkpoint)
    if value.get("host_closures") is not None:
        value["host_closures"] = tuple(
            HostClosure(**object_fields(c, HostClosure))
            for c in array_value(value["host_closures"], "host_closures")
        )
    return CheckContext(**value)
