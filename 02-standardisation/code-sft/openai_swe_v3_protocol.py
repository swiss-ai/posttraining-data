"""Narrow, source-evidenced terminal protocol repairs for pinned SWE-v3."""

import copy
import hashlib
import json
import re
from openai_source_repairs import recover_observed_nested_terminal
from openai_legacy_dispatch import REPO, REVISION

HARNESS_ERROR = "ERROR: 'ErrorObservation' object has no attribute 'metadata'"


def recover_intact_command_array(text, observation):
    # A malformed outer object can still contain a complete, valid command array.
    # Avoid losing any explicit completion flag while recovering only that array.
    if '"task_complete"' in text:
        return None
    candidates = {}
    for match in re.finditer(r'(?<!\\)"commands"\s*:\s*', text):
        try:
            commands, _ = json.JSONDecoder().raw_decode(text[match.end() :])
        except ValueError:
            continue
        if not isinstance(commands, list) or not commands:
            continue
        candidate = {"commands": commands}
        if (
            recover_observed_nested_terminal(
                json.dumps(candidate, ensure_ascii=False), observation
            )
            is not None
        ):
            candidates[json.dumps(candidate, sort_keys=True)] = candidate
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def recover_complete_object_with_closing_suffix(text, observation):
    if not observation.startswith(("New Terminal Output:", "Current Terminal Screen:")):
        return None

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key == "commands" and key in result:
                raise ValueError("Duplicate command arrays are ambiguous")
            result[key] = value
        return result

    decoder = json.JSONDecoder(object_pairs_hook=object_pairs)
    candidates = {}
    for match in re.finditer(r"\{", text):
        try:
            candidate, end = decoder.raw_decode(text[match.start() :])
        except ValueError:
            continue
        tail = text[match.start() + end :].strip()
        if (
            isinstance(candidate, dict)
            and isinstance(candidate.get("commands"), list)
            and set(candidate) <= {"analysis", "plan", "commands", "task_complete"}
            and all(
                isinstance(c, dict) and isinstance(c.get("keystrokes"), str)
                for c in candidate["commands"]
            )
            and tail
            and re.fullmatch(r"[\]}`\s]+", tail)
        ):
            keys = [
                command["keystrokes"].rstrip("\r\n")
                for command in candidate["commands"]
            ]
            waiting = bool(keys) and all(key == "" for key in keys)
            multiline = any("\n" in key or "\r" in key for key in keys)
            if observation.startswith("Current Terminal Screen:") and not waiting:
                continue
            if (
                not waiting
                and not multiline
                and recover_observed_nested_terminal(
                    json.dumps(candidate, ensure_ascii=False), observation
                )
                is None
            ):
                continue
            candidates[json.dumps(candidate, sort_keys=True)] = candidate
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def recover_confirmed_completion(text, observation):
    if (
        "Are you sure you want to mark the task as complete?" not in observation
        or 'include "task_complete": true in your JSON response again.'
        not in observation
    ):
        return None
    candidates = {}
    for match in re.finditer(r"\{", text):
        try:
            candidate, _ = json.JSONDecoder().raw_decode(text[match.start() :])
        except ValueError:
            continue
        if (
            isinstance(candidate, dict)
            and candidate.get("commands") == []
            and candidate.get("task_complete") is True
            and set(candidate) <= {"analysis", "plan", "commands", "task_complete"}
        ):
            candidates[json.dumps(candidate, sort_keys=True)] = candidate
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def repair_terminal_messages(messages, repo, revision, find_payload):
    if repo != REPO or revision != REVISION:
        return messages, []
    repaired = messages
    repairs = []
    copied = False
    for index in range(len(messages) - 1):
        action, feedback = messages[index : index + 2]
        if action.get("role") != "assistant" or not isinstance(
            action.get("content"), str
        ):
            continue
        text = action["content"]
        observation = feedback.get("content")
        if not isinstance(observation, str):
            continue
        payload, _ = find_payload(text)
        if (
            payload is None
            and '"commands"' in text
            and observation.startswith(
                ("New Terminal Output:", "Current Terminal Screen:")
            )
        ):
            recovered = recover_observed_nested_terminal(text, observation)
            method = "complete_command_object"
            if recovered is None:
                recovered = recover_intact_command_array(text, observation)
                method = "intact_command_array"
            if recovered is None:
                recovered = recover_complete_object_with_closing_suffix(
                    text, observation
                )
                method = "complete_object_with_extraneous_closing_delimiters"
            if recovered is not None:
                if not copied:
                    repaired = copy.deepcopy(messages)
                    copied = True
                # Keep the original response verbatim as context, followed by the
                # unique original command object proven by complete terminal echoes.
                repaired[index]["content"] = (
                    text
                    + "\n\n"
                    + json.dumps(recovered, ensure_ascii=False, allow_nan=False)
                )
                repairs.append(
                    {
                        "kind": (
                            "terminal_payload_with_extraneous_closing_delimiters"
                            if method
                            == "complete_object_with_extraneous_closing_delimiters"
                            else "terminal_payload_proven_by_complete_echoes"
                        ),
                        "assistant_message_index": index,
                        "observation_message_index": index + 1,
                        "original_content_sha256": hashlib.sha256(
                            text.encode()
                        ).hexdigest(),
                        "command_values_changed": False,
                        "payload_recovery": method,
                    }
                )
                payload = recovered
        if payload is None:
            confirmed = recover_confirmed_completion(text, observation)
            if confirmed is not None:
                if not copied:
                    repaired = copy.deepcopy(messages)
                    copied = True
                repaired[index]["content"] = (
                    text
                    + "\n\n"
                    + json.dumps(confirmed, ensure_ascii=False, allow_nan=False)
                )
                repairs.append(
                    {
                        "kind": "completion_payload_confirmed_by_source_harness",
                        "assistant_message_index": index,
                        "observation_message_index": index + 1,
                        "original_content_sha256": hashlib.sha256(
                            text.encode()
                        ).hexdigest(),
                        "command_values_changed": False,
                        "task_complete": True,
                    }
                )
                payload = confirmed
        if (
            feedback.get("role") == "user"
            and observation == HARNESS_ERROR
            and isinstance(payload, dict)
            and isinstance(payload.get("commands"), list)
            and all(
                isinstance(c, dict) and isinstance(c.get("keystrokes"), str)
                for c in payload["commands"]
            )
        ):
            if not copied:
                repaired = copy.deepcopy(messages)
                copied = True
            repaired[index + 1]["role"] = "tool"
            repairs.append(
                {
                    "kind": "explicit_terminal_harness_error_observation",
                    "assistant_message_index": index,
                    "observation_message_index": index + 1,
                    "original_role": "user",
                    "mapped_role": "tool",
                    "observation": observation,
                    "execution_status": "source_harness_reported_error",
                }
            )
    return repaired, repairs
