"""Recover answer-only targets from explicitly released, language-matching gold code."""

import copy
import hashlib
import re
from openai_reasoning import tag_errors

REPO = "PrimeIntellect/SYNTHETIC-1"
REVISION = "f08fe8ccdb5be7ea7d3645179aa0361a8ad0a37e"
ALIASES = {
    "python": "python",
    "python3": "python",
    "py": "python",
    "cpp": "cpp",
    "c++": "cpp",
    "c++17": "cpp",
    "javascript": "javascript",
    "js": "javascript",
    "rust": "rust",
}


def recover_reference(messages, repo, revision, raw):
    if (
        repo != REPO
        or revision != REVISION
        or raw.get("task_type") != "verifiable_code"
    ):
        return messages, []
    assistant = [
        i
        for i, m in enumerate(messages)
        if str(m.get("role", "")).lower() in {"assistant", "gpt", "bot"}
    ]
    if len(assistant) != 1:
        return messages, []
    index = assistant[0]
    if not tag_errors(messages[index].get("content")):
        return messages, []
    gold = raw.get("gold_standard_solution")
    prompt = raw.get("prompt")
    if not isinstance(gold, str) or not isinstance(prompt, str) or tag_errors(gold):
        return messages, []
    requested = re.search(
        r"using the programming language\s+([A-Za-z0-9+#]+)", prompt, re.I
    )
    block = re.fullmatch(r"\s*```([A-Za-z0-9+#]+)\s*\n(.+?)\n```\s*", gold, re.S)
    if not requested or not block:
        return messages, []
    expected = ALIASES.get(requested[1].lower())
    if not expected or ALIASES.get(block[1].lower()) != expected:
        return messages, []
    original = messages[index].get("content") or ""
    result = copy.deepcopy(messages)
    result[index]["content"] = gold
    for field in ("reasoning_content", "reasoning", "thought", "think", "analysis"):
        result[index].pop(field, None)
    return result, [
        {
            "kind": "answer_only_source_gold_reference",
            "source_message_index": index,
            "source_field": "gold_standard_solution",
            "language": expected,
            "original_response_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "reference_sha256": hashlib.sha256(gold.encode()).hexdigest(),
            "reasoning": "absent; incomplete generation retained in original source",
            "validation_scope": "Source-labelled gold reference, not re-executed. Source score/verification describe llm_response, not this replacement.",
        }
    ]
