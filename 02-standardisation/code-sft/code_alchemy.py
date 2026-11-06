"""Recover released Code Alchemy turn boundaries before substituting source code."""

import re

from openai_reasoning import FENCE, visible_tokens

REPO = "open-alchemy/code-alchemy"
REVISION = "d367da91def5024929d0fa8d46d47d4ef616b467"
PLACEHOLDER = "{{{REPLACE_WITH_BLOB_ID_SOURCE}}}"
CHAT_SUBSETS = {"code-qa", "code-dev", "code-dialogue", "code-trace"}
EVAL_SUBSETS = {"dev-eval", "trace-eval"}
ROLE = re.compile(r"(?m)^(User|Assistant):[ \t]*(?:\r?\n)?")


class CompletionOnly(ValueError):
    """A released completion has no explicit chat prompt/answer boundary."""


def protected_spans(text):
    spans = []
    opened = None
    for match in FENCE.finditer(text):
        fence = match[1]
        if opened is None:
            opened = (match.start(), fence)
        elif (
            fence[0] == opened[1][0]
            and len(fence) >= len(opened[1])
            and not match[0].lstrip()[len(fence) :].strip()
        ):
            spans.append((opened[0], match.end()))
            opened = None
    if opened is not None:
        if any(ROLE.finditer(text, opened[0])):
            raise ValueError("Ambiguous role markers inside an unclosed code fence")
        spans.append((opened[0], len(text)))
    depth = 0
    start = None
    for a, b, closing in visible_tokens(text):
        if not closing:
            if depth == 0:
                start = a
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                spans.append((start, b))
    if depth:
        spans.append((start, len(text)))
    return spans


def chat_messages(text, educational=False):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty Code Alchemy text")
    spans = protected_spans(text)

    def visible(offset):
        return not any(a <= offset < b for a, b in spans)

    if educational:
        problem = re.match(r"\A\s*## PROBLEM[ \t]*\r?\n", text)
        answers = [
            m
            for m in re.finditer(r"(?m)^## ANSWER[ \t]*\r?\n", text)
            if visible(m.start())
        ]
        if problem is None or len(answers) != 1:
            raise ValueError("Ambiguous educational QA headings")
        answer = answers[0]
        messages = [
            dict(
                role="user", content=text[problem.end() : answer.start()].strip("\r\n")
            ),
            dict(role="assistant", content=text[answer.end() :].strip("\r\n")),
        ]
    else:
        markers = [m for m in ROLE.finditer(text) if visible(m.start())]
        if not markers or text[: markers[0].start()].strip():
            raise ValueError("Missing explicit Code Alchemy chat boundary")
        messages = []
        for index, match in enumerate(markers):
            role = "user" if match[1] == "User" else "assistant"
            if role != ("user" if index % 2 == 0 else "assistant"):
                raise ValueError("Ambiguous/nonalternating Code Alchemy roles")
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            messages.append(
                dict(role=role, content=text[match.end() : end].strip("\r\n"))
            )
        if messages[-1]["role"] != "assistant":
            raise ValueError("Code Alchemy chat has no final assistant target")
    if any(not m["content"].strip() for m in messages):
        raise ValueError("Empty Code Alchemy turn")
    return messages


def adapt(row, subset, revision):
    if revision != REVISION:
        raise ValueError("Code Alchemy adapter requires its inspected source revision")
    if subset in EVAL_SUBSETS:
        raise ValueError("Code Alchemy evaluation subsets cannot enter SFT")
    if subset == "code-enhance" or (
        subset == "code-qa" and row.get("style") == "multipl-e"
    ):
        raise CompletionOnly(
            "Code-only completion: retain in the pretraining collection"
        )
    if subset not in CHAT_SUBSETS:
        raise ValueError("Unknown Code Alchemy subset")
    text = row.get("text_with_placeholders") or row.get("text")
    # Parse while placeholders still protect raw source from being mistaken for role headers.
    messages = chat_messages(
        text, educational=row.get("style") == "code_contests_educational"
    )
    if PLACEHOLDER in text:
        source = row.get("source_code")
        if (
            row.get("source_code_status") != "hydrated"
            or not isinstance(source, str)
            or not source.strip()
        ):
            raise ValueError(
                "Code Alchemy source context is unavailable or excluded by the source-materialization policy"
            )
        messages = [
            dict(m, content=m["content"].replace(PLACEHOLDER, source.strip()))
            for m in messages
        ]
    if any(PLACEHOLDER in m["content"] for m in messages):
        raise ValueError("Unresolved Code Alchemy source placeholder")
    return dict(row, messages=messages)
