"""Thinking delimiter integrity, excluding Markdown code literals."""

import re
import json

TOKEN = re.compile(r"(?<!\\)<\s*(/?)\s*think\s*>", re.I)
FENCE = re.compile(r"(?m)^[ \t]{0,3}(`{3,}|~{3,})[^\n]*(?:\n|$)")
PROTOCOL = re.compile(
    r"<function=[^>]+>.*?</function>|<execute>.*?</execute>|<tool_call>.*?</tool_call>",
    re.S,
)
INLINE = re.compile(r"(`+)([^\n]*?)\1")
IMPLICIT_OPEN_REPOS = {"microsoft/rStar-Coder", "Modotte/CodeX-7M-Non-Thinking"}


def visible_tokens(text):
    if not isinstance(text, str) or not TOKEN.search(text):
        return []
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
        spans.append((opened[0], len(text)))
    spans.extend((m.start(), m.end()) for m in PROTOCOL.finditer(text))
    decoder = json.JSONDecoder()
    for start in re.finditer(r"[\{\[]", text):
        if any(a <= start.start() < b for a, b in spans):
            continue
        try:
            value, end = decoder.raw_decode(text, start.start())
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            spans.append((start.start(), end))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    spans = merged
    cursor = 0
    literal_spans = list(spans)
    for start, end in spans + [(len(text), len(text))]:
        for match in INLINE.finditer(text, cursor, start):
            literal_spans.append((match.start(), match.end()))
        cursor = end
    literal_spans.sort()
    result = []
    span_index = 0
    for match in TOKEN.finditer(text):
        while (
            span_index < len(literal_spans)
            and literal_spans[span_index][1] <= match.start()
        ):
            span_index += 1
        if (
            span_index < len(literal_spans)
            and literal_spans[span_index][0]
            <= match.start()
            < literal_spans[span_index][1]
        ):
            continue
        result.append((match.start(), match.end(), bool(match[1])))
    return result


def tag_errors(text, allow_implicit_open=False):
    tokens = visible_tokens(text)
    if allow_implicit_open and len(tokens) == 1 and tokens[0][2]:
        return []
    issues = []
    opened = None
    for start, end, closing in tokens:
        if closing:
            if opened is None:
                issues.append(f"orphan closing think tag at character {start}")
            else:
                opened = None
        else:
            if opened is not None:
                issues.append(f"nested think tag at character {start}")
            opened = start
    if opened is not None:
        issues.append(f"unclosed think tag at character {opened}")
    return issues


def extract_leading(text, allow_implicit_open=False):
    if not isinstance(text, str):
        return text, []
    tokens = visible_tokens(text)
    if allow_implicit_open and len(tokens) == 1 and tokens[0][2]:
        start, end, _ = tokens[0]
        return text[end:].lstrip(), [text[:start].strip()]
    if tag_errors(text):
        return text, []
    cursor = 0
    values = []
    index = 0
    while index + 1 < len(tokens):
        opening, closing = tokens[index : index + 2]
        if opening[2] or not closing[2] or text[cursor : opening[0]].strip():
            break
        values.append(text[opening[1] : closing[0]].strip())
        cursor = closing[1]
        index += 2
    return (text[cursor:].lstrip(), values) if values else (text, [])


def explicit_reasoning_value(text):
    """A dedicated reasoning field already declares scope; unwrap its delimiter."""
    answer, values = extract_leading(text, allow_implicit_open=True)
    if values:
        return values
    if isinstance(text, str):
        tokens = visible_tokens(text)
        if len(tokens) == 1 and not tokens[0][2] and not text[: tokens[0][0]].strip():
            return [text[tokens[0][1] :].strip()]
        return [text.strip()]
    return []


def result_reasoning_errors(result):
    errors = []
    # Generated wrappers have an exact, independently recorded boundary. Markdown
    # code examples inside a dedicated reasoning field can contain unbalanced
    # fences; those fences must not hide the closing delimiter we inserted.
    # Check the complete prefix against the sidecar, then validate each original
    # field separately. Raw source messages still use the ordinary token parser.
    rendered = {}
    if result.get("payload_reasoning_mode") == "inline":
        for record in result.get("reasoning_by_message", []):
            rendered.setdefault(record["message_index"], []).extend(
                record.get("values", [])
            )
    for index, message in enumerate((result.get("chat") or {}).get("messages", [])):
        if message.get("role") == "assistant":
            content = message.get("content")
            values = list(dict.fromkeys(v for v in rendered.get(index, []) if v))
            if values:
                prefix = "<think>\n" + "\n\n".join(values) + "\n</think>\n\n"
                if not isinstance(content, str) or not content.startswith(prefix):
                    errors.append(
                        f"message {index}: rendered reasoning differs from recorded boundary/content"
                    )
                    continue
                content = content[len(prefix) :]
            errors.extend(f"message {index}: {error}" for error in tag_errors(content))
    for record in result.get("reasoning_by_message", []):
        for value in record.get("values", []):
            errors.extend(
                f"reasoning for message {record['message_index']}: {error}"
                for error in tag_errors(value)
            )
    return errors
