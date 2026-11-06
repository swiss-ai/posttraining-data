"""Render normalized reasoning inside the common training chat schema."""

from openai_reasoning import visible_tokens, result_reasoning_errors


def reasoning_text(value):
    tokens = visible_tokens(value)
    if not tokens:
        return value
    chunks = []
    start = 0
    for opening, end, _ in tokens:
        chunks.append(value[start:opening])
        start = end
    chunks.append(value[start:])
    return "".join(chunks).strip()


def render_reasoning(result, mode="inline"):
    if mode not in {"inline", "sidecar"}:
        raise ValueError("Unknown reasoning mode")
    if result.get("payload_reasoning_mode"):
        if result["payload_reasoning_mode"] != mode:
            raise ValueError("Payload was already rendered with another mode")
        return result
    if mode == "inline":
        by_message = {}
        for record in result.get("reasoning_by_message", []):
            values = []
            for value in record.get("values", []):
                clean = reasoning_text(value)
                if clean and clean not in values:
                    values.append(clean)
            record["values"] = values
            by_message.setdefault(record["message_index"], []).extend(values)
        for index, values in by_message.items():
            if not values:
                continue
            message = result["chat"]["messages"][index]
            if message["role"] != "assistant":
                raise ValueError("Reasoning is not linked to an assistant")
            reason = "\n\n".join(dict.fromkeys(values))
            message["content"] = (
                "<think>\n" + reason + "\n</think>\n\n" + (message.get("content") or "")
            )
    result["payload_reasoning_mode"] = mode
    errors = result_reasoning_errors(result)
    if errors:
        raise ValueError("Invalid reasoning after rendering: " + str(errors))
    return result
