"""Independent, cached validation of payloads and declared native/batched calls."""

import json
from functools import lru_cache
from pathlib import Path
import jsonschema
from openai_chat import iter_function_payloads, validate_chat

DECLARED = {
    "source",
    "source_prompt",
    "source_artifact",
    "producer_harness",
    "adapter_defined",
}


@lru_cache(maxsize=512)
def tool_validator(serialized):
    schema = json.loads(serialized)
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


@lru_cache(maxsize=1)
def chat_validator():
    return jsonschema.Draft202012Validator(
        json.loads(Path(__file__).with_name("openai_chat.schema.json").read_text())
    )


def check_result(result):
    errors = []
    chat = result.get("chat")
    if not chat:
        return ["Missing chat payload"]
    from openai_reasoning import result_reasoning_errors

    errors.extend(result_reasoning_errors(result))
    errors.extend(validate_chat(chat))
    errors.extend(e.message for e in chat_validator().iter_errors(chat))
    definitions = {
        t["function"]["name"]: t["function"]["parameters"]
        for t in chat.get("tools", [])
    }
    origins = {t["name"]: t["origin"] for t in result.get("tool_provenance", [])}
    for index, function in iter_function_payloads(chat["messages"]):
        name = function["name"]
        if origins.get(name) not in DECLARED:
            continue
        try:
            validator = tool_validator(json.dumps(definitions[name], sort_keys=True))
            for error in validator.iter_errors(json.loads(function["arguments"])):
                errors.append(f"{index}: {name}: {error.message}")
        except Exception as exc:
            errors.append(f"{index}: {name}: invalid source schema/arguments: {exc}")
    return errors
