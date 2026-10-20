# Linearisation Format Reference

## Input (Parts-Based Schema)

```json
{
  "conversation_id": "str",
  "dataset_source": "str",
  "original_metadata": {},
  "created_timestamp": "ISO_datetime",
  "system_prompt": { "content": "str" },
  "initial_prompt": { "content": "str" },
  "available_functions": [
    { "name": "str", "description": "str", "parameters": "JSON str or dict" }
  ],
  "conversation_branches": [{
    "messages": [{
      "role": "assistant|user",
      "parts": [
        { "type": "thought",              "content": "str" },
        { "type": "function-call",         "name": "str", "args": "str" },
        { "type": "function-output",       "content": "str" },
        { "type": "response",             "content": "str" },
        { "type": "verifiable-responses",  "answers": ["str"] }
      ]
    }]
  }]
}
```

**Part ordering within an assistant message:**
```
thought? -> (function-call, function-output)* -> response?
```
Call/output pairs must be interleaved 1:1. Violations drop remaining parts.

## Output (Linearised Format)

```json
{
  "conversation_id": "str",
  "dataset_source": "str",
  "original_metadata": {},
  "created_timestamp": "ISO_datetime",
  "messages": [
    { "role": "system",    "content": { "text": "system prompt" } },
    { "role": "developer", "content": {
        "tools": "JSON str of tool defs",
        "has_thinking": true,
        "formatted_tools": "TypeScript-type rendering"
    }},
    { "role": "user",      "content": { "parts": [{ "type": "text", "text": "str" }] } },
    { "role": "assistant", "content": { "blocks": [
        { "type": "thoughts",     "text": "str",  "calls": [], "outputs": [] },
        { "type": "tool_calls",   "text": "",     "calls": [{ "name": "str", "arguments": "str" }], "outputs": [] },
        { "type": "tool_outputs", "text": "",     "calls": [], "outputs": [{ "name": "", "output": "str" }] },
        { "type": "response",     "text": "str",  "calls": [], "outputs": [] }
    ]}}
  ]
}
```

The `developer` message is always present. `has_thinking` is `true` if any `thought` part exists. `verifiable-responses` parts become `display_answers` tool call + output blocks.

Stored as HF `DatasetDict` via `save_to_disk()` (arrow files).
