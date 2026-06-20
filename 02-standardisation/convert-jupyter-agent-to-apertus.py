#!/usr/bin/env python3
"""
Convert jupyter-agent/jupyter-agent-dataset into Apertus structured chat format.

The source dataset is OpenAI-style:
  - messages[0] is the user prompt
  - assistant messages contain natural-language content plus tool_calls
  - tool messages contain code execution outputs

This converter stores code execution as standard pre-linearized function parts:
`function-call` for code execution requests and `function-output` for tool
results. `linearise-dataset.py` turns those parts into final `tool_calls` and
`tool_outputs` blocks. After the first user prompt, synthetic user turns are
empty.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, DatasetDict, load_dataset


SRC = "jupyter-agent/jupyter-agent-dataset"
DEFAULT_SPLIT = "non_thinking"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def conv_id(seed: str) -> str:
    return f"{SRC}_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def dumps_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_jsonish(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def content_to_text(content: Any) -> str:
    content = parse_jsonish(content, content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {None, "text"} and item.get("text") is not None:
                    parts.append(str(item["text"]))
                elif item.get("content") is not None:
                    parts.append(content_to_text(item["content"]))
                else:
                    parts.append(dumps_json(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "output"):
            if content.get(key) is not None:
                return content_to_text(content[key])
    return dumps_json(content)


def make_part(
    part_type: str,
    content: str = "",
    name: str = "",
    args: Any = "",
) -> Dict[str, Any]:
    return {
        "type": part_type,
        "content": content or "",
        "metadata": {},
        "name": name or "",
        "args": dumps_json(args) if not isinstance(args, str) else args,
        "answers": [],
    }


def make_user_parts(text: str) -> List[Dict[str, Any]]:
    return [make_part("response", text)]


def normalize_tool(tool: Any) -> Optional[Dict[str, str]]:
    tool = parse_jsonish(tool, tool)
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = fn.get("name")
    if not name:
        return None
    params = fn.get("parameters", {})
    return {
        "name": str(name),
        "description": str(fn.get("description") or ""),
        "parameters": dumps_json(params) if params else "{}",
    }


def normalize_tools(tools: Any) -> List[Dict[str, str]]:
    tools = parse_jsonish(tools, [])
    if not isinstance(tools, list):
        return []
    out: List[Dict[str, str]] = []
    seen = set()
    for tool in tools:
        normalized = normalize_tool(tool)
        if not normalized:
            continue
        key = normalized["name"]
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def normalize_tool_call(call: Any) -> Optional[Dict[str, str]]:
    call = parse_jsonish(call, call)
    if not isinstance(call, dict):
        return None

    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = fn.get("name")
    if not name:
        return None

    arguments = fn.get("arguments", {})
    return {"name": str(name), "arguments": dumps_json(arguments)}


def tool_call_parts(tool_calls: Any) -> List[Dict[str, Any]]:
    tool_calls = parse_jsonish(tool_calls, [])
    if not isinstance(tool_calls, list):
        return []
    calls = [call for call in (normalize_tool_call(item) for item in tool_calls) if call]
    return [
        make_part("function-call", name=call["name"], args=call["arguments"])
        for call in calls
    ]


def tool_output_parts(tool_messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for msg in tool_messages:
        output = content_to_text(msg.get("content"))
        if output:
            outputs.append(make_part("function-output", content=output))
    return outputs


def assistant_message_from_source(
    msg: Dict[str, Any], following_tool_messages: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []

    text = content_to_text(msg.get("content")).strip()
    if text:
        parts.append(make_part("response", text))

    parts.extend(tool_call_parts(msg.get("tool_calls")))
    parts.extend(tool_output_parts(following_tool_messages))

    if not parts:
        return None
    return {"role": "assistant", "parts": parts}


def build_branch_messages(messages: List[Dict[str, Any]]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    first_user: Optional[Dict[str, Any]] = None
    branch_messages: List[Dict[str, Any]] = []
    have_branch_message = False
    i = 0

    while i < len(messages):
        msg = messages[i] or {}
        role = msg.get("role")

        if role == "system":
            i += 1
            continue

        if role == "user":
            text = content_to_text(msg.get("content"))
            if first_user is None:
                first_user = {
                    "role": "user",
                    "content": text,
                    "metadata": {},
                }
            else:
                branch_messages.append({"role": "user", "parts": make_user_parts("")})
                have_branch_message = True
            i += 1
            continue

        if role == "assistant":
            tool_messages: List[Dict[str, Any]] = []
            j = i + 1
            while j < len(messages) and (messages[j] or {}).get("role") == "tool":
                tool_messages.append(messages[j])
                j += 1

            assistant_msg = assistant_message_from_source(msg, tool_messages)
            if assistant_msg:
                if have_branch_message and (
                    not branch_messages or branch_messages[-1].get("role") != "user"
                ):
                    branch_messages.append({"role": "user", "parts": make_user_parts("")})
                branch_messages.append(assistant_msg)
                have_branch_message = True
            i = j
            continue

        if role == "tool":
            outputs = tool_output_parts([msg])
            if outputs:
                if have_branch_message and (
                    not branch_messages or branch_messages[-1].get("role") != "user"
                ):
                    branch_messages.append({"role": "user", "parts": make_user_parts("")})
                branch_messages.append({"role": "assistant", "parts": outputs})
                have_branch_message = True
            i += 1
            continue

        i += 1

    if first_user is None:
        first_user = {"role": "user", "content": "", "metadata": {}}
    return first_user, branch_messages


def extract_system(messages: List[Dict[str, Any]]) -> str:
    system_parts = [
        content_to_text(msg.get("content")).strip()
        for msg in messages
        if isinstance(msg, dict) and msg.get("role") == "system"
    ]
    return "\n\n".join(part for part in system_parts if part)


def convert_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    messages = parse_jsonish(row.get("messages"), [])
    if not isinstance(messages, list):
        messages = []

    initial_prompt, branch_messages = build_branch_messages(messages)
    row_id = str(row.get("id") or idx)
    seed = f"{row_id}\n{initial_prompt.get('content', '')}"

    original_metadata = {
        "source_id": row.get("id"),
        "edu_score": row.get("edu_score"),
        "files_used": row.get("files_used"),
        "packages_used": row.get("packages_used"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "kaggle_dataset_name": row.get("kaggle_dataset_name"),
        "executor_type": row.get("executor_type"),
        "original_notebook": row.get("original_notebook"),
    }

    return {
        "conversation_id": conv_id(seed),
        "dataset_source": SRC,
        "original_metadata": original_metadata,
        "created_timestamp": now_iso(),
        "system_prompt": {"content": extract_system(messages), "metadata": {}},
        "initial_prompt": initial_prompt,
        "available_functions": normalize_tools(row.get("tools")),
        "conversation_branches": [{"messages": branch_messages}],
    }


def convert_batch(batch: Dict[str, List[Any]], indices: List[int]) -> Dict[str, List[Any]]:
    keys = list(batch.keys())
    rows = [dict(zip(keys, values)) for values in zip(*(batch[key] for key in keys))]
    converted = [convert_row(row, idx) for row, idx in zip(rows, indices)]
    out: Dict[str, List[Any]] = {key: [] for key in converted[0].keys()}
    for row in converted:
        for key, value in row.items():
            out[key].append(value)
    return out


def executor_matches(row: Dict[str, Any], allowed_executor_types: set[str]) -> bool:
    executor_type = str(row.get("executor_type") or "").strip().lower()
    return executor_type in allowed_executor_types


def normalize_split(split: str) -> str:
    if split == "non-thinking":
        return "non_thinking"
    return split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert jupyter-agent/jupyter-agent-dataset to Apertus structured chat format."
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory for save_to_disk.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="HF split to load. Default: non_thinking.")
    parser.add_argument("--num-proc", type=int, default=1, help="Parallel map workers.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batched map size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--executor-types", nargs="+", default=["e2b"],
                        help="Executor types to keep. Default: e2b.")
    args = parser.parse_args()

    split = normalize_split(args.split)
    allowed_executor_types = {item.strip().lower() for item in args.executor_types if item.strip()}
    if not allowed_executor_types:
        raise ValueError("--executor-types must contain at least one non-empty value")
    print(f"Keeping executor_type in {sorted(allowed_executor_types)}")

    if args.limit is not None:
        print(f"Loading {SRC} split={split!r} in streaming mode for limit={args.limit}")
        rows = []
        seen = 0
        for source_idx, row in enumerate(load_dataset(SRC, split=split, streaming=True)):
            if seen >= args.limit:
                break
            if not executor_matches(row, allowed_executor_types):
                continue
            rows.append(convert_row(row, source_idx))
            seen += 1
        converted = Dataset.from_list(rows)
        print(f"Converted {len(converted)} rows")
    else:
        print(f"Loading {SRC} split={split!r}")
        data = load_dataset(SRC, split=split)
        before = len(data)
        data = data.filter(
            lambda row: executor_matches(row, allowed_executor_types),
            num_proc=args.num_proc,
            desc="Filtering executor_type",
        )
        print(f"Filtered executor_type: kept {len(data)}/{before} rows")
        print(f"Converting {len(data)} rows with num_proc={args.num_proc}")
        converted = data.map(
            convert_batch,
            batched=True,
            batch_size=args.batch_size,
            with_indices=True,
            remove_columns=data.column_names,
            num_proc=args.num_proc,
            desc="Converting Jupyter-agent rows",
        )

    dataset = DatasetDict({"train": converted})
    print(f"Saving to {args.output}")
    dataset.save_to_disk(args.output)
    print("Done")


if __name__ == "__main__":
    main()
