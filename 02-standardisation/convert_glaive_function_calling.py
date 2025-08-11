#!/usr/bin/env python3
"""
convert_glaive_function_calling.py
──────────────────────────────────
Convert the HuggingFace *glaive-function-calling* dataset (field `sample`)
into the unified chat-tool schema we use for XLAM.

• `available_functions`  → [{name, description, parameters (JSON-string)}]
• Messages carry `parts`  → {type, content, name, args}

Turn mapping
------------
USER line             → role="user"          part.type="response"
ASSISTANT line        → role="assistant"     part.type="response"
<functioncall>{...}   → role="tool-call"     part.type="function-call"
FUNCTION RESPONSE:{…} → role="tool-response" part.type="function-output"
"""

import re, json, sys, argparse, hashlib
from pathlib import Path
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional

from datasets import Dataset, DatasetDict, load_from_disk

SRC = "glaive-function-calling"

# ───────────── regexes ───────────── #
SYSTEM_RE    = re.compile(r"^SYSTEM:\s*(.*)",            re.I)
USER_RE      = re.compile(r"^USER:\s*(.*)",              re.I)
ASSIST_RE    = re.compile(r"^ASSISTANT:\s*(.*)",         re.I)
FUNC_RESP_RE = re.compile(r"^FUNCTION RESPONSE:\s*(.*)", re.I)
FUNC_CALL_RE = re.compile(r"^<functioncall>\s*(\{.*\})", re.I)

# ───────────— helpers ────────────── #
def conv_id(seed: str) -> str:
    return f"{SRC}_{hashlib.sha256(seed.encode()).hexdigest()[:12]}"

def make_part(ptype: str,
              content: str = "",
              name: str = "",
              args: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    return {
        "type": ptype,
        "content": content,
        "name": name,
        "args": json.dumps(args, ensure_ascii=False) if args else ""
    }

def norm_functions(funcs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Flatten & stringify parameters so every element has the same schema."""
    out = []
    for f in funcs:
        if not f.get("name"):
            continue
        out.append({
            "name":        str(f.get("name", "")),
            "description": str(f.get("description", "")),
            "parameters":  json.dumps(f.get("parameters", {}), ensure_ascii=False)
        })
    return out

def parse_functions(block: str) -> List[Dict[str, Any]]:
    """
    Extract only *top-level* JSON objects that have "name" from `block`.
    Avoids capturing nested braces inside the parameters schema.
    """
    objs, depth, start = [], 0, None
    in_str = esc = False

    for i, ch in enumerate(block):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidate = block[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and obj.get("name"):
                        objs.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return objs

# ───────────— sample parser ───────── #
def parse_sample(text: str) -> Dict[str, Any]:
    lines = text.splitlines()
    i, n = 0, len(lines)

    # SYSTEM + available functions
    system_text, funcs = "", []
    if i < n and (m := SYSTEM_RE.match(lines[i])):
        system_text = m.group(1).strip(); i += 1
        buf = []
        while i < n and not USER_RE.match(lines[i]):
            buf.append(lines[i]); i += 1
        funcs = parse_functions("\n".join(buf))

    # conversation
    messages: List[Dict[str, Any]] = []
    while i < n:
        ln = lines[i]

        if (m := USER_RE.match(ln)):
            messages.append({"role": "user",
                             "parts": [make_part("response", m.group(1).strip())]})

        elif (m := ASSIST_RE.match(ln)):
            body = m.group(1).strip()
            if call := FUNC_CALL_RE.match(body):
                try:
                    j = json.loads(call.group(1))
                    messages.append({"role": "FUNCTION-CALL",
                                     "parts": [make_part("function-call",
                                                         name=j.get("name", ""),
                                                         args=j.get("arguments", {}))]})
                except json.JSONDecodeError:
                    messages.append({"role": "assistant",
                                     "parts": [make_part("response", body)]})
            else:
                messages.append({"role": "assistant",
                                 "parts": [make_part("response", body)]})

        elif (m := FUNC_RESP_RE.match(ln)):
            messages.append({"role": "FUNCTION-RESPONSE",
                             "parts": [make_part("function-output",
                                                 content=m.group(1).strip())]})

        else:  # continuation
            if messages:
                last = messages[-1]["parts"][0]
                last["content"] += ("\n" if last["content"] else "") + ln.strip()

        i += 1

    # lift first user
    first_user = next((m for m in messages if m["role"] == "user"),
                      {"role": "user", "parts": [make_part("response", "")]})
    messages.remove(first_user)

    return {
        "system": system_text,
        "functions": norm_functions(funcs),
        "initial": {
            "role": "user",
            "content": first_user["parts"][0]["content"],
            "metadata": {}
        },
        "messages": messages,
    }

# ───────────— map row ─────────────── #
def convert_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    p = parse_sample(row.get("sample", ""))
    return {
        "conversation_id": conv_id(p["initial"]["content"] + str(idx)),
        "dataset_source":  SRC,
        "original_metadata": {"row_index": idx},
        "system_prompt": {"content": p["system"], "metadata": {}},
        "initial_prompt": p["initial"],
        "available_functions": p["functions"],
        "conversation_branches": [{"messages": p["messages"]}],
        "created_timestamp": datetime.now(UTC).isoformat()
    }

def process_split(ds: Dataset, num_proc: int) -> Dataset:
    return ds.map(convert_row,
                  with_indices=True,
                  num_proc=num_proc,
                  desc="Converting glaive-function-calling")

def subset(ds: Dataset, lim: Optional[int]):
    return ds if not lim or lim <= 0 or lim >= ds.num_rows else ds.select(range(lim))

def load_existing_metadata(input_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    meta_file = Path(input_path) / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None

def save_dataset_and_metadata(dataset_dict: DatasetDict, output_path: Path, 
                             input_path: Path, args: argparse.Namespace):
    """Save converted dataset with processing metadata."""
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    dataset_dict.save_to_disk(str(output_path))
    
    # Load existing metadata or create new
    metadata = load_existing_metadata(input_path) or {}
    
    # Create processing entry
    processing_entry = {
        "operation": "convert_glaive_function_calling",
        "script": "convert_glaive_function_calling.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "description": "Converted Glaive function calling dataset to standardized chat format with parts structure"
    }
    
    # Add to processing log
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Add format metadata if not already present
    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = "glaive-function-calling"
    if "conversion_details" not in metadata:
        metadata["conversion_details"] = {
            "function_format": "parts_based_function_calling",
            "parameter_schema": "json_string",
            "conversation_type": "multi_turn_function_calling",
            "tool_extraction": "regex_based_fence_parsing"
        }
    
    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")

# ───────────— CLI / main ───────────── #
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("input_path")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()

def main():
    a = cli()
    inp = Path(a.input_path)
    out = Path(a.output if not a.output.endswith("/")
               else a.output + inp.name + "-converted")

    if out.exists() and input(f"{out} exists. overwrite? [y/N]: ").lower() != "y":
        sys.exit(0)

    ds = load_from_disk(str(inp))
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    out_ds = DatasetDict()
    for split, d in ds.items():
        print(f"{split}: {d.num_rows:,} rows")
        d = subset(d, a.limit)
        out_ds[split] = process_split(d, a.num_proc)

    save_dataset_and_metadata(out_ds, out, inp, a)

if __name__ == "__main__":
    main()
