#!/usr/bin/env python3
"""
convert_glaive_function_calling_v2.py
──────────────────────────────────
Convert the HuggingFace *glaive-function-calling-v2* dataset
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

SRC = "glaive-function-calling-v2"

# ───────────── regexes ───────────── #
SYSTEM_RE    = re.compile(r"^SYSTEM:\s*(.*)",            re.I)
USER_RE      = re.compile(r"^USER:\s*(.*)",              re.I)
ASSIST_RE    = re.compile(r"^ASSISTANT:\s*(.*?)(?:\s*<\|endoftext\|>)?$",         re.I)
FUNC_RESP_RE = re.compile(r"^FUNCTION RESPONSE:\s*(.*)", re.I)
FUNC_CALL_RE = re.compile(r"<functioncall>\s*(.*)", re.I)

# ───────────— helpers ────────────── #
def conv_id(seed: str) -> str:
    return f"{SRC}_{hashlib.sha256(seed.encode()).hexdigest()[:12]}"

def make_part(ptype: str,
              content: str = "",
              name: str = "",
              args: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Create unified part structure compatible with other datasets."""
    return {
        "type": ptype,
        "content": content,
        "metadata": {},  # Add metadata field for compatibility
        "name": name,
        "args": json.dumps(args, ensure_ascii=False) if args else "",
        "answers": []  # Add answers field for Tulu compatibility
    }

def norm_functions(funcs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize functions with JSON string parameters for schema consistency."""
    out = []
    for f in funcs:
        if not f.get("name"):
            continue
        
        # Keep parameters as JSON string to avoid complex schema conflicts
        params = f.get("parameters", {})
        if isinstance(params, dict):
            # Ensure it has proper OpenAI structure before stringifying
            if "type" not in params:
                params = {
                    "type": "object", 
                    "properties": params if params else {},
                    "required": []
                }
            params_str = json.dumps(params, ensure_ascii=False)
        else:
            # Already a string or something else
            params_str = str(params) if params else "{}"
        
        out.append({
            "name": str(f.get("name", "")),
            "description": str(f.get("description", "")),
            "parameters": params_str  # Keep as JSON string for consistency
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
def parse_sample(system: str, chat: str) -> Optional[Dict[str, Any]]:
    system_lines = system.splitlines()
    i, n = 0, len(system_lines)

    # SYSTEM + available functions
    system_text, funcs = "", []
    if i < n and (m := SYSTEM_RE.match(system_lines[i])):
        system_text = m.group(1).strip(); i += 1
        buf = []
        while i < n and not USER_RE.match(system_lines[i]):
            buf.append(system_lines[i]); i += 1
        funcs = parse_functions("\n".join(buf))

    # Split chat by USER turns
    user_turns = []
    current_turn = ""
    
    for line in chat.split("\n\n\n"):
        line = line.strip()
        if USER_RE.match(line):
            if current_turn:
                user_turns.append(current_turn)
            current_turn = line
        else:
            if current_turn:
                current_turn += "\n\n\n" + line
    
    if current_turn:
        user_turns.append(current_turn)

    # Parse conversation
    messages: List[Dict[str, Any]] = []
    first_user = None
    
    for turn in user_turns:
        lines = turn.split("\n\n\n")
        
        # First line should be USER
        if not lines or not (user_match := USER_RE.match(lines[0].strip())):
            continue
            
        user_content = user_match.group(1).strip()
        user_msg = {"role": "user", "parts": [make_part("response", user_content)], "metadata": {}}
        
        if first_user is None:
            first_user = user_msg
        else:
            messages.append(user_msg)
        
        # Process assistant parts
        if len(lines) > 1:
            assistant_parts = []
            
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for assistant response
                if (assist_match := ASSIST_RE.match(line)):
                    body = assist_match.group(1).strip()
                    
                    if "<functioncall>" in body:
                        # Extract function call using delimiter-based regex
                        func_call_match = FUNC_CALL_RE.search(body).group(1)
                        if func_call_match:
                            try:
                                name = func_call_match.split("\"name\": \"")[1].split("\"")[0]

                                if "\'" in func_call_match:
                                    try:
                                        arguments = json.loads(func_call_match.split("\"arguments\": \'")[1].split("\'}")[0].replace("\\", ""))
                                    except:
                                        print(f"Error parsing arguments: {func_call_match}")
                                        return None
                                else:
                                    arguments = {}
                                
                                assistant_parts.append(make_part("function-call", name=name, args=arguments))
                            except json.JSONDecodeError:
                                assistant_parts.append(make_part("response", body))
                        else:
                            assistant_parts.append(make_part("response", body))
                    else:
                        # Regular assistant response
                        assistant_parts.append(make_part("response", body))
                
                # Check for function response
                elif (func_resp_match := FUNC_RESP_RE.match(line)):
                    response_content = func_resp_match.group(1).strip()
                    assistant_parts.append(make_part("function-output", content=response_content))
            
            # Add assistant message if we have parts
            if assistant_parts:
                messages.append({"role": "assistant", "parts": assistant_parts, "metadata": {}})

    # Use first user as initial if found
    if first_user is None:
        first_user = {"role": "user", "parts": [make_part("response", "")], "metadata": {}}

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
def convert_row(row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    p = parse_sample(row.get("system", ""), row.get("chat", ""))
    if p is None:
        return None
    return {
        "conversation_id": conv_id(p["initial"]["content"] + str(idx)),
        "dataset_source":  SRC,
        "original_metadata": {"original_id": idx},  # Use original_id for compatibility
        "system_prompt": {"content": p["system"], "metadata": {}},
        "initial_prompt": p["initial"],
        "available_functions": p["functions"],
        "conversation_branches": [{"messages": p["messages"]}],
        "created_timestamp": datetime.now(UTC).isoformat()
    }

def process_split(ds: Dataset, num_proc: int) -> Dataset:
    # Remove the 'sample' column after processing to avoid schema conflicts
    result = ds.map(convert_row,
                    with_indices=True,
                    num_proc=num_proc,
                    desc="Converting glaive-function-calling")

    result = result.filter(lambda x: x is not None, num_proc=num_proc)
    # Remove the original 'sample' column to clean up the schema
    if 'sample' in result.column_names:
        result = result.remove_columns(['sample'])
    return result

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
