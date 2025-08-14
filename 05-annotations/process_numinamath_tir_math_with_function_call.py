#!/usr/bin/env python3
"""
process_numinamath_tir_math_with_function_call.py — Extract ```python ...``` and ```output ...``` blocks
from NuminaMath TIR dataset and convert to function call format.

ONLY two tool shapes are recognized:
- ```python ...```  -> {type:"function-call", name:"execute_python", args:{"code": "..."}}
- ```output ...```  -> {type:"function-output", content:"..."}

All other text (including other fenced languages) is kept as
{type:"response", content:"..."} in-order.

User/System messages -> single response part (string; list joined).
Keeps/backfills system_prompt; preserves existing non-empty initial_prompt
(else lifts earliest user and removes that one instance).
Adds available_functions to the conversation structure.

All parts include unified schema fields for Arrow compatibility:
- type, content, metadata, name, args (empty strings when not used)
"""

import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from datasets import load_from_disk, Dataset, DatasetDict

# Available functions definition
AVAILABLE_FUNCTIONS = [
    {
        "name": "execute_python",
        "description": "Executes Python code in a sandboxed environment and returns the output or any raised exceptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to be executed. Supports multi-line code blocks, standard library imports, and variable persistence within the execution context."
                }
            },
            "required": ["code"]
        }
    }
]

# Robust fenced block matcher:
# - allows indentation before ```
# - supports \n or \r\n
# - language token captured in group 1
# - body captured non-greedily in group 2
FENCE = re.compile(
    r"""(?s)```([a-zA-Z0-9_-]+)\s*\n(.*?)```""",
    re.IGNORECASE,
)

PREF_SPLITS = ["train", "validation", "test"]

# ------------------------------- helpers ------------------------------------ #
def _ensure_metadata(d: Dict[str, Any]) -> Dict[str, Any]:
    if "metadata" not in d or d["metadata"] is None:
        d["metadata"] = {}
    return d

def _strip(s: Any) -> str:
    return s.strip() if isinstance(s, str) else ""

def contains_bash_code_blocks(text: str) -> bool:
    """Check if text contains bash code blocks that should be filtered out."""
    return "```bash" in text

def should_filter_sample(sample: Dict[str, Any]) -> bool:
    """Check if a sample should be filtered out due to bash code blocks."""
    conversation_branches = sample.get("conversation_branches", [])
    
    for branch in conversation_branches:
        messages = branch.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and contains_bash_code_blocks(content):
                return True
    return False

def _create_unified_part(part_type: str, 
                         content: str = "", 
                         name: str = "", 
                         args: Any = None,
                         metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Create a part with unified schema for Arrow compatibility."""
    # Convert args to proper format with encoding preservation
    if args is None:
        args_str = ""
    elif isinstance(args, dict):
        # Use ensure_ascii=False to preserve Unicode characters
        args_str = json.dumps(args, ensure_ascii=False) if args else ""
    else:
        args_str = str(args) if args else ""
    
    return {
        "type": part_type,
        "content": content or "",
        "metadata": metadata or {},
        "name": name or "",
        "args": args_str
    }

def _validate_part(part: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and ensure proper structure for a part with unified schema."""
    if not isinstance(part, dict):
        return _create_unified_part("response", content=str(part))
    
    part_type = part.get("type", "").lower()
    
    if part_type == "function-call":
        # Extract args - could be dict or string
        args = part.get("args") or part.get("parameters") or {}
        return _create_unified_part(
            "function-call",
            name=part.get("name", ""),
            args=args,
            metadata=part.get("metadata")
        )
    elif part_type == "function-output":
        return _create_unified_part(
            "function-output",
            content=part.get("content", ""),
            metadata=part.get("metadata")
        )
    else:
        return _create_unified_part(
            "response",
            content=part.get("content", ""),
            metadata=part.get("metadata")
        )

def pick_split(ds):
    if isinstance(ds, DatasetDict):
        for s in PREF_SPLITS:
            if s in ds:
                return s
        return next(iter(ds.keys()))
    return "train"

# --------------------------- parts construction ----------------------------- #
def assistant_parts_from_string(text: Optional[str], debug: bool = False) -> List[Dict[str, Any]]:
    """
    Parse assistant free-text for ```python ...``` and ```output ...``` fences.
    Emit parts in order. Only python fences that are followed by output fences are treated as tools.
    Unknown fences (plaintext, markdown, etc.) are kept as visible response (fenced block included).
    """
    s = text or ""
    parts: List[Dict[str, Any]] = []
    pos = 0
    
    if debug:
        print(f"DEBUG: Processing text of length {len(s)}")
        print(f"DEBUG: Text preview: {repr(s[:200])}")

    # Find all fence matches
    fence_matches = list(FENCE.finditer(s))
    
    for i, m in enumerate(fence_matches):
        start = m.start(0)
        end = m.end(0)
        lang = (m.group(1) or "").lower()
        body = m.group(2)  # Don't strip - preserve original content

        # preamble text before fence
        pre = _strip(s[pos:start])
        if pre:
            parts.append(_create_unified_part("response", content=pre))

        if lang == "python":
            # Check if this python block is followed by an output block
            has_output_after = False
            if i + 1 < len(fence_matches):
                next_match = fence_matches[i + 1]
                next_lang = (next_match.group(1) or "").lower()
                if next_lang == "output":
                    has_output_after = True
            
            if has_output_after:
                # Create function call with full code block including markers
                full_code_block = s[start:end]
                parts.append(_create_unified_part(
                    "function-call",
                    name="execute_python",
                    args={"code": full_code_block}
                ))
            else:
                # Keep as visible response since no output follows
                fenced = s[start:end]
                parts.append(_create_unified_part("response", content=fenced.strip()))
                
        elif lang == "output":
            parts.append(_create_unified_part(
                "function-output",
                content=body
            ))
        else:
            # keep unknown fenced block as visible response (plaintext, markdown, etc.)
            fenced = s[start:end]
            parts.append(_create_unified_part("response", content=fenced.strip()))

        pos = end

    # trailing text after last fence
    tail = _strip(s[pos:])
    if tail:
        parts.append(_create_unified_part("response", content=tail))

    # Safety: if nothing matched but there was text, emit a single response
    if not parts and _strip(s):
        parts.append(_create_unified_part("response", content=s.strip()))

    return parts

def assistant_parts_from_blocks(blocks: Any) -> List[Dict[str, Any]]:
    """
    If the assistant message is an array (legacy), parse any string-y segments
    for fences; passthrough known parts if already in `parts` shape.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(blocks, list):
        return out

    for b in blocks:
        if isinstance(b, str):
            out.extend(assistant_parts_from_string(b))
            continue
        if not isinstance(b, dict):
            continue

        # Already in parts shape?
        t = (b.get("type") or "").lower().replace("_", "-")
        if t in ("response", "thought"):
            out.append(_create_unified_part(t, content=b.get("content")))
        elif t == "function-call":
            args = b.get("args") or b.get("parameters") or {}
            out.append(_create_unified_part(
                "function-call",
                name=b.get("name"),
                args=args
            ))
        elif t == "function-output":
            out.append(_create_unified_part(
                "function-output",
                content=b.get("content")
            ))
        else:
            # Legacy role text/thought?
            role = (b.get("role") or "").lower()
            if role in ("text", "assistant_text", "thought"):
                typ = "response" if role != "thought" else "thought"
                c = b.get("content")
                if isinstance(c, str):
                    if typ == "response":
                        out.extend(assistant_parts_from_string(c))
                    else:
                        out.append(_create_unified_part("thought", content=c.strip()))
                else:
                    out.append(_create_unified_part(typ, content=_strip(c)))
            else:
                # Unknown dict — salvage text if present
                c = b.get("content")
                if isinstance(c, str) and c.strip():
                    out.extend(assistant_parts_from_string(c))
    return out

def user_or_system_parts_from_any(content: Any) -> List[Dict[str, Any]]:
    """User/System → single response part (join list to string if needed)."""
    if isinstance(content, list):
        frags = []
        for x in content:
            if isinstance(x, dict):
                v = x.get("content")
                frags.append(v if isinstance(v, str) else ("" if v is None else str(v)))
            elif isinstance(x, str):
                frags.append(x)
        text = "\n".join(f for f in frags if f)
    else:
        text = content if isinstance(content, str) else ""
    return [_create_unified_part("response", content=text.strip())]

def normalize_message_to_parts(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Return {role, parts, metadata} with tool extraction applied."""
    role = msg.get("role")
    md = msg.get("metadata") or {}
    out = {"role": role, "metadata": md}

    if role == "assistant":
        # Check if message already has parts
        existing_parts = msg.get("parts")
        if existing_parts and isinstance(existing_parts, list):
            # Process existing parts to extract fences from text content
            new_parts = []
            for part in existing_parts:
                if isinstance(part, dict) and part.get("type") == "response":
                    content = part.get("content", "")
                    if isinstance(content, str):
                        # Extract fences from the text content
                        extracted_parts = assistant_parts_from_string(content)
                        new_parts.extend(extracted_parts)
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            parts = new_parts
        else:
            # Fall back to content-based processing
            content = msg.get("content")
            if isinstance(content, list):
                parts = assistant_parts_from_blocks(content)
            else:
                parts = assistant_parts_from_string(content)
        
        if not parts:
            s = _strip(msg.get("content") or "")
            if s:
                parts = [_create_unified_part("response", content=s)]
        out["parts"] = parts
    else:
        out["parts"] = user_or_system_parts_from_any(msg.get("content"))

    # nuke legacy 'content'
    out.pop("content", None)
    return out

def normalize_branch(branch: Dict[str, Any]) -> Dict[str, Any]:
    msgs_in = branch.get("messages") or []
    msgs_out = [normalize_message_to_parts(m) for m in msgs_in]
    return {"messages": msgs_out, **{k: v for k, v in branch.items() if k != "messages"}}

# --------------------- initial/system prompt handling ----------------------- #
def keep_or_lift_initial_prompt(current_initial: Optional[Dict[str, Any]],
                                branches: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Keep if non-empty
    if isinstance(current_initial, dict):
        c = current_initial.get("content")
        if isinstance(c, str) and c.strip():
            return {
                "role": current_initial.get("role", "user"),
                "content": c,
                "metadata": current_initial.get("metadata") or {}
            }, branches

    # Else lift earliest user and remove it
    new_branches: List[Dict[str, Any]] = []
    captured: Optional[Dict[str, Any]] = None
    removed = False
    for br in branches or []:
        msgs = (br or {}).get("messages") or []
        nm = []
        for m in msgs:
            if not removed and m.get("role") == "user":
                # Build from response parts
                texts = [p.get("content") for p in (m.get("parts") or []) if isinstance(p, dict) and p.get("type") == "response" and isinstance(p.get("content"), str)]
                captured = {"role": "user", "content": "\n".join(t for t in texts if t).strip(), "metadata": m.get("metadata") or {}}
                removed = True
                continue
            nm.append(m)
        new_branches.append({"messages": nm, **{k: v for k, v in br.items() if k != "messages"}})

    if not captured:
        captured = {"role": "user", "content": "", "metadata": {}}

    return captured, new_branches

# ---------------------------- map operations -------------------------------- #
def process_batch(batch: Dict[str, List]) -> Dict[str, List]:
    import copy
    out = copy.deepcopy(batch)
    n = len(out["conversation_id"])

    if "system_prompt" not in out:
        out["system_prompt"] = [None] * n
    if "initial_prompt" not in out:
        out["initial_prompt"] = [None] * n

    for i in range(n):
        # Normalize branches
        in_branches = (out.get("conversation_branches") or [None]*n)[i] or []
        norm_branches = [normalize_branch(br) for br in in_branches]

        # Filter out samples containing bash code blocks
        sample_data = {k: v[i] for k, v in out.items() if isinstance(v, list) and i < len(v)}
        if should_filter_sample(sample_data):
            # Mark this sample for filtering by setting empty content
            out["system_prompt"][i] = {"content": "", "metadata": {}}
            out["initial_prompt"][i] = {"role": "user", "content": "", "metadata": {}}
            out["conversation_branches"][i] = []
            continue

        # System prompt
        sp = (out.get("system_prompt") or [None]*n)[i]
        spc = sp.get("content") if isinstance(sp, dict) else None
        orig_meta = (out.get("original_metadata") or [None]*n)[i] or {}
        ctk = (orig_meta.get("chat_template_kwargs") or {})
        fallback = ctk.get("custom_instructions") or ""
        out["system_prompt"][i] = {"content": (spc if isinstance(spc, str) and spc.strip() else fallback), "metadata": {}}

        # Initial prompt
        existing_init = (out.get("initial_prompt") or [None]*n)[i]
        new_init, new_branches = keep_or_lift_initial_prompt(existing_init, norm_branches)
        out["initial_prompt"][i] = new_init
        out["conversation_branches"][i] = new_branches

    # Add available_functions as a list with the same value for each row
    out["available_functions"] = [AVAILABLE_FUNCTIONS] * n

    # Filter out samples that were marked for removal (empty content)
    valid_indices = []
    for i in range(n):
        # Check if this sample has any meaningful content
        has_content = False
        if out["system_prompt"][i].get("content", "").strip():
            has_content = True
        if out["initial_prompt"][i].get("content", "").strip():
            has_content = True
        for branch in out["conversation_branches"][i]:
            for msg in branch.get("messages", []):
                for part in msg.get("parts", []):
                    if part.get("content", "").strip():
                        has_content = True
                        break
                if has_content:
                    break
            if has_content:
                break
        
        if has_content:
            valid_indices.append(i)
    
    # Filter all fields to keep only valid samples
    if len(valid_indices) < n:
        for key in out:
            if isinstance(out[key], list):
                out[key] = [out[key][i] for i in valid_indices]

    return out

def map_split(ds: Dataset, num_proc: int, batch_size: int) -> Dataset:
    return ds.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Extract ```python``` and ```output``` into execute_python tool parts; add available_functions; normalize to {role, parts, metadata}"
    )

# ------------------------------- I/O & main --------------------------------- #
def subset(ds: Dataset, n: Optional[int]) -> Dataset:
    return ds if n is None or n <= 0 or n >= ds.num_rows else ds.select(range(n))

def load_existing_metadata(input_path: Path) -> Optional[Dict[str, Any]]:
    meta_file = Path(input_path) / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None

def save_with_meta(out_ds: DatasetDict, out_path: Path,
                   in_path: Path, args: argparse.Namespace):
    out_path.mkdir(parents=True, exist_ok=True)
    out_ds.save_to_disk(str(out_path))

    metadata = load_existing_metadata(in_path) or {}
    metadata.setdefault("processing_log", []).append({
        "operation": "extract_execute_python_and_output_fences_with_available_functions",
        "script": "process_numinamath_tir_math_with_function_call.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(in_path),
        "output_path": str(out_path),
        "batch_size": args.batch_size,
        "num_processes": args.num_proc,
        "limit": args.limit,
    })
    with open(out_path / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved dataset + metadata to {out_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Extract ```python``` and ```output``` fenced blocks to execute_python tool parts and add available_functions")
    p.add_argument("input_path", help="Path to dataset saved via datasets.save_to_disk")
    p.add_argument("--output", "-o", required=True,
                   help="Output directory (if ends with '/', appends '<input>-toolExtracted')")
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--debug", action="store_true", help="Enable debug output")
    return p.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.input_path)
    if not in_path.exists():
        sys.exit(f"Input path not found: {in_path}")

    out_path = (Path(args.output) / f"{in_path.name}-toolExtracted"
                if args.output.endswith("/") else Path(args.output))
    if out_path.exists():
        if input(f"{out_path} exists. Overwrite? [y/N]: ").lower() != "y":
            sys.exit("Aborted.")

    ds = load_from_disk(str(in_path))
    nproc, bsz = args.num_proc, args.batch_size
    print(f"num_proc={nproc}, batch_size={bsz}, limit={args.limit}")
    
    if args.debug:
        print("DEBUG: Debug mode enabled")

    if isinstance(ds, DatasetDict):
        processed = {}
        for split, d in ds.items():
            print(f"Split {split}: {d.num_rows:,} rows")
            d = subset(d, args.limit)
            processed[split] = map_split(d, nproc, bsz)
        out_ds = DatasetDict(processed)
    else:
        ds = subset(ds, args.limit)
        out_ds = DatasetDict({"train": map_split(ds, nproc, bsz)})

    save_with_meta(out_ds, out_path, in_path, args)

def test_extraction():
    """Test the extraction with the example format."""
    example_text = """To solve this problem, we need to find the balance point of the rod, which is known as the center of mass. The center of mass (or center of gravity) can be calculated using the principle of moments. The principle of moments states that for an object to be in equilibrium, the sum of the clockwise moments about any point must be equal to the sum of the counterclockwise moments about the same point.

Given the weights and their distances from one end of the rod, the center of mass (x_cm) can be calculated using the formula:

x_cm = sum(m_i * x_i) / sum(m_i)

where m_i are the masses and x_i are their respective distances from a reference point.

Let's break down the problem:
1. Weights and distances from one end of the rod:
   - 20 kg at 0 meters
   - 30 kg at 1 meter
   - 40 kg at 2 meters
   - 50 kg at 3 meters
   - 60 kg at 4 meters

2. Calculate the center of mass using the provided formula.

Now let's implement this in Python using sympy to find the exact position where the rod should be supported to remain balanced.

```python
import sympy as sp

# Define the weights and their distances from one end of the rod
weights = [20, 30, 40, 50, 60]
distances = [0, 1, 2, 3, 4]

# Calculate the center of mass
numerator = sum(weight * distance for weight, distance in zip(weights, distances))
denominator = sum(weights)
x_cm = numerator / denominator

print(x_cm)
```
```output
2.5
```
The center of mass of the rod is located 2.5 meters from the end where the 20 kg weight is attached.

Thus, the point at which the rod should be supported to remain balanced is at 2.5 meters from the end with the 20 kg weight."""
    
    print("Testing extraction with example format...")
    parts = assistant_parts_from_string(example_text, debug=True)
    
    print(f"\nExtracted {len(parts)} parts:")
    for i, part in enumerate(parts):
        print(f"Part {i+1}: {part['type']}")
        if part['type'] == 'function-call':
            print(f"  Name: {part['name']}")
            print(f"  Args: {part['args'][:100]}...")
        elif part['type'] == 'function-output':
            print(f"  Content: {part['content'][:100]}...")
        else:
            print(f"  Content: {part['content'][:100]}...")
    
    return parts

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_extraction()
    else:
        main()