#!/usr/bin/env python3
"""
convert_smoltalk2_think.py

Extract <think>...</think> blocks from response parts in new format datasets.
This script is specifically designed for datasets that already use the parts structure.

Processes messages with existing parts arrays:
- Assistant parts: Extract <think> tags from response parts into separate thought parts
- User/System parts: Keep existing response parts unchanged
- Other part types: Preserve unchanged (function-call, function-output, etc.)

Output: Messages with extracted thought parts separate from response content.
"""

import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from datasets import load_from_disk, Dataset, DatasetDict

THINK_BLOCK = re.compile(r"<think\s*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)

# ------------------------------- helpers ------------------------------------ #
def extract_thinks(text: str) -> Tuple[str, List[str]]:
    """Return (cleaned_text, [think1, think2, …]) from a string."""
    thinks: List[str] = []
    def _collect(m: re.Match) -> str:
        thinks.append(m.group(1))
        return ""  # strip the think block
    cleaned = THINK_BLOCK.sub(_collect, text or "")
    return (cleaned or "").strip(), [t.strip() for t in thinks if t and t.strip()]

def _ensure_metadata(d: Dict[str, Any]) -> Dict[str, Any]:
    if "metadata" not in d or d["metadata"] is None:
        d["metadata"] = {}
    return d

def process_response_part(part: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a response part to extract think blocks."""
    content = part.get("content", "")
    if not isinstance(content, str):
        return [part]  # Return original if not string
    
    cleaned, thinks = extract_thinks(content)
    result_parts = []
    
    # Add thought parts first
    for think_content in thinks:
        result_parts.append(_ensure_metadata({
            "type": "thought", 
            "content": think_content
        }))
    
    # Add response part with cleaned content if any remains
    if cleaned:
        result_parts.append(_ensure_metadata({
            "type": "response", 
            "content": cleaned,
            "metadata": part.get("metadata", {})
        }))
    
    return result_parts if result_parts else [part]

def process_assistant_parts(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process assistant parts array to extract think blocks from response parts."""
    result_parts = []
    
    for part in parts:
        if not isinstance(part, dict):
            continue
            
        part_type = part.get("type", "").lower()
        
        if part_type == "response":
            # Extract think blocks from response parts
            extracted_parts = process_response_part(part)
            result_parts.extend(extracted_parts)
        else:
            # Keep other part types unchanged (thought, function-call, function-output, etc.)
            result_parts.append(part)
    
    return result_parts

def normalize_message_newformat(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a message that already has parts structure.
    Extract think blocks from response parts for assistant messages.
    """
    role = msg.get("role")
    existing_parts = msg.get("parts", [])
    metadata = msg.get("metadata", {})
    
    if role == "assistant":
        # Process assistant parts to extract think blocks
        new_parts = process_assistant_parts(existing_parts)
    else:
        # Keep user/system parts unchanged
        new_parts = existing_parts
    
    result = {
        "role": role,
        "parts": new_parts,
        "metadata": metadata
    }
    
    return result

def normalize_messages_newformat(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_message_newformat(m) for m in (messages or [])]

# --------------- initial_prompt handling (same as original) --------------- #
def keep_or_lift_initial_prompt(
    current_initial: Optional[Dict[str, Any]],
    branches: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    If current initial_prompt has non-empty string content → keep & don't alter branches.
    Else: lift earliest user msg across ALL branches → initial_prompt; remove that one.
    Else: fallback to empty user initial_prompt.
    """
    if isinstance(current_initial, dict):
        c = current_initial.get("content")
        if isinstance(c, str) and c.strip():
            return {
                "role": current_initial.get("role", "user"),
                "content": c,
                "metadata": current_initial.get("metadata") or {}
            }, branches

    new_branches: List[Dict[str, Any]] = []
    captured: Optional[Dict[str, Any]] = None
    removed = False

    for br in branches or []:
        msgs = (br or {}).get("messages") or []
        nm = []
        for m in msgs:
            if not removed and m.get("role") == "user":
                # compose content from this user message's response parts
                parts = m.get("parts") or []
                texts = []
                for p in parts:
                    if isinstance(p, dict) and p.get("type") == "response":
                        v = p.get("content")
                        texts.append(v if isinstance(v, str) else ("" if v is None else str(v)))
                captured = {
                    "role": "user",
                    "content": "\n".join(t for t in texts if t).strip(),
                    "metadata": m.get("metadata") or {}
                }
                removed = True
                continue
            nm.append(m)
        nb = dict(br)
        nb["messages"] = nm
        new_branches.append(nb)

    if not captured:
        captured = {"role": "user", "content": "", "metadata": {}}

    return captured, new_branches

# ---------------------------- map operations -------------------------------- #
def process_batch(batch: Dict[str, List]) -> Dict[str, List]:
    """
    Process batch for new format datasets with existing parts.
    """
    import copy
    out = copy.deepcopy(batch)
    n = len(out["conversation_id"])

    if "system_prompt" not in out:
        out["system_prompt"] = [None] * n
    if "initial_prompt" not in out:
        out["initial_prompt"] = [None] * n
    if "available_functions" not in out:
        out["available_functions"] = [[] for _ in range(n)]

    for i in range(n):
        # Ensure available_functions is always a list (not None)
        if "available_functions" in out:
            if out["available_functions"][i] is None:
                out["available_functions"][i] = []
        
        # system_prompt (same logic as original)
        sp = (out.get("system_prompt") or [None]*n)[i]
        current_content = sp.get("content") if isinstance(sp, dict) else None
        orig_meta_i = (out.get("original_metadata") or [None]*n)[i] or {}
        ctk = (orig_meta_i.get("chat_template_kwargs") or {})
        fallback = ctk.get("custom_instructions") or ""
        out["system_prompt"][i] = {
            "content": (current_content if isinstance(current_content, str) and current_content.strip() else fallback),
            "metadata": {}
        }

        # branches -> process parts to extract think blocks
        in_branches = (out.get("conversation_branches") or [None]*n)[i] or []
        new_branches: List[Dict[str, Any]] = []
        for br in in_branches:
            msgs = br.get("messages") or []
            norm_msgs = normalize_messages_newformat(msgs)
            # ensure no 'content' key leaked (should not be present in new format)
            for m in norm_msgs:
                m.pop("content", None)
            nb = {"messages": norm_msgs}
            for k, v in br.items():
                if k != "messages":
                    nb[k] = v
            new_branches.append(nb)

        # initial_prompt (keep or lift earliest user)
        existing_init = (out.get("initial_prompt") or [None]*n)[i]
        new_init, stripped = keep_or_lift_initial_prompt(existing_init, new_branches)
        out["initial_prompt"][i] = new_init
        out["conversation_branches"][i] = stripped

    return out

def map_split(ds: Dataset, num_proc: int, batch_size: int) -> Dataset:
    return ds.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Extract <think> blocks from response parts in new format datasets"
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
        "operation": "convert_smoltalk2_think",
        "script": "convert_smoltalk2_think.py",
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
    p = argparse.ArgumentParser(
        description="Extract <think> blocks from response parts in new format datasets (parts structure)"
    )
    p.add_argument("input_path", help="Path to new format dataset saved via datasets.save_to_disk")
    p.add_argument("--output", "-o", required=True,
                   help="Output dir (if ends with '/', appends '<input>-ThinkFormatted')")
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Process first N rows of each split")
    return p.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.input_path)
    if not in_path.exists():
        sys.exit(f"Input path not found: {in_path}")

    out_path = (Path(args.output) / f"{in_path.name}-ThinkFormatted"
                if args.output.endswith("/") else Path(args.output))
    if out_path.exists():
        if input(f"{out_path} exists. Overwrite? [y/N]: ").lower() != "y":
            sys.exit("Aborted.")

    ds = load_from_disk(str(in_path))
    nproc, bsz = args.num_proc, args.batch_size
    print(f"num_proc={nproc}, batch_size={bsz}, limit={args.limit}")

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

if __name__ == "__main__":
    main()