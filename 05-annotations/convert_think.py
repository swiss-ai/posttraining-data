#!/usr/bin/env python3
"""
convert_think.py  — final (with system_prompt from original_metadata)

What it does
------------
1) Promotes every <think>...</think> block found in ASSISTANT messages into its own
   message: {"role": "thought", "content": "..."}
2) Cleans the assistant message by removing the <think> blocks and keeps it (if non-empty)
3) Ensures messages are minimal dicts with {"role", "content"} and preserves metadata
4) Sets `system_prompt.content` from
     original_metadata.chat_template_kwargs.custom_instructions
   (creates `system_prompt` if missing; sets `metadata` = {})

CLI
---
python convert_think.py <INPUT_DATASET_DIR> \
  --output <OUT_DIR or prefix/> \
  [--batch-size 10000] [--num-proc 8] [--limit N]

• If --output ends with '/', the script appends "<input_name>-thinkPromoted".
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
    """Return (cleaned_text, [think1, think2, …])."""
    thinks: List[str] = []

    def _collect(m: re.Match) -> str:
        thinks.append(m.group(1))
        return ""  # remove the whole block

    cleaned = THINK_BLOCK.sub(_collect, text or "")
    return cleaned.strip(), [t.strip() for t in thinks if t and t.strip()]

def minimal(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a message to {'role','content'} with preserved metadata (strip whitespace)."""
    result = {"role": msg.get("role"), "content": (msg.get("content") or "").strip()}
    # Preserve metadata if it exists to avoid data loss
    if 'metadata' in msg and msg['metadata']:
        result['metadata'] = msg['metadata']
    return result

def promote_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Insert thought messages before cleaned assistant; preserve metadata."""
    import copy
    out: List[Dict[str, Any]] = []
    
    for m in messages or []:
        # Create a deep copy of the message to ensure complete decoupling
        message_copy = copy.deepcopy(m)
        role = message_copy.get("role")
        
        if role != "assistant":
            out.append(minimal(message_copy))
            continue

        cleaned, thinks = extract_thinks(message_copy.get("content", ""))
        
        # Create new thought messages (completely new objects)
        for t in thinks:
            out.append({"role": "thought", "content": t})
        
        # Create new assistant message if there's cleaned content
        if cleaned:
            # Preserve original metadata but update content
            new_assistant_msg = minimal(message_copy)
            new_assistant_msg["content"] = cleaned
            out.append(new_assistant_msg)
    return out

# ---------------------------- map operations -------------------------------- #
def process_batch(batch: Dict[str, List]) -> Dict[str, List]:
    """
    Per-batch transform:
      • system_prompt.content ← original_metadata.chat_template_kwargs.custom_instructions
      • promote <think> to thought messages; minimize each message
    """
    import copy
    out = copy.deepcopy(batch)
    n = len(out["conversation_id"])

    # Ensure system_prompt column exists
    if "system_prompt" not in out:
        out["system_prompt"] = [None] * n

    for i in range(n):
        # ----- system_prompt from nested original_metadata -----
        orig_meta_i = (out.get("original_metadata") or [None]*n)[i] or {}
        ctk = (orig_meta_i.get("chat_template_kwargs") or {})
        custom_val = ctk.get("custom_instructions")

        # Guarantee shape
        out["system_prompt"][i] = {
            "content": custom_val or "",
            "metadata": {}
        }

        # ----- promote <think> blocks inside messages -----
        branches = (out.get("conversation_branches") or [None]*n)[i] or []
        new_branches = []
        for br in branches:
            msgs = br.get("messages") or []
            # Create completely new branch structure to avoid in-place modification
            new_branch = {"messages": promote_messages(msgs)}
            # Preserve any other branch-level metadata
            for key, value in br.items():
                if key != "messages":
                    new_branch[key] = value
            new_branches.append(new_branch)
        out["conversation_branches"][i] = new_branches

    return out

def map_split(ds: Dataset, num_proc: int, batch_size: int) -> Dataset:
    return ds.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Promoting <think> → thought & setting system_prompt from original_metadata"
    )

# ------------------------------- I/O & main --------------------------------- #
def subset(ds: Dataset, n: Optional[int]) -> Dataset:
    return ds if n is None or n <= 0 or n >= ds.num_rows else ds.select(range(n))

def load_existing_metadata(input_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    metadata_file = Path(input_path) / "dataset_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None

def save_with_meta(out_ds: DatasetDict, out_path: Path,
                   in_path: Path, args: argparse.Namespace):
    out_path.mkdir(parents=True, exist_ok=True)
    out_ds.save_to_disk(str(out_path))

    # Load or create metadata
    metadata = load_existing_metadata(in_path) or {}
    
    # Add processing log entry
    processing_entry = {
        "operation": "convert_think_and_system_prompt_from_original_metadata",
        "script": "convert_think.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(in_path),
        "output_path": str(out_path),
        "batch_size": args.batch_size,
        "num_processes": args.num_proc,
        "limit": args.limit,
    }
    
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Save updated metadata
    metadata_file = out_path / "dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved dataset + metadata to {out_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Promote <think> blocks and set system_prompt from original_metadata.chat_template_kwargs.custom_instructions")
    p.add_argument("input_path", help="Path to dataset saved via datasets.save_to_disk")
    p.add_argument("--output", "-o", required=True,
                   help="Output directory (if ends with '/', appends '<input>-thinkPromoted')")
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--num-proc",   type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Process only first N rows of each split")
    return p.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.input_path)
    if not in_path.exists():
        sys.exit(f"Input path not found: {in_path}")

    out_path = (Path(args.output) / f"{in_path.name}-thinkPromoted"
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
