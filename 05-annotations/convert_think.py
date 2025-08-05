#!/usr/bin/env python3
"""
convert_think.py — final uniform schema (all messages -> content: list[blocks])

What it does
------------
1) For every message (assistant, user, etc.), `content` is a LIST of blocks.
   - Assistant:
       • Parse <think>...</think> from string content.
       • Build blocks: [{"role":"thought","content":...},{"role":"text","content":...}]
       • If already a list, keep only roles {"thought","text"} and normalize.
   - Non-assistant:
       • Wrap string content into a single block [{"role":"text","content":...}]
       • If already a list, normalize to {"role":"text","content":...} blocks.

   This avoids mixing list vs string at:
   conversation_branches[].messages[].content (fixes ArrowInvalid).

2) `system_prompt`:
   - Keep existing non-empty `system_prompt.content`.
   - Else backfill from original_metadata.chat_template_kwargs.custom_instructions.
   - Ensure `system_prompt.metadata = {}`.

3) `initial_prompt`:
   - If row already has a non-empty `initial_prompt.content`, keep it as-is.
   - Else find the earliest user message across ALL branches, set as `initial_prompt`,
     and remove that one instance from its branch.
   - Else fallback to {"role":"user","content":""}.

4) Preserve metadata everywhere (message-level and block-level; default {}).

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
ASSISTANT_BLOCK_ROLES_KEEP = {"thought", "text"}

# ------------------------------- helpers ------------------------------------ #
def extract_thinks(text: str) -> Tuple[str, List[str]]:
    """Return (cleaned_text, [think1, think2, …]) from a string."""
    thinks: List[str] = []

    def _collect(m: re.Match) -> str:
        thinks.append(m.group(1))
        return ""  # remove the whole block

    cleaned = THINK_BLOCK.sub(_collect, text or "")
    return (cleaned or "").strip(), [t.strip() for t in thinks if t and t.strip()]

def _ensure_metadata(d: Dict[str, Any]) -> Dict[str, Any]:
    if "metadata" not in d or d["metadata"] is None:
        d["metadata"] = {}
    return d

def _assistant_blocks_from_string(content: Optional[str]) -> List[Dict[str, Any]]:
    """Assistant: parse <think>, produce list of {'thought'|'text'} blocks."""
    cleaned, thinks = extract_thinks(content or "")
    blocks: List[Dict[str, Any]] = []
    for t in thinks:
        blocks.append(_ensure_metadata({"role": "thought", "content": t}))
    if cleaned:
        blocks.append(_ensure_metadata({"role": "text", "content": cleaned}))
    return blocks

def _normalize_assistant_blocks(blocks: Any) -> List[Dict[str, Any]]:
    """
    Assistant: normalize existing list content:
      - keep only roles in ASSISTANT_BLOCK_ROLES_KEEP
      - coerce strings to {"role":"text","content":...}
      - ensure 'metadata' exists
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(blocks, list):
        return out
    for b in blocks:
        if isinstance(b, str):
            out.append(_ensure_metadata({"role": "text", "content": b.strip()}))
            continue
        if not isinstance(b, dict):
            continue
        role = b.get("role")
        if role not in ASSISTANT_BLOCK_ROLES_KEEP:
            continue
        content = b.get("content")
        if isinstance(content, str):
            content = content.strip()
        out.append(_ensure_metadata({"role": role, "content": content}))
    return out

def _nonassistant_blocks_from_any(content: Any) -> List[Dict[str, Any]]:
    """
    Non-assistant: always return a list of 'text' blocks.
      - If string -> single text block.
      - If list -> flatten any dict/string into text blocks.
    """
    blocks: List[Dict[str, Any]] = []
    if isinstance(content, list):
        for x in content:
            if isinstance(x, dict):
                txt = x.get("content")
                if isinstance(txt, str):
                    txt = txt.strip()
                blocks.append(_ensure_metadata({"role": "text", "content": txt}))
            elif isinstance(x, str):
                blocks.append(_ensure_metadata({"role": "text", "content": x.strip()}))
            else:
                blocks.append(_ensure_metadata({"role": "text", "content": ""}))
    else:
        # string/None/other -> single text block
        txt = content if isinstance(content, str) else ""
        blocks.append(_ensure_metadata({"role": "text", "content": txt.strip()}))
    return blocks

def normalize_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a new message dict with:
      - role preserved
      - content -> list[blocks] (assistant vs non-assistant policies)
      - metadata ensured
    """
    role = msg.get("role")
    content = msg.get("content")
    new_msg = {"role": role}

    if role == "assistant":
        if isinstance(content, list):
            blocks = _normalize_assistant_blocks(content)
        elif isinstance(content, str) or content is None:
            blocks = _assistant_blocks_from_string(content or "")
        else:
            blocks = []
        new_msg["content"] = blocks
    else:
        # user/system/other
        new_msg["content"] = _nonassistant_blocks_from_any(content)

    new_msg = _ensure_metadata({**new_msg, "metadata": msg.get("metadata")})
    return new_msg

def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize every message so `content` is list[blocks] with consistent schema."""
    out: List[Dict[str, Any]] = []
    for m in messages or []:
        out.append(normalize_message(m))
    return out

def extract_initial_prompt_if_missing(
    current_initial: Optional[Dict[str, Any]],
    branches: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    - If current_initial has non-empty string content -> keep and do NOT remove any user msg.
    - Else: find earliest 'user' across ALL branches; set as initial_prompt; remove that instance.
    - Else fallback to {"role":"user","content":""}.
    Note: since we normalized, user content is a list[{"role":"text","content":...}], we will join text blocks.
    """
    # Respect existing non-empty initial_prompt
    if isinstance(current_initial, dict):
        c = current_initial.get("content")
        if isinstance(c, str) and c.strip():
            ip = {
                "role": current_initial.get("role", "user"),
                "content": c,
                "metadata": current_initial.get("metadata") or {}
            }
            return ip, branches

    # Search for earliest user message
    new_branches: List[Dict[str, Any]] = []
    captured: Optional[Dict[str, Any]] = None
    removed_once = False

    for br in branches or []:
        msgs = (br or {}).get("messages") or []
        new_msgs = []
        for msg in msgs:
            if (not removed_once) and msg.get("role") == "user":
                # Build a plain string from list blocks
                parts = []
                for blk in msg.get("content") or []:
                    if isinstance(blk, dict):
                        v = blk.get("content")
                        if isinstance(v, str):
                            parts.append(v)
                        elif v is None:
                            continue
                        else:
                            parts.append(str(v))
                captured = {
                    "role": "user",
                    "content": "\n".join(p for p in parts if p).strip(),
                    "metadata": msg.get("metadata") or {}
                }
                removed_once = True
                continue  # remove this user message
            new_msgs.append(msg)
        nb = dict(br)
        nb["messages"] = new_msgs
        new_branches.append(nb)

    if not captured:
        captured = {"role": "user", "content": "", "metadata": {}}

    return captured, new_branches

# ---------------------------- map operations -------------------------------- #
def process_batch(batch: Dict[str, List]) -> Dict[str, List]:
    """
    Per-batch:
      • system_prompt: keep/backfill content, ensure metadata.
      • conversation_branches: normalize ALL messages to list[blocks].
      • initial_prompt: keep if non-empty else extract earliest user across ALL branches.
    """
    import copy
    out = copy.deepcopy(batch)
    n = len(out["conversation_id"])

    # Ensure columns exist
    if "system_prompt" not in out:
        out["system_prompt"] = [None] * n
    if "initial_prompt" not in out:
        out["initial_prompt"] = [None] * n

    for i in range(n):
        # ----- system_prompt handling -----
        current_sp = (out.get("system_prompt") or [None]*n)[i]
        current_content = current_sp.get("content") if isinstance(current_sp, dict) else None

        orig_meta_i = (out.get("original_metadata") or [None]*n)[i] or {}
        ctk = (orig_meta_i.get("chat_template_kwargs") or {})
        fallback_custom = ctk.get("custom_instructions") or ""

        out["system_prompt"][i] = {
            "content": (current_content if isinstance(current_content, str) and current_content.strip()
                        else fallback_custom),
            "metadata": {}
        }

        # ----- normalize messages in branches -----
        branches = (out.get("conversation_branches") or [None]*n)[i] or []
        norm_branches: List[Dict[str, Any]] = []
        for br in branches:
            msgs = br.get("messages") or []
            norm_msgs = normalize_messages(msgs)
            nb = {"messages": norm_msgs}
            for k, v in br.items():
                if k != "messages":
                    nb[k] = v
            norm_branches.append(nb)

        # ----- initial_prompt (preserve/extract) -----
        existing_init = (out.get("initial_prompt") or [None]*n)[i]
        new_init, new_branches = extract_initial_prompt_if_missing(existing_init, norm_branches)
        out["initial_prompt"][i] = new_init
        out["conversation_branches"][i] = new_branches

    return out

def map_split(ds: Dataset, num_proc: int, batch_size: int) -> Dataset:
    return ds.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Uniform content schema (list[blocks]); embed <think>; keep/backfill system_prompt; preserve/extract initial_prompt"
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

    metadata = load_existing_metadata(in_path) or {}
    processing_entry = {
        "operation": "convert_to_new_chat_format_uniform_blocks",
        "script": "convert_think.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(in_path),
        "output_path": str(out_path),
        "batch_size": args.batch_size,
        "num_processes": args.num_proc,
        "limit": args.limit,
    }
    metadata.setdefault("processing_log", []).append(processing_entry)

    metadata_file = out_path / "dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved dataset + metadata to {out_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Convert to new chat format with uniform content blocks")
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
