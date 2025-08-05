#!/usr/bin/env python3
"""
convert_to_parts_schema.py — final

Outputs messages[*] with:
  { "role": "<user|assistant|system>", "parts": [ ... ], "metadata": {} }

Assistant parts:
  - "thought" (from <think>…</think>)
  - "response" (assistant visible text)
  - optional tool parts passed through / normalized:
      "function-call"  {name, args}
      "function-output" {content}
      "verifiable-responses" {answers}

User/System parts:
  - single {type:"response", content:"..."}

Also:
  - Keep existing non-empty initial_prompt as-is (don’t remove any user turn).
  - Else lift earliest user across ALL branches into initial_prompt and remove that one.
  - Keep/backfill system_prompt.content from original_metadata.chat_template_kwargs.custom_instructions.
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

def _norm_type(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    t = str(raw).lower().replace("_", "-")
    if t in ("response", "thought"):
        return t
    if t in ("function-call",):
        return "function-call"
    if t in ("function-output",):
        return "function-output"
    if t in ("verifiable-response", "verifiable-responses"):
        return "verifiable-responses"
    return None

# --------------- parts builders ---------------- #
def assistant_parts_from_string(content: Optional[str]) -> List[Dict[str, Any]]:
    """Assistant string → parts: [thought*] + response?"""
    cleaned, thinks = extract_thinks(content or "")
    parts: List[Dict[str, Any]] = []
    for t in thinks:
        parts.append(_ensure_metadata({"type": "thought", "content": t}))
    if cleaned:
        parts.append(_ensure_metadata({"type": "response", "content": cleaned}))
    return parts

def assistant_parts_from_blocks(blocks: Any) -> List[Dict[str, Any]]:
    """
    Accept legacy assistant blocks:
      {role: text|assistant_text|thought} OR
      {type: response|thought|function-call|function-output|verifiable-response(s)} OR
      plain strings (→ response).
    Preserve tool fields (name/args/content/answers).
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(blocks, list):
        return out

    for b in blocks:
        if isinstance(b, str):
            out.append(_ensure_metadata({"type": "response", "content": b.strip()}))
            continue
        if not isinstance(b, dict):
            continue

        # prefer explicit role→type mapping if present; else use 'type'
        role = b.get("role")
        typ = b.get("type")
        if role:
            r = str(role).lower()
            if r in ("text", "assistant_text"):
                typ = "response"
            elif r == "thought":
                typ = "thought"
        tnorm = _norm_type(typ)

        if tnorm is None:
            # Unknown block (e.g., tool stuff we don't recognize) → skip
            continue

        if tnorm in ("response", "thought"):
            part = {"type": tnorm, "content": b.get("content")}
            out.append(_ensure_metadata(part))
        elif tnorm == "function-call":
            out.append({
                "type": "function-call",
                "name": b.get("name"),
                "args": b.get("args") or b.get("arguments") or {},
            })
        elif tnorm == "function-output":
            out.append({
                "type": "function-output",
                "content": b.get("content"),
            })
        elif tnorm == "verifiable-responses":
            answers = b.get("answers")
            if answers is None and "answer" in b:
                answers = [b.get("answer")]
            out.append({
                "type": "verifiable-responses",
                "answers": answers or []
            })

    return out

def user_or_system_parts_from_any(content: Any) -> List[Dict[str, Any]]:
    """User/System → one response part with string content (join list-y content)."""
    if isinstance(content, list):
        pieces = []
        for x in content:
            if isinstance(x, dict):
                v = x.get("content")
                pieces.append(v if isinstance(v, str) else ("" if v is None else str(v)))
            elif isinstance(x, str):
                pieces.append(x)
        text = "\n".join(p for p in pieces if p)
    else:
        text = content if isinstance(content, str) else ""
    return [_ensure_metadata({"type": "response", "content": text.strip()})]

def normalize_message_to_parts(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert any input message into {role, parts[], metadata} — removes any 'content' key entirely.
    """
    role = msg.get("role")
    content = msg.get("content")
    out = {"role": role}

    if role == "assistant":
        if isinstance(content, list):
            parts = assistant_parts_from_blocks(content)
        else:
            parts = assistant_parts_from_string(content)
        out["parts"] = parts
    else:
        out["parts"] = user_or_system_parts_from_any(content)

    out = _ensure_metadata({**out, "metadata": msg.get("metadata")})
    return out

def normalize_messages_to_parts(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_message_to_parts(m) for m in (messages or [])]

# --------------- initial_prompt handling --------------- #
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
    Per-row:
      • system_prompt: keep/backfill content; ensure metadata.
      • conversation_branches: convert ALL messages to {role, parts[], metadata}.
      • initial_prompt: keep if non-empty else lift earliest user.
      • Never keep 'content' on messages (only 'parts').
    """
    import copy
    out = copy.deepcopy(batch)
    n = len(out["conversation_id"])

    if "system_prompt" not in out:
        out["system_prompt"] = [None] * n
    if "initial_prompt" not in out:
        out["initial_prompt"] = [None] * n

    for i in range(n):
        # system_prompt
        sp = (out.get("system_prompt") or [None]*n)[i]
        current_content = sp.get("content") if isinstance(sp, dict) else None
        orig_meta_i = (out.get("original_metadata") or [None]*n)[i] or {}
        ctk = (orig_meta_i.get("chat_template_kwargs") or {})
        fallback = ctk.get("custom_instructions") or ""
        out["system_prompt"][i] = {
            "content": (current_content if isinstance(current_content, str) and current_content.strip() else fallback),
            "metadata": {}
        }

        # branches -> normalize to parts (and REMOVE any 'content' keys)
        in_branches = (out.get("conversation_branches") or [None]*n)[i] or []
        new_branches: List[Dict[str, Any]] = []
        for br in in_branches:
            msgs = br.get("messages") or []
            norm_msgs = normalize_messages_to_parts(msgs)
            # ensure no 'content' key leaked
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
        desc="Convert to messages[*].parts; keep tools; backfill system_prompt; keep/lift initial_prompt"
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
        "operation": "convert_to_parts_schema_final",
        "script": "convert_to_parts_schema.py",
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
    p = argparse.ArgumentParser(description="Convert dataset to messages[*].parts schema (assistant/user/system)")
    p.add_argument("input_path", help="Path to dataset saved via datasets.save_to_disk")
    p.add_argument("--output", "-o", required=True,
                   help="Output dir (if ends with '/', appends '<input>-thinkPromoted')")
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Process first N rows of each split")
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
