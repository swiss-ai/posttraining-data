#!/usr/bin/env python3
"""Build a self-contained HTML viewer for decontamination removal logs.

Shows removed training prompts next to their matched benchmark prompts, with the matched token runs
highlighted in both. One dataset can be selected at a time.

  python removal_log_viewer.py OUT.html DIR [DIR ...] [--sample 300] [--tokenizer_name ...]
                               [--compare_to OLD_DIR [--diff_sample 100]]

Each DIR is a decontaminated output (or any directory) containing `decontamination_removed/`; the
dataset is named after DIR. Datasets without removals are listed too, with empty views.

--compare_to adds an older pipeline version: OLD_DIR/<dataset>/decontamination_removed. Each dataset
then has four views: the removals of the new and of the old version, and the samples removed only by
one of them (matched by content, since conversation_ids may differ between versions). Counts are
always complete; the samples shown are random. Training prompts longer than
EXCERPT_ABOVE characters are shown as an excerpt: their start plus context around every matched run. Highlights are recomputed with the same matching as the decontamination
step (difflib runs of >= 5 tokens), so pass the tokenizer the reports were made with.
"""
import argparse
import collections
import difflib
import hashlib
import json
import random
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from removal_log import MIN_BLOCK, REMOVAL_LOG_DIR

HEAD_CHARS = 1500  # long training prompts keep their start ...
CONTEXT_CHARS = 600  # ... and this much context around every matched run; the rest is omitted
EXCERPT_ABOVE = 4000  # prompts up to this length are shown in full
MAX_MATCHES = 10  # matches shown per sample (highest ratio first), to keep the page small
NO_OWN_MATCH = "(no own match: shared conversation_id)"
KEY_COLUMNS = ["system_prompt", "initial_prompt", "conversation_branches"]
ROW_COLUMNS = ["conversation_id", "initial_prompt", "decontamination_matches"]


def excerpt(text, spans_per_match):
    """Shorten `text` to its start plus context around all spans; return (text, shifted spans per match)."""
    if len(text) <= EXCERPT_ABOVE:
        return text, spans_per_match
    keep = [[0, HEAD_CHARS]] + [[max(0, a - CONTEXT_CHARS), min(len(text), b + CONTEXT_CHARS)]
                                for spans in spans_per_match for a, b in spans]
    keep.sort()
    merged = [keep[0]]
    for a, b in keep[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    parts, shift, pos = [], [], 0  # shift: (original start, original end, new start) per kept range
    for a, b in merged:
        if a > pos:
            parts.append(f"\n[… {a - pos:,} characters omitted …]\n")
        shift.append((a, b, sum(map(len, parts))))
        parts.append(text[a:b])
        pos = b
    if pos < len(text):
        parts.append(f"\n[… {len(text) - pos:,} characters omitted …]")

    def move(a, b):
        for start, end, new in shift:
            if start <= a and b <= end:
                return [new + a - start, new + b - start]

    return "".join(parts), [[s for a, b in spans if (s := move(a, b))] for spans in spans_per_match]


def char_spans(offsets, start, size):
    """Character span of tokens [start, start + size), skipping special tokens (offset (0, 0))."""
    toks = [o for o in offsets[start:start + size] if o[1] > o[0]]
    return [toks[0][0], toks[-1][1]] if toks else None


def match_spans(tok, train_text, bench_text):
    a = tok(train_text, return_offsets_mapping=True)
    b = tok(bench_text, return_offsets_mapping=True)
    blocks = [m for m in difflib.SequenceMatcher(None, a["input_ids"], b["input_ids"], autojunk=False)
              .get_matching_blocks() if m.size >= MIN_BLOCK]
    train = [s for m in blocks if (s := char_spans(a["offset_mapping"], m.a, m.size))]
    bench = [s for m in blocks if (s := char_spans(b["offset_mapping"], m.b, m.size))]
    return train, bench


def family(benchmark):
    return "__".join(benchmark.split("__")[:2])


def content_key(r):
    """Identity of a sample across pipeline versions, whose conversation_ids may differ: system prompt,
    first user turn and all conversation branches."""
    s = lambda x: "" if x is None else str(x)
    branches = [[(s(m.get("role")), [(s(p.get("type")), s(p.get("content")), s(p.get("name"))) for p in m.get("parts") or []])
                 for m in (b or {}).get("messages") or []] for b in r.get("conversation_branches") or []]
    key = [s((r.get("system_prompt") or {}).get("content")), s((r.get("initial_prompt") or {}).get("content")), branches]
    return hashlib.sha1(json.dumps(key, ensure_ascii=False).encode()).hexdigest()


def load_log(log_path: Path):
    """(split, row, content key) of every removed sample in a removal log; [] if it does not exist."""
    if not log_path.exists():
        return []
    log = load_from_disk(str(log_path))
    rows = []
    for split, data in (log.items() if hasattr(log, "keys") else [(None, log)]):
        cols = [c for c in KEY_COLUMNS + ROW_COLUMNS if c in data.column_names]
        for batch in data.select_columns(cols).iter(batch_size=1000):
            for i in range(len(batch["conversation_id"])):
                r = {c: batch[c][i] for c in cols}
                rows.append((split, {c: r[c] for c in ROW_COLUMNS if c in r}, content_key(r)))
    return rows


def only_in(rows, other):
    """Rows whose content is not (or less often) in `other`."""
    left = collections.Counter(k for _, _, k in other)
    out = []
    for row in rows:
        if left[row[2]] > 0:
            left[row[2]] -= 1
        else:
            out.append(row)
    return out


def build_item(tok, split, r, threshold):
    text = r["initial_prompt"]["content"]
    ms = sorted(r["decontamination_matches"], key=lambda m: -m["match_ratio"])
    matches, train_spans = [], []
    for m in ms[:MAX_MATCHES]:
        ts, bs = match_spans(tok, text, m["benchmark_prompt"])
        train_spans.append(ts)
        matches.append({
            "benchmark": m["benchmark"], "family": family(m["benchmark"]), "eval_idx": m["eval_idx"],
            "ratio": round(m["match_ratio"], 4), "longest": m["longest_block"],
            "bench_tokens": m["benchmark_prompt_tokens"], "bench": m["benchmark_prompt"], "bench_spans": bs,
        })
    shown_text, train_spans = excerpt(text, train_spans)
    for m, ts in zip(matches, train_spans):
        m["train_spans"] = ts
    return {"id": r["conversation_id"], "split": split, "train": shown_text, "train_chars": len(text),
            "excerpt": len(shown_text) != len(text), "matches": matches, "matches_total": len(ms),
            "own": any(m["match_ratio"] >= threshold for m in ms)}


def build_view(tok, rows, sample: int, rng, threshold):
    """One list of removals: full counts, per-benchmark counts and a random sample of items.

    Benchmarks are counted over matches reaching the threshold; samples without such a match (removed
    because another sample had the same conversation_id) are counted as NO_OWN_MATCH."""
    rows = [(s, r) for s, r, _ in rows if r["decontamination_matches"]]
    fams = collections.Counter()
    for _, r in rows:
        own = {family(m["benchmark"]) for m in r["decontamination_matches"] if m["match_ratio"] >= threshold}
        fams.update(own or [NO_OWN_MATCH])
    shown = rng.sample(rows, sample) if len(rows) > sample else rows
    return {"removed": len(rows), "families": fams.most_common(),
            "items": [build_item(tok, s, r, threshold) for s, r in shown]}


def build_dataset(tok, name: str, new_path, old_path, sample: int, diff_sample: int, rng, threshold):
    new = load_log(new_path / REMOVAL_LOG_DIR) if new_path else []
    old = load_log(old_path / REMOVAL_LOG_DIR) if old_path else []
    lists = {"new": (new, sample), "old": (old, sample)}
    if old_path:
        lists.update({"new_only": (only_in(new, old), diff_sample), "old_only": (only_in(old, new), diff_sample)})
    views = {k: build_view(tok, rows, n, rng, threshold) for k, (rows, n) in lists.items()}
    print(f"{name}: " + ", ".join(f"{k} {v['removed']} (showing {len(v['items'])})" for k, v in views.items()), flush=True)
    return {"name": name, "views": views}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--compare_to", help="Directory with <dataset>/decontamination_removed logs of an older version")
    p.add_argument("--sample", type=int, default=300, help="Removed samples shown per dataset and version (random)")
    p.add_argument("--diff_sample", type=int, default=100, help="Samples shown per difference list (random)")
    p.add_argument("--diff_threshold", type=float, default=0.5, help="Threshold the reports were made with")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer_name", default="swiss-ai/Apertus-8B-Instruct-2509")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    rng = random.Random(args.seed)
    new = {Path(d).name: Path(d) for d in args.dirs}
    old = {d.name: d for d in Path(args.compare_to).iterdir() if d.is_dir()} if args.compare_to else {}
    names = list(new) + sorted(n for n in old if n not in new)
    data = {"tokenizer": args.tokenizer_name, "threshold": args.diff_threshold, "compared": bool(old),
            "no_own_match": NO_OWN_MATCH,
            "datasets": [build_dataset(tok, n, new.get(n), old.get(n) if old else None, args.sample,
                                       args.diff_sample, rng, args.diff_threshold) for n in names]}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("\ufffd", "\\ufffd")  # broken bytes from token decoding, escaped
    html = (Path(__file__).with_name("removal_log_viewer.html").read_text()
            .replace("/*__DATA__*/null", payload))
    Path(args.out).write_text(html)
    print(f"Wrote {args.out} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
