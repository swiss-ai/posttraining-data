"""Removal log for decontamination: the removed samples together with what they matched.

Written by decontamination.py and merge-decontamination-reports.py at filter time to
<output>/decontamination_removed/. It has the same structure as the input (DatasetDict with the same
splits, or a single Dataset) and keeps every removed row unchanged, plus a `decontamination_matches`
column with one entry per benchmark whose report flagged the sample:

  benchmark               benchmark (prompt split) name
  eval_idx                index of the matched prompt in that benchmark
  benchmark_prompt        the matched benchmark prompt
  benchmark_prompt_tokens its length in tokens
  match_ratio             share of its tokens matched in aligned runs of >= MIN_BLOCK tokens
                          (the value thresholded by diff_threshold)
  longest_block           longest matched run in tokens
  matched_text            the matched runs, decoded and joined with " [...] "

Reports only store the first benchmark prompt above the threshold, so `benchmark_prompt` is a match
of that benchmark, not necessarily its closest one. match_ratio is recomputed here with the same
matching as the decontamination step, so it is only meaningful with the same tokenizer.
"""
import difflib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict

from datasets import DatasetDict, Features, Value
from transformers import AutoTokenizer

REMOVAL_LOG_DIR = "decontamination_removed"
MIN_BLOCK = 5  # same minimum block size as the decontamination matching
MATCH_FEATURES = {
    "benchmark": Value("string"),
    "eval_idx": Value("int64"),
    "benchmark_prompt": Value("string"),
    "benchmark_prompt_tokens": Value("int64"),
    "match_ratio": Value("float64"),
    "longest_block": Value("int64"),
    "matched_text": Value("string"),
}

_tokenizer = None


def _init_worker(tokenizer_name):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def _score(pair):
    train_text, eval_text = pair
    train_tokens = _tokenizer(train_text)["input_ids"]
    eval_tokens = _tokenizer(eval_text)["input_ids"]
    blocks = [b for b in difflib.SequenceMatcher(None, train_tokens, eval_tokens, autojunk=False).get_matching_blocks()
              if b.size >= MIN_BLOCK]
    return {
        "benchmark_prompt_tokens": len(eval_tokens),
        "match_ratio": sum(b.size for b in blocks) / len(eval_tokens) if eval_tokens else 0.0,
        "longest_block": max((b.size for b in blocks), default=0),
        "matched_text": " [...] ".join(_tokenizer.decode(eval_tokens[b.b:b.b + b.size]) for b in blocks),
    }


def write_removal_log(data, reports: Dict[str, Dict[str, int]], decontamination_prompts, tokenizer_name: str,
                      output_path: Path, diff_threshold: float, num_proc: int = 16) -> dict:
    """Save the rows of `data` flagged in `reports` with their matches to <output_path>/decontamination_removed.

    `reports` maps a benchmark (prompt split name of `decontamination_prompts`) to its report
    {conversation_id: eval_idx}. Returns a summary for the processing log.
    """
    matches_by_id = defaultdict(list)
    for benchmark, report in reports.items():
        if not report:
            continue
        idxs = sorted(set(report.values()))
        prompts = dict(zip(idxs, decontamination_prompts[benchmark].select(idxs)["prompt"]))
        for cid, idx in report.items():
            matches_by_id[str(cid)].append({"benchmark": benchmark, "eval_idx": int(idx),
                                            "benchmark_prompt": str(prompts[idx])})

    splits = data if isinstance(data, DatasetDict) else {None: data}
    removed = {k: v.filter(lambda x: str(x["conversation_id"]) in matches_by_id) for k, v in splits.items()}

    # Score every (removed sample, matched benchmark prompt) pair
    pairs, owners = [], []
    for k, split in removed.items():
        for cid, initial_prompt in zip(split["conversation_id"], split["initial_prompt"]):
            for m in matches_by_id[str(cid)]:
                pairs.append((initial_prompt["content"], m["benchmark_prompt"]))
                owners.append(m)
    print(f"Scoring {len(pairs)} matched pairs of {len(matches_by_id)} removed samples for the removal log...")
    if pairs:
        with Pool(num_proc, initializer=_init_worker, initargs=(tokenizer_name,)) as p:
            for m, score in zip(owners, p.imap(_score, pairs, chunksize=64)):
                m.update(score)

    below = sum(m["match_ratio"] < diff_threshold for m in owners)
    if below:
        print(f"Warning: {below}/{len(owners)} logged matches score below diff_threshold={diff_threshold}. "
              f"Was decontamination run with a different tokenizer than {tokenizer_name}?")

    def add_matches(batch):
        return {"decontamination_matches": [matches_by_id[str(cid)] for cid in batch["conversation_id"]]}

    log = {}
    for k, split in removed.items():
        features = Features({**split.features, "decontamination_matches": [MATCH_FEATURES]})
        log[k] = split.map(add_matches, batched=True, features=features)
    log = DatasetDict(log) if isinstance(data, DatasetDict) else log[None]

    log_path = Path(output_path) / REMOVAL_LOG_DIR
    print(f"Saving removal log to {log_path}...")
    log.save_to_disk(str(log_path))

    removed_by_benchmark = {b: len(r) for b, r in sorted(reports.items(), key=lambda x: -len(x[1])) if r}
    return {
        "removal_log_path": str(log_path),
        "removed_by_benchmark": removed_by_benchmark,
        "logged_matches_below_threshold": below,
    }
