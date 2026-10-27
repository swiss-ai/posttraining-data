"""Removal log for decontamination: the removed samples together with what they matched.

Written by merge-decontamination-reports.py at filter time to
<output>/decontamination_removed/. It has the same structure as the input (DatasetDict with the same
splits, or a single Dataset) and keeps every removed row unchanged, plus a `decontamination_matches`
column with one entry per benchmark whose report flagged the sample:

  benchmark               benchmark (prompt split) name
  eval_idx                index of the matched prompt in that benchmark
  benchmark_prompt        the matched benchmark prompt
  benchmark_prompt_tokens its length in units (tokens, numbers as one unit; see matching_units.py)
  match_ratio             share of its units matched in aligned runs of >= MIN_BLOCK units (or of the
                          whole prompt, if it is shorter than MIN_BLOCK)
                          (the value thresholded by diff_threshold)
  longest_block           longest matched run in units
  matched_text            the matched runs of the normalised benchmark prompt, joined with " [...] "
  user_turn               the user message with the highest match_ratio: its index in
                          matching_units.user_prompts (0 = initial prompt)

Reports only store the first benchmark prompt above the threshold, so `benchmark_prompt` is a match
of that benchmark, not necessarily its closest one. match_ratio is recomputed here with the same
matching as the decontamination step, so it is only meaningful with the same tokenizer and
matching_units.UNITS_VERSION.
"""
import difflib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict

from datasets import Dataset, DatasetDict, Features, Value
from transformers import AutoTokenizer

from matching_units import units_with_spans, user_prompts

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
    "user_turn": Value("int64"),
}

_tokenizer = None


def _init_worker(tokenizer_name):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def _score(pair):
    train_text, eval_text = pair
    _, train_tokens, _ = units_with_spans(_tokenizer, train_text)
    eval_norm, eval_tokens, eval_spans = units_with_spans(_tokenizer, eval_text)
    blocks = [b for b in difflib.SequenceMatcher(None, train_tokens, eval_tokens, autojunk=False).get_matching_blocks()
              if b.size >= max(1, min(MIN_BLOCK, len(eval_tokens)))]
    return {
        "benchmark_prompt_tokens": len(eval_tokens),
        "match_ratio": sum(b.size for b in blocks) / len(eval_tokens) if eval_tokens else 0.0,
        "longest_block": max((b.size for b in blocks), default=0),
        "matched_text": " [...] ".join(eval_norm[eval_spans[b.b][0]:eval_spans[b.b + b.size - 1][1]] for b in blocks),
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
    removed = {k: v.filter(lambda ids: [str(c) in matches_by_id for c in ids], input_columns="conversation_id",
                           batched=True) for k, v in splits.items()}

    # Score every (user message of a removed sample, matched benchmark prompt) pair; keep the best turn
    pairs, owners, scored = [], [], []
    for k, split in removed.items():
        for cid, initial_prompt, branches in zip(split["conversation_id"], split["initial_prompt"],
                                                 split["conversation_branches"]):
            scored.extend(matches_by_id[str(cid)])
            for turn, text in enumerate(user_prompts(initial_prompt, branches)):
                for m in matches_by_id[str(cid)]:
                    pairs.append((text, m["benchmark_prompt"]))
                    owners.append((m, turn))
    print(f"Scoring {len(pairs)} (user message, benchmark prompt) pairs of {len(matches_by_id)} removed samples "
          f"for the removal log...")
    if pairs:
        with Pool(num_proc, initializer=_init_worker, initargs=(tokenizer_name,)) as p:
            for (m, turn), score in zip(owners, p.imap(_score, pairs, chunksize=64)):
                if "match_ratio" not in m or score["match_ratio"] > m["match_ratio"]:
                    m.update(score, user_turn=turn)

    below = sum(m["match_ratio"] < diff_threshold for m in scored)
    if below:
        print(f"Warning: {below}/{len(scored)} logged matches score below diff_threshold={diff_threshold}. "
              f"Was decontamination run with a different tokenizer than {tokenizer_name}?")

    def add_matches(batch):
        return {"decontamination_matches": [matches_by_id[str(cid)] for cid in batch["conversation_id"]]}

    log = {}
    for k, split in removed.items():
        features = Features({**split.features, "decontamination_matches": [MATCH_FEATURES]})
        # map over zero rows would drop the new column, so build empty logs directly
        log[k] = (split.map(add_matches, batched=True, features=features) if len(split)
                  else Dataset.from_dict({c: [] for c in features}, features=features))
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
