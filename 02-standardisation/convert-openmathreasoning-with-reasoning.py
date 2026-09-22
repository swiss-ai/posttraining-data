#!/usr/bin/env python3
"""
Convert nvidia/OpenMathReasoning to the unified chat format while preserving
reasoning traces and filtering low-quality reasoning.

Filters added on top of the regular OpenMathReasoning converter:
- reasoning length must be <= --max-reasoning-tokens using the Apertus IT tokenizer
- chunked n-gram TTR over reasoning tokens must be >= --min-chunk-ttr

The reasoning is written as an assistant "thought" part, followed by the final
solution as a "response" part.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer


SRC = "nvidia/OpenMathReasoning"
UTC = timezone.utc
DEFAULT_TOKENIZER = "swiss-ai/Apertus-8B-Instruct-2509"
_TOKENIZER = None
_TOKENIZER_NAME = None


def parse_pass_rate(raw: str) -> Optional[float]:
    """Convert pass_rate_72b_tir string to float, or None for 'n/a'."""
    if not raw or raw == "n/a":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def extract_reasoning_and_solution(generated_solution: str) -> tuple[str, str]:
    """Split <think>...</think> traces from the final visible solution."""
    think_blocks = []
    for match in re.finditer(r"<think>(.*?)</think>", generated_solution, flags=re.DOTALL):
        content = match.group(1).strip()
        if content:
            think_blocks.append(content)
    solution = re.sub(r"<think>.*?</think>", "", generated_solution, flags=re.DOTALL).strip()
    return "\n\n".join(think_blocks), solution


def prompt_text(sample: Dict[str, Any]) -> str:
    """Extract prompt text from raw OpenMath rows or converted chat rows."""
    if sample.get("problem") is not None:
        return str(sample["problem"])

    initial_prompt = sample.get("initial_prompt")
    if isinstance(initial_prompt, dict):
        content = initial_prompt.get("content", "")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    if isinstance(initial_prompt, str):
        return initial_prompt

    return ""


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ngram_ttr(token_ids: list[int], n: int) -> float:
    if len(token_ids) < n:
        return 1.0 if token_ids else 0.0
    ngrams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def chunked_ttr(
    token_ids: list[int],
    chunk_size: int,
    n: int,
    aggregation: str,
) -> tuple[float, list[float]]:
    if not token_ids:
        return 0.0, []
    scores = [
        ngram_ttr(token_ids[start:start + chunk_size], n)
        for start in range(0, len(token_ids), chunk_size)
    ]
    if aggregation == "mean":
        return mean(scores), scores
    return min(scores), scores


def get_tokenizer(tokenizer_name: str):
    global _TOKENIZER, _TOKENIZER_NAME
    if _TOKENIZER is None or _TOKENIZER_NAME != tokenizer_name:
        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=True)
        _TOKENIZER_NAME = tokenizer_name
    return _TOKENIZER


def make_parts(reasoning: str, solution: str, expected_answer: str | None) -> list[Dict[str, Any]]:
    parts: list[Dict[str, Any]] = [
        {
            "type": "thought",
            "content": reasoning,
            "metadata": {},
        },
        {
            "type": "response",
            "content": solution,
            "metadata": {},
        },
    ]
    if expected_answer:
        parts.append({
            "type": "verifiable-responses",
            "answers": [expected_answer],
        })
    return parts


def quality_status_for_batch(batch: dict[str, list[Any]], args: argparse.Namespace) -> dict[str, list[Any]]:
    tokenizer = get_tokenizer(args.tokenizer_name)
    generated_solutions = batch["generated_solution"]
    extracted = [extract_reasoning_and_solution(text) for text in generated_solutions]
    reasonings = [item[0] for item in extracted]
    solutions = [item[1] for item in extracted]

    reasoning_token_ids = tokenizer(reasonings, add_special_tokens=False)["input_ids"]
    if args.min_response_tokens > 0:
        response_token_ids = tokenizer(solutions, add_special_tokens=False)["input_ids"]
    else:
        response_token_ids = [[] for _ in solutions]

    statuses = []
    reasoning_token_counts = []
    response_token_counts = []
    chunk_ttrs = []
    chunk_ttr_scores_list = []

    for reasoning, reasoning_ids, response_ids in zip(reasonings, reasoning_token_ids, response_token_ids):
        reasoning_tokens = len(reasoning_ids)
        response_tokens = len(response_ids)
        chunk_ttr, chunk_ttr_scores = chunked_ttr(
            reasoning_ids,
            chunk_size=args.ttr_chunk_tokens,
            n=args.ttr_n,
            aggregation=args.ttr_aggregation,
        )

        if not reasoning and not args.allow_missing_reasoning:
            status = "missing_reasoning"
        elif reasoning_tokens > args.max_reasoning_tokens:
            status = "reasoning_too_long"
        elif args.min_response_tokens > 0 and response_tokens <= args.min_response_tokens:
            status = "response_too_short"
        elif chunk_ttr <= args.min_chunk_ttr:
            status = "low_ttr"
        else:
            status = "keep"

        statuses.append(status)
        reasoning_token_counts.append(reasoning_tokens)
        response_token_counts.append(response_tokens)
        chunk_ttrs.append(chunk_ttr)
        chunk_ttr_scores_list.append(chunk_ttr_scores)

    return {
        "_reasoning": reasonings,
        "_solution": solutions,
        "_reasoning_tokens": reasoning_token_counts,
        "_response_tokens": response_token_counts,
        "_chunk_ttr": chunk_ttrs,
        "_chunk_ttr_scores": chunk_ttr_scores_list,
        "_quality_status": statuses,
    }


def convert_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    rows = {
        "conversation_id": [],
        "dataset_source": [],
        "original_metadata": [],
        "created_timestamp": [],
        "system_prompt": [],
        "initial_prompt": [],
        "available_functions": [],
        "conversation_branches": [],
    }

    for i, problem in enumerate(batch["problem"]):
        problem_hash = hashlib.sha256(problem.encode()).hexdigest()[:16]
        expected_answer = batch.get("expected_answer", [None] * len(batch["problem"]))[i]
        reasoning = batch["_reasoning"][i]
        solution = batch["_solution"][i]

        rows["conversation_id"].append(problem_hash)
        rows["dataset_source"].append(SRC)
        rows["original_metadata"].append({
            "problem_source": batch.get("problem_source", [""] * len(batch["problem"]))[i],
            "problem_type": batch.get("problem_type", [""] * len(batch["problem"]))[i],
            "generation_model": batch.get("generation_model", [""] * len(batch["problem"]))[i],
            "pass_rate_72b_tir": parse_pass_rate(batch.get("pass_rate_72b_tir", ["n/a"] * len(batch["problem"]))[i]),
            "inference_mode": batch.get("inference_mode", [""] * len(batch["problem"]))[i],
            "used_in_kaggle": batch.get("used_in_kaggle", [False] * len(batch["problem"]))[i],
            "reasoning_tokens": batch["_reasoning_tokens"][i],
            "response_tokens": batch["_response_tokens"][i],
            "reasoning_chunk_ttr": batch["_chunk_ttr"][i],
            "reasoning_chunk_ttr_scores": batch["_chunk_ttr_scores"][i],
        })
        rows["created_timestamp"].append(datetime.now(UTC).isoformat())
        rows["system_prompt"].append({"content": "", "metadata": {}})
        rows["initial_prompt"].append({
            "role": "user",
            "content": problem,
            "metadata": {},
        })
        rows["available_functions"].append([])
        rows["conversation_branches"].append([
            {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": make_parts(reasoning, solution, expected_answer),
                    }
                ]
            }
        ])

    return rows


def prompt_hash_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
    keys = list(batch.keys())
    if not keys:
        return {"_prompt_hash": []}
    num_rows = len(batch[keys[0]])
    hashes = []
    for i in range(num_rows):
        sample = {key: batch[key][i] for key in keys}
        text = prompt_text(sample)
        hashes.append(prompt_hash(text) if text else "")
    return {"_prompt_hash": hashes}


def convert_sample(
    sample: Dict[str, Any],
    reasoning: str,
    solution: str,
    reasoning_tokens: int,
    chunk_ttr: float,
    chunk_ttr_scores: list[float],
) -> Dict[str, Any]:
    """Convert a single sample to the unified chat format."""
    problem_hash = hashlib.sha256(sample["problem"].encode()).hexdigest()[:16]

    return {
        "conversation_id": problem_hash,
        "dataset_source": SRC,
        "original_metadata": {
            "problem_source": sample.get("problem_source", ""),
            "problem_type": sample.get("problem_type", ""),
            "generation_model": sample.get("generation_model", ""),
            "pass_rate_72b_tir": parse_pass_rate(sample.get("pass_rate_72b_tir", "n/a")),
            "inference_mode": sample.get("inference_mode", ""),
            "used_in_kaggle": sample.get("used_in_kaggle", False),
            "reasoning_tokens": reasoning_tokens,
            "reasoning_chunk_ttr": chunk_ttr,
            "reasoning_chunk_ttr_scores": chunk_ttr_scores,
        },
        "created_timestamp": datetime.now(UTC).isoformat(),
        "system_prompt": {"content": "", "metadata": {}},
        "initial_prompt": {
            "role": "user",
            "content": sample["problem"],
            "metadata": {},
        },
        "available_functions": [],
        "conversation_branches": [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": make_parts(reasoning, solution, sample.get("expected_answer")),
                    }
                ]
            }
        ],
    }


def load_existing_metadata(output_path: Path) -> Optional[Dict[str, Any]]:
    meta_file = output_path / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def save_dataset_and_metadata(
    dataset_dict: DatasetDict,
    output_path: Path,
    args: argparse.Namespace,
    stats: dict[str, int],
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    metadata = load_existing_metadata(output_path) or {}
    processing_entry = {
        "operation": f"convert_{SRC}_with_reasoning",
        "script": "convert-openmathreasoning-with-reasoning.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": args.input,
        "output_path": str(output_path),
        "exclude_prompts_from": args.exclude_prompts_from,
        "num_processes": args.num_proc,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "min_pass_rate": args.min_pass_rate,
        "keep_zero_pass": args.keep_zero_pass,
        "problem_types": args.problem_types,
        "tokenizer_name": args.tokenizer_name,
        "max_reasoning_tokens": args.max_reasoning_tokens,
        "min_response_tokens": args.min_response_tokens,
        "ttr_chunk_tokens": args.ttr_chunk_tokens,
        "ttr_n": args.ttr_n,
        "min_chunk_ttr": args.min_chunk_ttr,
        "ttr_aggregation": args.ttr_aggregation,
        "filter_stats": stats,
        "description": f"Converted {SRC} dataset to unified chat format with reasoning traces",
    }

    metadata.setdefault("processing_log", []).append(processing_entry)
    metadata.setdefault("format", "chat_format_v1")
    metadata.setdefault("source_dataset", SRC)
    metadata.setdefault(
        "conversion_details",
        {
            "conversation_type": "math_reasoning",
            "added_fields": ["system_prompt", "conversation_branches"],
            "edited_fields": [
                "generated_solution (<think> content preserved as thought part)",
                "generated_solution (<think> tags stripped from response part)",
            ],
            "format": "new_chat_format_with_parts",
        },
    )

    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {output_path}", flush=True)
    print(f"Metadata saved to {metadata_file}", flush=True)


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Convert {SRC} to unified chat format with reasoning")
    p.add_argument("-i", "--input", default=None,
                   help="Input JSON file path. If omitted, loads from HuggingFace Hub.")
    p.add_argument("-o", "--output", required=True, help="Output directory path")
    p.add_argument("--exclude-prompts-from", default=None,
                   help=(
                       "Optional dataset path to remove overlapping prompts from before quality filtering. "
                       "This matches openmath.ipynb: prompt hashes are built from problem/initial_prompt only."
                   ))
    p.add_argument("--num-proc", type=int, default=8, help="Number of processes for dataset operations")
    p.add_argument("--batch-size", type=int, default=1000,
                   help="Batch size for parallel tokenization/filtering maps.")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    p.add_argument("--min-pass-rate", type=float, default=None,
                   help=(
                       "Drop solutions with pass_rate_72b_tir below this threshold. "
                       "Solutions with pass_rate='n/a' are kept."
                   ))
    p.add_argument("--keep-zero-pass", action="store_true", default=False,
                   help="Keep solutions with pass_rate=0.0. Filtered by default.")
    p.add_argument("--problem-types", choices=["has_answer", "has_answer_and_proof", "all"],
                   default="all",
                   help=(
                       "Filter by problem_type. 'has_answer' keeps only "
                       "'has_answer_extracted'; 'has_answer_and_proof' keeps "
                       "'has_answer_extracted' + 'converted_proof'; 'all' keeps all."
                   ))
    p.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER,
                   help="Tokenizer used for reasoning length and TTR filters.")
    p.add_argument("--max-reasoning-tokens", type=int, default=4000,
                   help="Drop samples whose reasoning is longer than this many tokenizer tokens.")
    p.add_argument("--min-response-tokens", type=int, default=0,
                   help="Drop samples whose response is this many tokenizer tokens or shorter.")
    p.add_argument("--ttr-chunk-tokens", type=int, default=512,
                   help="Token chunk size for chunked TTR computation.")
    p.add_argument("--ttr-n", type=int, default=2,
                   help="N-gram size for TTR. Defaults to 2 for bigram TTR.")
    p.add_argument("--min-chunk-ttr", type=float, default=0.2,
                   help=(
                       "Drop samples whose aggregated chunked n-gram TTR is at or below this threshold. "
                       "Default is 0.2; tune this after inspecting filter counts."
                   ))
    p.add_argument("--ttr-aggregation", choices=["min", "mean"], default="mean",
                   help="Aggregate per-chunk TTR with min or mean. Defaults to mean, matching openmath.ipynb.")
    p.add_argument("--allow-missing-reasoning", action="store_true", default=False,
                   help="Keep samples without a <think> reasoning trace.")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Overwrite output without prompting.")
    return p.parse_args()


def load_input_dataset(args: argparse.Namespace) -> Dataset:
    if args.input is None:
        return load_dataset("nvidia/OpenMathReasoning", split="cot")

    print(f"Loading data from {args.input}")
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading input file: {e}", flush=True)
        sys.exit(1)
    return Dataset.from_list(raw)


def load_exclusion_dataset(path: str) -> Dataset:
    try:
        loaded = load_from_disk(path)
    except Exception:
        loaded = load_dataset(path)

    if isinstance(loaded, DatasetDict):
        split = "train" if "train" in loaded else next(iter(loaded.keys()))
        return loaded[split]
    return loaded


def main() -> None:
    args = cli()
    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)

    print(f"Tokenizer for quality filters: {args.tokenizer_name}", flush=True)
    print(f"Parallel workers: {args.num_proc}, batch size: {args.batch_size}", flush=True)

    data = load_input_dataset(args)
    print(f"Loaded {len(data)} samples", flush=True)

    if args.limit and args.limit > 0:
        data = data.select(range(min(args.limit, len(data))))
        print(f"Limited to {len(data)} samples", flush=True)

    allowed_problem_types = {
        "has_answer": {"has_answer_extracted"},
        "has_answer_and_proof": {"has_answer_extracted", "converted_proof"},
        "all": None,
    }[args.problem_types]
    if allowed_problem_types is not None:
        print(f"Filtering to problem_types: {sorted(allowed_problem_types)}", flush=True)

    stats = {
        "input_samples": len(data),
        "filtered_existing_prompt": 0,
        "filtered_problem_type": 0,
        "filtered_zero_pass": 0,
        "filtered_low_pass": 0,
        "filtered_missing_reasoning": 0,
        "filtered_reasoning_too_long": 0,
        "filtered_response_too_short": 0,
        "filtered_low_ttr": 0,
        "kept": 0,
    }

    if args.exclude_prompts_from:
        print(f"Excluding prompts already present in {args.exclude_prompts_from}", flush=True)
        exclusion_data = load_exclusion_dataset(args.exclude_prompts_from)
        exclusion_data = exclusion_data.map(
            prompt_hash_batch,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            desc="Hashing exclusion prompts",
        )
        excluded_prompt_hashes = {item for item in exclusion_data["_prompt_hash"] if item}
        print(f"Loaded {len(excluded_prompt_hashes)} exclusion prompt hashes", flush=True)

        before = len(data)
        data = data.map(
            prompt_hash_batch,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            desc="Hashing OpenMath prompts",
        )
        data = data.filter(
            lambda sample: sample["_prompt_hash"] not in excluded_prompt_hashes,
            num_proc=args.num_proc,
            desc="Filtering existing prompts",
        )
        stats["filtered_existing_prompt"] = before - len(data)

    print("Applying cheap metadata filters...", flush=True)
    before = len(data)
    if allowed_problem_types is not None:
        data = data.filter(
            lambda sample: sample.get("problem_type") in allowed_problem_types,
            num_proc=args.num_proc,
            desc="Filtering problem_type",
        )
        stats["filtered_problem_type"] = before - len(data)
        before = len(data)

    if not args.keep_zero_pass:
        data = data.filter(
            lambda sample: parse_pass_rate(sample.get("pass_rate_72b_tir", "n/a")) != 0.0,
            num_proc=args.num_proc,
            desc="Filtering zero pass-rate",
        )
        stats["filtered_zero_pass"] = before - len(data)
        before = len(data)

    if args.min_pass_rate is not None:
        data = data.filter(
            lambda sample: (
                parse_pass_rate(sample.get("pass_rate_72b_tir", "n/a")) is None
                or parse_pass_rate(sample.get("pass_rate_72b_tir", "n/a")) >= args.min_pass_rate
            ),
            num_proc=args.num_proc,
            desc="Filtering low pass-rate",
        )
        stats["filtered_low_pass"] = before - len(data)

    print("Computing reasoning length/TTR quality filters in parallel...", flush=True)
    data = data.map(
        quality_status_for_batch,
        fn_kwargs={"args": args},
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc="Computing reasoning quality",
    )

    status_counts = Counter(data["_quality_status"])
    stats["filtered_missing_reasoning"] = status_counts.get("missing_reasoning", 0)
    stats["filtered_reasoning_too_long"] = status_counts.get("reasoning_too_long", 0)
    stats["filtered_response_too_short"] = status_counts.get("response_too_short", 0)
    stats["filtered_low_ttr"] = status_counts.get("low_ttr", 0)

    data = data.filter(
        lambda sample: sample["_quality_status"] == "keep",
        num_proc=args.num_proc,
        desc="Keeping quality-passing samples",
    )
    stats["kept"] = len(data)

    print("\nFilter summary:", flush=True)
    for key, value in stats.items():
        print(f"  {key}: {value}", flush=True)

    print("Converting kept samples to chat format in parallel...", flush=True)
    converted = data.map(
        convert_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=data.column_names,
        desc="Converting to Apertus chat format",
    )

    print("Creating DatasetDict...", flush=True)
    dataset_dict = DatasetDict({"train": converted})
    save_dataset_and_metadata(dataset_dict, output_path, args, stats)
    print("Conversion complete!", flush=True)


if __name__ == "__main__":
    main()
