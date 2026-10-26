#!/usr/bin/env python3
"""
decontamination.py
──────────────────
Dataset decontamination tool for removing evaluation benchmark contamination.

This script identifies and removes training samples that overlap with evaluation
benchmarks to prevent data leakage. Uses n-gram overlap detection and exact
matching to find contaminated samples across multiple benchmark datasets.

Key features:
- N-gram overlap detection with configurable thresholds
- Support for multiple benchmark formats
- Memory-efficient processing for large datasets
- Detailed contamination reporting and statistics
- Preserves dataset metadata and processing history
"""
import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer
from multiprocessing import Pool
from functools import partial
import json
import os
import time
import gc
from pathlib import Path
from datetime import datetime

from benchmark_filters import filter_benchmark_names
from conversation_id_checks import validate_conversation_ids
from decontamination_common import (compute_benchmark_ngrams, get_benchmark_cache_key, get_benchmark_cache_path,
                                    get_eval_match_indices, init_contamination_worker, init_get_eval_match_indices,
                                    load_cached_benchmark_ngrams, process_tokens, save_benchmark_ngrams_to_cache,
                                    verify_contamination_worker_optimized, load_existing_metadata)
from matching_units import (MATCHING_VERSION, max_short_train_units, short_prompt_index, short_prompt_match,
                            to_units)
from removal_log import write_removal_log

"""
Code modified from: https://github.com/huggingface/cosmopedia/blob/main/decontamination/decontaminate.py
Enhanced with pre-filtering and parallelization optimizations.
V2: Optimizes n-gram matching by pre-filtering training samples that have no n-gram overlap.
V3: Parallelizes Step 3 (contamination verification) using shared memory to avoid serialization overhead.
"""


def save_dataset_and_metadata(train_data, output_path: Path, input_path: Path,
                              contaminated_ids: set, processed_benchmarks: list,
                              tokenizer_name: str, ngram_length: int, diff_threshold: float,
                              samples_removed: int, removal_summary: dict):
    """Save filtered dataset and update metadata with processing log."""
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save dataset
    print(f"Saving filtered dataset to {output_path}...")
    train_data.save_to_disk(str(output_path))

    # Load or create metadata
    metadata = load_existing_metadata(input_path) or {}

    # Calculate samples removed per split if DatasetDict
    samples_by_split = {}
    total_samples_before = 0
    total_samples_after = 0

    if hasattr(train_data, 'keys'):  # DatasetDict
        for split_name, split_data in train_data.items():
            samples_by_split[split_name] = len(split_data)
            total_samples_after += len(split_data)
    else:  # Single Dataset
        total_samples_after = len(train_data)

    # Add processing log entry
    processing_entry = {
        "operation": "decontamination",
        "script": "decontamination.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "tokenizer_name": tokenizer_name,
        "ngram_length": ngram_length,
        "matching_version": MATCHING_VERSION,
        "diff_threshold": diff_threshold,
        "benchmarks_processed": len(processed_benchmarks),
        "benchmark_names": processed_benchmarks,
        "contaminated_ids_flagged": len(contaminated_ids),
        "contaminated_samples_removed": samples_removed,
        "samples_after_filtering": total_samples_after,
        **removal_summary,
        "decontamination_success": True
    }

    # Add split information if DatasetDict
    if samples_by_split:
        processing_entry["samples_by_split"] = samples_by_split

    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)

    # Save updated metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")


def main(args):
    print(f"\n=== DECONTAMINATION PIPELINE ===\nLoading decontamination prompts from: {args.decontamination_prompts}")
    eval_data = load_from_disk(args.decontamination_prompts)
    # Get list of benchmarks to use for decontamination
    if args.benchmark_name is None:
        benchmark_list = list(eval_data.keys())
        print(f"Using all {len(benchmark_list)} available benchmarks")
    else:
        benchmark_list = args.benchmark_name if isinstance(args.benchmark_name, list) else [args.benchmark_name]
        print(f"Using specified benchmark(s): {benchmark_list}")
    benchmark_list, excluded_benchmarks = filter_benchmark_names(benchmark_list)
    if excluded_benchmarks:
        print(f"Excluded {len(excluded_benchmarks)} benchmark(s): {', '.join(sorted(excluded_benchmarks))}")
    print(f"Total benchmarks to process: {len(benchmark_list)}\n")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    # Load and preprocess training data
    train_data = load_from_disk(args.dataset_path)
    # Original single split extraction (commented out for processing all splits):
    # if args.train_dataset_split:
    #     train_data = train_data[args.train_dataset_split]

    # Process all splits in DatasetDict for complete decontamination
    if hasattr(train_data, 'keys'):  # DatasetDict
        # Concatenate all splits for contamination detection
        all_splits_data = []
        for split_name in train_data.keys():
            all_splits_data.append(train_data[split_name])
        if len(all_splits_data) == 1:
            train_data = all_splits_data[0]
        else:
            from datasets import concatenate_datasets
            train_data = concatenate_datasets(all_splits_data)
    # If single Dataset, use as-is
    validate_conversation_ids(train_data["conversation_id"], args.dataset_path)
    if not os.path.exists(args.report_path):
        print(f"Creating contamination reports directory: {args.report_path}")
        os.makedirs(args.report_path, exist_ok=True)
    print(f"Training data loaded: {len(train_data)} samples")
    print("Tokenizing training prompts...")
    train_data = train_data.map(
        lambda batch: {
            "prompt_token_ids": to_units(tokenizer, [x["content"] or "" for x in batch["initial_prompt"]])
        },
        batched=True,
    )
    train_data_prompts_token_ids = train_data["prompt_token_ids"]
    train_conversation_ids = train_data["conversation_id"]
    del train_data  # Remove the training data for memory free-up
    print(f"Generating n-grams from training data ({len(train_data_prompts_token_ids)} prompts)...")
    start_time = time.time()

    with Pool(args.num_proc) as p:
        train_ngrams = []
        total_processed = 0
        # Use imap for progress tracking
        for result in p.imap(partial(process_tokens, ngram_length=args.ngram_length),
                             train_data_prompts_token_ids,
                             chunksize=1000):
            train_ngrams.append(result)
            total_processed += 1
            if total_processed % 1000 == 0 or total_processed == len(train_data_prompts_token_ids):
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                eta = (len(train_data_prompts_token_ids) - total_processed) / rate if rate > 0 else 0
                print(f"  Processed {total_processed}/{len(train_data_prompts_token_ids)} prompts "
                      f"({total_processed * 100 // len(train_data_prompts_token_ids)}%) - "
                      f"{rate:.0f} prompts/sec - ETA: {int(eta)}s", end='\r')
        print()  # New line after progress

    elapsed = time.time() - start_time
    print(f"Training data preprocessing complete in {int(elapsed / 60)}m {elapsed % 60:.1f}s")
    # Only training prompts this short can match a benchmark prompt shorter than the n-gram length
    short_train_samples = [(idx, sample["tokens"]) for idx, sample in enumerate(train_ngrams)
                           if 0 < len(sample["tokens"]) <= max_short_train_units(args.ngram_length)]
    print(f"{len(short_train_samples)} training prompts are short enough to match a short benchmark prompt")
    short_prompt_counts = {"empty": 0, "exact_only": 0, "contained": 0, "benchmarks_counted": 0}
    gc.collect()

    # Iterate over benchmarks for decontamination.
    processed_benchmarks = []
    for i, eval_dataset_name in enumerate(benchmark_list, 1):
        start_time = time.time()
        print(f"\n[{i}/{len(benchmark_list)}] Processing benchmark: {eval_dataset_name}")

        # Load and process evaluation prompts
        output_path = os.path.join(
            args.report_path,
            eval_dataset_name.replace("/", "_") + "__contamination_report.json"
        )  # Reports are saved inside the input dataset's directory
        if os.path.exists(output_path) and not args.overwrite:
            print("contamination_report already exists, skipping decontamination")
            processed_benchmarks.append(eval_dataset_name)  # still apply the existing report in the final filter
            continue
        # Step 1: Load/compute benchmark n-grams
        step_start = time.time()
        cache_key = get_benchmark_cache_key(eval_dataset_name, args.tokenizer_name, args.ngram_length)
        cache_path = get_benchmark_cache_path(args.cache_dir, cache_key)

        cached_data = load_cached_benchmark_ngrams(cache_path)
        if cached_data is not None:
            print(f"  Loading cached n-grams from {cache_path}")
            eval_ngrams = cached_data['ngrams']
            eval_ngram_to_eval_idx = cached_data['lookup_table']
            print(f"  Loaded {len(eval_ngrams)} cached n-grams with {len(eval_ngram_to_eval_idx)} unique entries")
        else:
            # Compute n-grams and cache them
            eval_dataset = eval_data[eval_dataset_name]
            eval_ngrams, eval_ngram_to_eval_idx = compute_benchmark_ngrams(
                eval_dataset, tokenizer, args.ngram_length, args.num_proc
            )
            save_benchmark_ngrams_to_cache(eval_ngrams, eval_ngram_to_eval_idx, cache_path)
            del eval_dataset

        step1_time = time.time() - step_start
        print(f"  Step 1 (Load/compute n-grams): {step1_time:.1f}s")

        if args.base_report_path:  # take the n-gram matches from an earlier run: steps 2-3 find nothing
            with open(os.path.join(args.base_report_path, os.path.basename(output_path))) as f:
                base_report = json.load(f)
            eval_ngram_to_eval_idx = {}

        # Step 2: Find n-gram matches (with pre-filtering optimization)
        step_start = time.time()
        print(f"  Finding n-gram matches with training data...")

        # Pre-filtering: Only process training samples with ANY n-gram overlap
        benchmark_ngrams_set = set(eval_ngram_to_eval_idx.keys())
        train_samples_with_overlap = []
        train_indices_with_overlap = []

        print(f"  Pre-filtering training samples for n-gram overlap...")
        prefilter_start = time.time()
        for idx, train_sample in enumerate(train_ngrams):
            if train_sample["ngram"] & benchmark_ngrams_set:  # Fast set intersection
                train_samples_with_overlap.append(train_sample)
                train_indices_with_overlap.append(idx)

        prefilter_time = time.time() - prefilter_start
        overlap_ratio = len(train_samples_with_overlap) / len(train_ngrams) * 100
        print(
            f"  Pre-filtering complete: {len(train_samples_with_overlap)}/{len(train_ngrams)} samples have overlap ({overlap_ratio:.1f}%) - {prefilter_time:.1f}s")

        # Only process samples with potential matches
        if len(train_samples_with_overlap) > 0:
            with Pool(args.num_proc, initializer=init_get_eval_match_indices, initargs=(eval_ngram_to_eval_idx,)) as p:
                ngram_match_results = p.map(get_eval_match_indices, train_samples_with_overlap)

            # Map back to original indices
            train_idx_match_indices = {}
            for local_idx, matches in enumerate(ngram_match_results):
                if len(matches) > 0:
                    original_idx = train_indices_with_overlap[local_idx]
                    train_idx_match_indices[original_idx] = matches
            del ngram_match_results
        else:
            train_idx_match_indices = {}

        del train_samples_with_overlap, train_indices_with_overlap, benchmark_ngrams_set, eval_ngram_to_eval_idx
        step2_time = time.time() - step_start
        print(f"  Found potential matches for {len(train_idx_match_indices)} training samples")
        print(f"  Step 2 (Find n-gram matches): {step2_time:.1f}s")

        # Step 3: Verify contamination with sequence matching (parallelized with shared memory)
        step_start = time.time()
        print(f"  Checking {len(train_idx_match_indices)} potential matches against threshold (parallelized)...")

        # Prepare optimized tasks for parallel processing (no eval_ngrams in each task)
        verification_tasks = []
        for train_idx, eval_indices in train_idx_match_indices.items():
            verification_tasks.append((
                train_idx,
                eval_indices,
                train_ngrams[train_idx]["tokens"],
                args.diff_threshold,
                train_conversation_ids[train_idx]
            ))

        # Execute parallel contamination verification with shared memory
        contamination_mapping = {}
        contaminated_samples_shown = 0

        if len(verification_tasks) > 0:
            # Use shared memory initialization to avoid serializing eval_ngrams to each worker
            with Pool(args.num_proc, initializer=init_contamination_worker, initargs=(eval_ngrams,)) as p:
                results = p.map(verify_contamination_worker_optimized, verification_tasks)

            # Process results
            for result in results:
                if result is not None:
                    train_conversation_id, eval_idx, match_ratio, train_tokens_full, eval_tokens_full = result
                    contamination_mapping[train_conversation_id] = eval_idx

                    # Show contaminated samples if requested
                    if args.show_contaminated and contaminated_samples_shown < 10:
                        print(f"\n  [CONTAMINATED] Sample {train_conversation_id}")
                        print(f"    Training prompt: {tokenizer.decode(train_tokens_full)}")
                        print(f"    Benchmark prompt: {tokenizer.decode(eval_tokens_full)}")
                        print(f"    Match ratio: {match_ratio:.2f}")
                        contaminated_samples_shown += 1
        step3_time = time.time() - step_start
        print(f"  Found {len(contamination_mapping)} contaminated samples")
        print(f"  Step 3 (Verify contamination): {step3_time:.1f}s")

        if args.base_report_path:
            contamination_mapping.update(base_report)
            print(f"  {len(base_report)} contaminated samples from the base report")

        # Step 3b: benchmark prompts shorter than the n-gram length have no n-gram (see matching_units.py)
        short_index, counts = short_prompt_index([x["tokens"] for x in eval_ngrams], args.ngram_length)
        lengths = sorted({len(k) for k in short_index})
        short_found = 0
        for train_idx, train_tokens in short_train_samples:
            train_conversation_id = train_conversation_ids[train_idx]
            if train_conversation_id not in contamination_mapping:
                eval_idx = short_prompt_match(train_tokens, short_index, lengths)
                if eval_idx is not None:
                    contamination_mapping[train_conversation_id] = eval_idx
                    short_found += 1
        for k, v in counts.items():
            short_prompt_counts[k] += v
        short_prompt_counts["benchmarks_counted"] += 1
        print(f"  Short benchmark prompts {counts}: {short_found} more contaminated samples")

        del eval_ngrams, train_idx_match_indices, short_index
        gc.collect()

        # Step 4: Save report
        step_start = time.time()
        json.dump(contamination_mapping, open(output_path, "w"), separators=(",", ":"))
        step4_time = time.time() - step_start
        print(f"  Step 4 (Save report): {step4_time:.1f}s")

        # Total time summary
        running_time = time.time() - start_time
        print(f"  Total benchmark time: {int(running_time / 60)}m {running_time % 60:.1f}s")
        print(f"    - Load/compute n-grams: {step1_time:.1f}s ({step1_time / running_time * 100:.1f}%)")
        print(f"    - Find n-gram matches: {step2_time:.1f}s ({step2_time / running_time * 100:.1f}%)")
        print(f"      * Pre-filtering: {prefilter_time:.1f}s")
        print(f"    - Verify contamination: {step3_time:.1f}s ({step3_time / running_time * 100:.1f}%)")
        print(f"    - Save report: {step4_time:.1f}s ({step4_time / running_time * 100:.1f}%)")

        processed_benchmarks.append(eval_dataset_name)
        del contamination_mapping
        gc.collect()

    # Load the contamination reports and filter the training data
    print(f"\n=== FINAL FILTERING ===\nCombining contamination reports from {len(processed_benchmarks)} processed benchmarks...")
    gc.collect()
    train_data = load_from_disk(args.dataset_path)
    contaminated_ids = set()
    contamination_summary = []
    reports = {}
    for eval_dataset_name in processed_benchmarks:
        report_path = os.path.join(
            args.report_path,
            eval_dataset_name.replace("/", "_") + "__contamination_report.json"
        )
        with open(report_path, "r") as f:
            report = json.load(f)
        reports[eval_dataset_name] = report
        contaminated_ids = contaminated_ids.union(set(report.keys()))
        if len(report) > 0:
            contamination_summary.append((eval_dataset_name, len(report)))
        print(f"  - {eval_dataset_name}: {len(report)} contaminated samples")

    print(f"\n=== CONTAMINATION SUMMARY ===")
    print(f"Total contaminated samples to remove: {len(contaminated_ids)}")
    if contamination_summary:
        print(f"\nBenchmarks with contamination:")
        contamination_summary.sort(key=lambda x: x[1], reverse=True)
        for benchmark, count in contamination_summary[:20]:  # Show top 20
            print(f"  {benchmark}: {count} samples")
        if len(contamination_summary) > 20:
            print(f"  ... and {len(contamination_summary) - 20} more benchmarks")
    # Original single dataset filtering (commented out for DatasetDict compatibility):
    # train_data = train_data.filter(lambda x: x["conversation_id"] not in contaminated_ids)
    # train_data.save_to_disk(args.output)

    # Handle DatasetDict format - filter all splits if DatasetDict, otherwise filter single dataset
    # Report keys are JSON strings, so compare IDs as strings.
    unfiltered_data = train_data
    if hasattr(train_data, 'keys'):  # DatasetDict
        from datasets import DatasetDict
        samples_before = sum(len(v) for v in train_data.values())
        train_data = DatasetDict({k: v.filter(lambda x: str(x["conversation_id"]) not in contaminated_ids)
                                  for k, v in train_data.items()})
        samples_after = sum(len(v) for v in train_data.values())
    else:  # Single Dataset
        samples_before = len(train_data)
        train_data = train_data.filter(lambda x: str(x["conversation_id"]) not in contaminated_ids)
        samples_after = len(train_data)
    samples_removed = samples_before - samples_after
    print(f"Removed {samples_removed} samples ({len(contaminated_ids)} contaminated IDs flagged)")
    if samples_removed != len(contaminated_ids):
        raise ValueError(
            f"Removed {samples_removed} samples but {len(contaminated_ids)} IDs were flagged: the reports in "
            f"{args.report_path} do not match the conversation_ids of {args.dataset_path}. "
            f"Not saving the filtered dataset."
        )

    # Log the removed samples with their matches
    removal_summary = write_removal_log(unfiltered_data, reports, eval_data, args.tokenizer_name, Path(args.output),
                                        args.diff_threshold, args.num_proc)
    del unfiltered_data, eval_data, reports
    removal_summary["short_benchmark_prompts"] = short_prompt_counts
    removal_summary["base_report_path"] = args.base_report_path

    # Save dataset with metadata
    save_dataset_and_metadata(
        train_data,
        Path(args.output),
        Path(args.dataset_path),
        contaminated_ids,
        processed_benchmarks,
        args.tokenizer_name,
        args.ngram_length,
        args.diff_threshold,
        samples_removed,
        removal_summary
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a decontamination report for a dataset with optimized pre-filtering and parallel verification."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path or name of the training dataset to process. Must include a column with 'messages' that "
             "contains the standard HuggingFace chat format of List[Dict[str, str]]",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for filtered dataset",
    )
    parser.add_argument(
        "--decontamination_prompts",
        type=str,
        default=None,
        help="Name of the dataset with benchmark samples to use for decontamination. "
             "It expects a DatasetDict with multiple benchmark samples init, each with columns:"
             " config_name, split_name, prompts",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="alehc/swissai-tokenizer",
        help="Name of the tokenizer to use for decontamination. "
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default=None,
        help="Name of the benchmark samples to use for decontamination. Use either a single benchmark's name or a list with comma separation."
    )
    parser.add_argument(
        "--report_path",
        type=str,
        required=True,
        help="Path for the output JSON with decontamination report.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate contamination reports even if they already exist. Without this flag, existing reports are skipped.",
    )
    parser.add_argument(
        "--base_report_path",
        type=str,
        default=None,
        help="Reports of an earlier run with the same units (UNITS_VERSION) but without the short "
             "benchmark prompt check: their n-gram matches are used and only that check runs.",
    )
    parser.add_argument(
        "--ngram_length",
        type=int,
        default=8,
        help="Length of the n-grams to consider.",
    )
    parser.add_argument(
        "--diff_threshold",
        type=float,
        default=0.5,
        help="Threshold for filtering based on difference ratio.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=16,
        help="Number of processes to use for map operations.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./benchmark_cache",
        help="Directory to store cached benchmark n-grams (default: ./benchmark_cache).",
    )
    parser.add_argument(
        "--show_contaminated",
        action="store_true",
        help="Display examples of contaminated samples during processing (shows first 10 per benchmark).",
    )

    args = parser.parse_args()
    main(args)
