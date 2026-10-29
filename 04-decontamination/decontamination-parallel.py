#!/usr/bin/env python3
"""
decontamination-parallel.py
───────────────────────────
Writes a contamination report for each benchmark of a subset (one array job), or of all benchmarks.

For every benchmark, the training prompts (every user message of a sample) are matched against the
benchmark prompts (see matching_units.py for the units and rules) and `<report_path>/<benchmark>__contamination_report.json`
maps each contaminated conversation_id to the matched benchmark prompt. Existing reports are reused.
merge-decontamination-reports.py combines the reports and filters the dataset.

Code modified from: https://github.com/huggingface/cosmopedia/blob/main/decontamination/decontaminate.py
(training samples without any n-gram overlap are skipped before the verification, which runs in
parallel with the benchmark n-grams shared by the worker processes).
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

from benchmark_filters import filter_benchmark_names, report_stem
from conversation_id_checks import validate_conversation_ids
from decontamination_common import (compute_benchmark_ngrams, get_benchmark_cache_key, get_benchmark_cache_path,
                                    get_eval_match_indices, init_contamination_worker, init_get_eval_match_indices,
                                    load_cached_benchmark_ngrams, process_tokens, save_benchmark_ngrams_to_cache,
                                    verify_contamination_worker_optimized)
from matching_units import max_short_train_units, short_prompt_index, short_prompt_match, to_units, user_prompts


def user_prompt_units(batch, tokenizer):
    """One row per user message of the samples in batch: its conversation_id and units."""
    conversation_ids, texts = [], []
    for cid, initial_prompt, branches in zip(batch["conversation_id"], batch["initial_prompt"],
                                             batch["conversation_branches"]):
        for text in user_prompts(initial_prompt, branches):
            conversation_ids.append(cid)
            texts.append(text)
    return {"conversation_id": conversation_ids, "prompt_token_ids": to_units(tokenizer, texts)}


def get_benchmark_subset(all_benchmarks, start_idx=None, end_idx=None):
    """Get a subset of benchmarks for parallel processing."""
    if start_idx is not None and end_idx is not None:
        # Use range-based subsetting
        return all_benchmarks[start_idx:end_idx]
    else:
        # Use all benchmarks (default behavior)
        return all_benchmarks


def main(args):
    print(f"\n=== PARALLEL DECONTAMINATION PIPELINE ===\nLoading decontamination prompts from: {args.decontamination_prompts}")
    eval_data = load_from_disk(args.decontamination_prompts)
    
    # Get complete list of available benchmarks
    all_available_benchmarks, excluded_benchmarks = filter_benchmark_names(list(eval_data.keys()))
    print(f"Total available benchmarks after exclusions: {len(all_available_benchmarks)}")
    if excluded_benchmarks:
        print(f"Excluded {len(excluded_benchmarks)} benchmark(s): {', '.join(sorted(excluded_benchmarks))}")
    
    # Determine benchmark subset to process
    if args.benchmark_name is not None:
        # Legacy behavior: specific benchmarks
        benchmark_list = args.benchmark_name if isinstance(args.benchmark_name, list) else [args.benchmark_name]
        print(f"Using specified benchmark(s): {benchmark_list}")
    elif args.benchmark_list is not None:
        # New: explicit benchmark list
        with open(args.benchmark_list, 'r') as f:
            benchmark_list = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"Using benchmarks from file {args.benchmark_list}: {len(benchmark_list)} benchmarks")
    else:
        # Use range-based subsetting for parallel processing
        benchmark_list = get_benchmark_subset(
            all_available_benchmarks, 
            args.benchmark_start, 
            args.benchmark_end
        )
        if args.benchmark_start is not None or args.benchmark_end is not None:
            print(f"Processing benchmark subset [{args.benchmark_start}:{args.benchmark_end}]: {len(benchmark_list)} benchmarks")
        else:
            print(f"Using all {len(benchmark_list)} available benchmarks")
    benchmark_list, excluded_requested_benchmarks = filter_benchmark_names(benchmark_list)
    if excluded_requested_benchmarks:
        print(f"Excluded {len(excluded_requested_benchmarks)} requested benchmark(s): {', '.join(sorted(excluded_requested_benchmarks))}")
    
    # Validate benchmarks exist
    missing_benchmarks = [b for b in benchmark_list if b not in all_available_benchmarks]
    if missing_benchmarks:
        print(f"Warning: {len(missing_benchmarks)} benchmarks not found in dataset:")
        for missing in missing_benchmarks[:5]:  # Show first 5
            print(f"  - {missing}")
        if len(missing_benchmarks) > 5:
            print(f"  ... and {len(missing_benchmarks) - 5} more")
        # Filter out missing benchmarks
        benchmark_list = [b for b in benchmark_list if b in all_available_benchmarks]
    
    print(f"Final benchmark list: {len(benchmark_list)} benchmarks to process\n")

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
    print("Tokenizing training prompts (every user message)...")
    num_samples = len(train_data)
    train_data = train_data.map(
        partial(user_prompt_units, tokenizer=tokenizer),
        batched=True,
        remove_columns=train_data.column_names,
    )
    # One entry per user message; a match of any of them reports the sample's conversation_id
    train_data_prompts_token_ids = train_data["prompt_token_ids"]
    train_conversation_ids = train_data["conversation_id"]
    print(f"{len(train_conversation_ids)} user messages in {num_samples} samples")
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
                      f"({total_processed*100//len(train_data_prompts_token_ids)}%) - "
                      f"{rate:.0f} prompts/sec - ETA: {int(eta)}s", end='\r')
        print()  # New line after progress
    
    elapsed = time.time() - start_time
    print(f"Training data preprocessing complete in {int(elapsed/60)}m {elapsed%60:.1f}s")
    # Only training prompts this short can match a benchmark prompt shorter than the n-gram length
    short_train_samples = [(idx, sample["tokens"]) for idx, sample in enumerate(train_ngrams)
                           if 0 < len(sample["tokens"]) <= max_short_train_units(args.ngram_length)]
    print(f"{len(short_train_samples)} training prompts are short enough to match a short benchmark prompt")
    gc.collect()

    # Iterate over benchmarks for decontamination.
    processed_benchmarks = []
    for i, eval_dataset_name in enumerate(benchmark_list, 1):
        start_time = time.time()
        print(f"\n[{i}/{len(benchmark_list)}] Processing benchmark: {eval_dataset_name}")

        # Load and process evaluation prompts
        output_path = os.path.join(
            args.report_path,
            report_stem(eval_dataset_name) + "__contamination_report.json"
        )  # Reports are saved inside the input dataset's directory
        if os.path.exists(output_path) and not args.overwrite:
            print("contamination_report already exists, skipping decontamination")
            processed_benchmarks.append(eval_dataset_name)  # count existing reports as processed
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
        print(f"  Pre-filtering complete: {len(train_samples_with_overlap)}/{len(train_ngrams)} samples have overlap ({overlap_ratio:.1f}%) - {prefilter_time:.1f}s")
        
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
        print(f"  Short benchmark prompts {counts}: {short_found} more contaminated samples")

        del eval_ngrams, train_idx_match_indices, short_index
        gc.collect()

        # Step 4: Save report
        step_start = time.time()
        json.dump(contamination_mapping, open(output_path, "w"), separators=(",", ":"))
        # The merge sums these into short_benchmark_prompts of the processing log
        with open(output_path.replace("__contamination_report.json", "__short_prompts.json"), "w") as f:
            json.dump({**counts, "base_report_path": args.base_report_path}, f)
        step4_time = time.time() - step_start
        print(f"  Step 4 (Save report): {step4_time:.1f}s")
        
        # Total time summary
        running_time = time.time() - start_time
        print(f"  Total benchmark time: {int(running_time/60)}m {running_time%60:.1f}s")
        print(f"    - Load/compute n-grams: {step1_time:.1f}s ({step1_time/running_time*100:.1f}%)")
        print(f"    - Find n-gram matches: {step2_time:.1f}s ({step2_time/running_time*100:.1f}%)")
        print(f"      * Pre-filtering: {prefilter_time:.1f}s")
        print(f"    - Verify contamination: {step3_time:.1f}s ({step3_time/running_time*100:.1f}%)")
        print(f"    - Save report: {step4_time:.1f}s ({step4_time/running_time*100:.1f}%)")
        
        processed_benchmarks.append(eval_dataset_name)
        del contamination_mapping
        gc.collect()

    # Parallel processing complete - contamination reports saved
    print(f"\n=== PARALLEL PROCESSING COMPLETE ===")
    print(f"Processed {len(processed_benchmarks)} benchmarks in this job")
    if processed_benchmarks:
        print(f"Benchmarks processed:")
        for eval_dataset_name in processed_benchmarks:
            report_path = os.path.join(
                args.report_path,
                report_stem(eval_dataset_name) + "__contamination_report.json"
            )
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    report = json.load(f)
                print(f"  - {eval_dataset_name}: {len(report)} contaminated samples")
            else:
                print(f"  - {eval_dataset_name}: report missing")
    
    print(f"\nContamination reports saved to: {args.report_path}")
    print("Use merge-decontamination-reports.py to combine all reports and filter the dataset.")
    
    del eval_data
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate decontamination reports for a benchmark subset (parallel processing version)."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path of the training dataset (chat format of 02-standardisation, a Dataset or DatasetDict). "
             "Every user message is checked: initial_prompt and the user turns of conversation_branches.",
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
        default="swiss-ai/Apertus-8B-Instruct-2509",
        help="Name of the tokenizer to use for decontamination. "
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default=None,
        help="Name of the benchmark samples to use for decontamination. Use either a single benchmark's name or a list with comma separation."
    )
    parser.add_argument(
        "--benchmark-start",
        type=int,
        default=None,
        help="Start index for benchmark subset (for parallel processing)"
    )
    parser.add_argument(
        "--benchmark-end", 
        type=int,
        default=None,
        help="End index for benchmark subset (for parallel processing)"
    )
    parser.add_argument(
        "--benchmark-list",
        type=str,
        default=None,
        help="File containing list of benchmarks to process (one per line, # for comments)"
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
