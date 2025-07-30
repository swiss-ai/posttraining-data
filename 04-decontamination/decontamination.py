import argparse
import difflib
from tqdm.auto import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer
from multiprocessing import Pool
from collections import defaultdict
from functools import partial
import json
import os
import time
import gc
import os

"""
Code modified from: https://github.com/huggingface/cosmopedia/blob/main/decontamination/decontaminate.py
"""
def get_ngrams(tokens, n):
    """Generate n-grams from tokens."""
    return set(zip(*[tokens[i:-(n-i)] for i in range(n)]))

def process_tokens(x, ngram_length):
    return {
        "ngram": get_ngrams(x, ngram_length),
        "tokens": x
    }

shared_eval_ngram_to_eval_idx = None
shared_eval_ngram_to_eval_idx_key_set = None
def init_get_eval_match_indices(ngram_to_eval_idx, ):
    global shared_eval_ngram_to_eval_idx, shared_eval_ngram_to_eval_idx_key_set
    shared_eval_ngram_to_eval_idx = ngram_to_eval_idx
    shared_eval_ngram_to_eval_idx_key_set = set(ngram_to_eval_idx.keys())

def get_eval_match_indices(train_sample):
    global shared_eval_ngram_to_eval_idx, shared_eval_ngram_to_eval_idx_key_arr
    shared_ngrams = train_sample["ngram"] & shared_eval_ngram_to_eval_idx_key_set
    if len(shared_ngrams) == 0:
        return tuple([])
    matches = []
    for ngram in shared_ngrams:
        matches.extend(shared_eval_ngram_to_eval_idx.get(ngram))
    return tuple(set(matches))


shared_train_ngrams = None
shared_eval_ngrams = None
def init_check_matching(train_ngrams, eval_ngrams):
    global shared_train_ngrams, shared_eval_ngrams
    shared_train_ngrams = train_ngrams
    shared_eval_ngrams = eval_ngrams

def check_matching(train_idx, eval_indices):
    global shared_train_ngrams, shared_eval_ngrams
    # train_idx, eval_indices = inputs
    eval_idx_match = None
    for eval_idx in eval_indices:
        matcher = difflib.SequenceMatcher(
            None,
            shared_train_ngrams[train_idx]["tokens"],
            shared_eval_ngrams[eval_idx],
            autojunk=False
        )
        matching_blocks = matcher.get_matching_blocks()
        match_length = sum([x.size if x.size >= 5 else 0 for x in matching_blocks])
        del matcher, matching_blocks
        if match_length / len(shared_eval_ngrams[eval_idx]["tokens"]) >= args.diff_threshold:
            eval_idx_match = eval_idx
            break
    return train_idx, eval_idx_match

def main(args):
    eval_data = load_from_disk(args.decontamination_prompts)
    # Get list of benchmarks to use for decontamination
    if args.benchmark_name is None:
        benchmark_list = list(eval_data.keys())
    else:
        benchmark_list = args.benchmark_name if isinstance(args.benchmark_name, list) else [args.benchmark_name]
    print("Benchmarks to consider: ", benchmark_list)

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
    if not os.path.exists(args.report_path):
        print("Creating directory for contamination reports at: ", args.report_path)
        os.mkdir(args.report_path)
    print(f"Training data loaded with {len(train_data)} samples")
    train_data = train_data.map(
        lambda batch: {
            "prompt_token_ids": tokenizer([x["content"] for x in batch["initial_prompt"]])["input_ids"]
        },
        batched=True,
    )
    train_data_prompts_token_ids = train_data["prompt_token_ids"]
    train_conversation_ids = train_data["conversation_id"]
    del train_data  # Remove the training data for memory free-up
    print("Prompts tokenized")
    with Pool(args.num_proc) as p:
        train_ngrams = p.map(partial(process_tokens, ngram_length=args.ngram_length), train_data_prompts_token_ids)  # train_ngrams: List[{"ngram": List[set], "tokens": List[int]}]
    print("Training ngrams converted")
    gc.collect()

    # Iterate over benchmarks for decontamination.
    for eval_dataset_name in benchmark_list:
        start_time = time.time()
        print("Running decontamination for benchmark: ", eval_dataset_name)

        # Load and process evaluation prompts
        output_path = os.path.join(
            args.report_path,
            eval_dataset_name.replace("/", "_") + "__contamination_report.json"
        )  # Reports are saved inside the input dataset's directory
        if os.path.exists(output_path) and not args.overwrite:
            print("contamination_report already exists, skipping decontamination")
            continue
        eval_dataset = eval_data[eval_dataset_name]
        eval_data_tokens = tokenizer(eval_dataset["prompt"][:])["input_ids"]  # List of lists with token ids, List[List[int]]
        with Pool(args.num_proc) as p:
            eval_ngrams = p.map(partial(process_tokens, ngram_length=args.ngram_length), eval_data_tokens)  # eval_ngrams[dataset_name]: List[{"ngram": List[set], "tokens": List[int]}]
        del eval_data_tokens, eval_dataset
        print("Eval dataset ngram converted")

        eval_ngram_to_eval_idx = defaultdict(list)  # Lookup dictionary: {ngram: (benchmark indices of appearance)}
        append = eval_ngram_to_eval_idx.__getitem__
        for idx, s in enumerate(eval_ngrams):
            for element in s["ngram"]:
                append(element).append(idx)
        print("Number of unique ngrams in eval_ngram_to_eval_idx: ", len(eval_ngram_to_eval_idx))

        # Calculate ngram matches between the training and evaluation prompts
        with Pool(args.num_proc, initializer=init_get_eval_match_indices, initargs=(eval_ngram_to_eval_idx,)) as p:
            ngram_match_idx_map = p.map(
                get_eval_match_indices,
                train_ngrams
            )  # List[Tuples]
        train_idx_match_indices = {idx: x for idx, x in enumerate(ngram_match_idx_map) if len(x) > 0}  # Dict[idx, Tuple[int]]
        del ngram_match_idx_map, eval_ngram_to_eval_idx
        print("train_idx_match_indices calculated")

        contamination_mapping = {}
        for train_idx, eval_indices in tqdm(train_idx_match_indices.items()):
            for eval_idx in eval_indices:
                matcher = difflib.SequenceMatcher(None, train_ngrams[train_idx]["tokens"], eval_ngrams[eval_idx]["tokens"], autojunk=False)
                matching_blocks = matcher.get_matching_blocks()
                match_length = sum([x.size if x.size >= 5 else 0 for x in matching_blocks])
                del matcher, matching_blocks
                if match_length / len(eval_ngrams[eval_idx]["tokens"]) >= args.diff_threshold:
                    train_sample_id = train_conversation_ids[train_idx]
                    contamination_mapping[train_sample_id] = eval_idx
                    break
        print("Contamination mapping done with number of contamination mapping: ", len(contamination_mapping))

        del eval_ngrams, train_idx_match_indices
        gc.collect()

        # Save
        json.dump(contamination_mapping, open(output_path, "w"), separators=(",", ":"))
        running_time = time.time() - start_time
        print(f"Benchmark decontamination finished in {int(running_time/60)} minutes {running_time%60} seconds.")
        del contamination_mapping
        gc.collect()

    # Load the contamination reports and filter the training data
    benchmark_list = list(eval_data.keys())
    del eval_data, tokenizer
    gc.collect()
    train_data = load_from_disk(args.dataset_path)
    contaminated_ids = set()
    for eval_dataset_name in benchmark_list:
        report_path = os.path.join(
            args.report_path,
            eval_dataset_name.replace("/", "_") + "__contamination_report.json"
        )
        with open(report_path, "r") as f:
            report = json.load(f)
        contaminated_ids = contaminated_ids.union(set(report.keys()))
    print("Number of contaminated ids: ", len(contaminated_ids))
    # Original single dataset filtering (commented out for DatasetDict compatibility):
    # train_data = train_data.filter(lambda x: x["conversation_id"] not in contaminated_ids)
    # train_data.save_to_disk(args.output)
    
    # Handle DatasetDict format - filter all splits if DatasetDict, otherwise filter single dataset
    if hasattr(train_data, 'keys'):  # DatasetDict
        from datasets import DatasetDict
        train_data = DatasetDict({k: v.filter(lambda x: x["conversation_id"] not in contaminated_ids) 
                                 for k, v in train_data.items()})
    else:  # Single Dataset
        train_data = train_data.filter(lambda x: x["conversation_id"] not in contaminated_ids)
    train_data.save_to_disk(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a decontamination report for a dataset."
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
        help="Whether to overwrite the output file if it exists.",
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

    args = parser.parse_args()
    main(args)
