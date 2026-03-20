"""
Stage 1: Filter a preference dataset by maximum sequence length.

Tokenizes both 'chosen' and 'rejected' columns and removes rows
where either exceeds max_seq_len tokens.

Input:  HuggingFace dataset with 'chosen' and 'rejected' columns
        (list of message dicts: [{role, content}, ...])
Output: Filtered dataset saved to disk (same schema, fewer rows).

Usage:
    python filter_by_length.py \
        --dataset-path /path/to/dataset \
        --output-dir /path/to/output \
        --model-name-or-path /path/to/model \
        --max-seq-len 4096

Example:  python 1-filter-dataset/filter_by_length.py --dataset-path /iopsstor/scratch/cscs/dmelikidze/posttraining-data/preference_acquisition/datasets/MaxMin \
    --output-dir=./datasets/MaxMin-Filtered \
    --model-name-or-path=/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364
    
python 1-filter-dataset/filter_by_length.py --dataset-path /iopsstor/scratch/cscs/dmelikidze/posttraining-data/preference_acquisition/datasets/Qwen3-32B_vs_0.6B \
    --output-dir=./datasets/DeltaQwen-Filtered \
    --model-name-or-path=/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364
"""

import argparse
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer


def count_tokens(messages, tokenizer):
    """Tokenize a conversation and return the token count."""
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    return len(token_ids)


def main(args):
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    # If loaded as a DatasetDict, extract the train_split
    if hasattr(dataset, "keys"):
        split = "train_split" if "train_split" in dataset else list(dataset.keys())[0]
        print(f"DatasetDict detected, using '{split}' split")
        dataset = dataset[split]

    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    original_size = len(dataset)
    print(f"Original dataset size: {original_size}")

    def keep_role_content(example):
        example["chosen"] = [{"role": m["role"], "content": m["content"]} for m in example["chosen"]]
        example["rejected"] = [{"role": m["role"], "content": m["content"]} for m in example["rejected"]]
        return example

    dataset = dataset.map(keep_role_content, num_proc=args.num_proc, desc="Stripping extra message keys")

    cols_to_keep = {"chosen", "rejected", "chosen_score", "rejected_score", "chosen_model", "rejected_model"}
    cols_to_remove = [c for c in dataset.column_names if c not in cols_to_keep]
    if cols_to_remove:
        print(f"Removing extra columns: {cols_to_remove}")
        dataset = dataset.remove_columns(cols_to_remove)

    print(f"Filtering with max_seq_len={args.max_seq_len}")

    def is_within_length(example):
        try:
            chosen_len = count_tokens(example["chosen"], tokenizer)
            rejected_len = count_tokens(example["rejected"], tokenizer)
        except Exception:
            return False
        return chosen_len <= args.max_seq_len and rejected_len <= args.max_seq_len

    dataset = dataset.filter(
        is_within_length,
        num_proc=args.num_proc,
        desc="Filtering by sequence length",
    )

    filtered_size = len(dataset)
    removed = original_size - filtered_size
    print(f"Filtered dataset size: {filtered_size} (removed {removed}, {removed/original_size*100:.1f}%)")

    print(f"Saving to {args.output_dir}")
    DatasetDict({"train_split": dataset}).save_to_disk(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter preference dataset by max sequence length")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to input HF dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save filtered dataset")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Model/tokenizer to use for token counting")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum sequence length in tokens")
    parser.add_argument("--num-proc", type=int, default=64, help="Number of parallel workers for filtering")
    args = parser.parse_args()
    main(args)
