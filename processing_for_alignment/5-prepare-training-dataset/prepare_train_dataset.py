"""
Step 5: Compute quantile rewards and prepare the final training dataset.

Adapted from swiss_alignment.data_alignment.prepare_train_dataset_swissaiformat.

Takes three datasets:
  1. --rewards-dataset-path: the original filtered dataset with chosen_score/rejected_score,
     plus optionally chosen_model/rejected_model
  2. --dataset-path: the combined annotations dataset with chosen, rejected,
     ref_completion_rewards
  3. --logprobs-dataset-path: the logprobs dataset with ref_chosen_logprob, ref_rejected_logprob

Merges columns from (1) and (3) into (2), computes quantile rewards, and keeps only
the columns needed by PreferenceTrainer.

Usage:
    python prepare_train_dataset.py \
        --dataset-path /path/to/combined_dataset \
        --rewards-dataset-path /path/to/filtered_dataset \
        --logprobs-dataset-path /path/to/logprobs_dataset \
        --output-dir /path/to/training_dataset \
        --num-proc 288
"""

import argparse
from datasets import DatasetDict, load_from_disk



def load_dataset_split(path, label=""):
    """Load a dataset, extracting train_split if it's a DatasetDict."""
    ds = load_from_disk(path)
    if hasattr(ds, "keys"):
        split = "train_split" if "train_split" in ds else list(ds.keys())[0]
        print(f"  {label}DatasetDict detected, using '{split}' split")
        ds = ds[split]
    return ds


def main(args):
    # Load the combined annotations dataset (chosen, rejected, ref_completion_rewards)
    print(f"Loading annotations dataset from {args.dataset_path}")
    dataset = load_dataset_split(args.dataset_path, label="annotations: ")
    print(f"  Size: {len(dataset)}, Columns: {dataset.column_names}")

    # Load the rewards dataset (chosen_score/rejected_score)
    print(f"Loading rewards dataset from {args.rewards_dataset_path}")
    rewards_ds = load_dataset_split(args.rewards_dataset_path, label="rewards: ")
    print(f"  Size: {len(rewards_ds)}, Columns: {rewards_ds.column_names}")

    # Load the logprobs dataset (ref_chosen_logprob, ref_rejected_logprob)
    print(f"Loading logprobs dataset from {args.logprobs_dataset_path}")
    logprobs_ds = load_dataset_split(args.logprobs_dataset_path, label="logprobs: ")
    print(f"  Size: {len(logprobs_ds)}, Columns: {logprobs_ds.column_names}")

    assert len(dataset) == len(rewards_ds), (
        f"Annotations vs rewards size mismatch: {len(dataset)} vs {len(rewards_ds)}"
    )
    assert len(dataset) == len(logprobs_ds), (
        f"Annotations vs logprobs size mismatch: {len(dataset)} vs {len(logprobs_ds)}"
    )

    # Pre-extract columns as lists (fast, arrow-backed)
    print("Extracting columns to merge...")
    chosen_scores = rewards_ds["chosen_score"]
    rejected_scores = rewards_ds["rejected_score"]
    has_models = "chosen_model" in rewards_ds.column_names
    if has_models:
        chosen_models = rewards_ds["chosen_model"]
        rejected_models = rewards_ds["rejected_model"]

    ref_chosen_logprobs = logprobs_ds["ref_chosen_logprob"]
    ref_rejected_logprobs = logprobs_ds["ref_rejected_logprob"]

    # Verify required columns
    for col in ["chosen", "rejected", "ref_completion_rewards"]:
        assert col in dataset.column_names, f"Missing required column: {col}"

    # Single .map() pass: merge all external columns + compute quantiles
    def process_row(row, idx):
        ref_rewards = row["ref_completion_rewards"]
        n = len(ref_rewards)
        chosen_r = float(chosen_scores[idx])
        rejected_r = float(rejected_scores[idx])
        result = {
            "chosen_reward": chosen_r,
            "rejected_reward": rejected_r,
            "chosen_quantile_reward": sum(r <= chosen_r for r in ref_rewards) / n,
            "rejected_quantile_reward": sum(r <= rejected_r for r in ref_rewards) / n,
            "ref_rewards": ref_rewards,
            "ref_chosen_logprob": float(ref_chosen_logprobs[idx]),
            "ref_rejected_logprob": float(ref_rejected_logprobs[idx]),
        }
        if has_models:
            result["chosen_model"] = chosen_models[idx]
            result["rejected_model"] = rejected_models[idx]
        return result

    print("Computing quantile rewards & merging columns...")
    dataset = dataset.map(
        process_row,
        with_indices=True,
        num_proc=args.num_proc,
        desc="Merging rewards & computing quantiles",
    )

    # Stats
    avg_chosen_q = sum(dataset["chosen_quantile_reward"]) / len(dataset)
    avg_rejected_q = sum(dataset["rejected_quantile_reward"]) / len(dataset)
    print(f"  avg chosen_quantile_reward:   {avg_chosen_q:.4f}")
    print(f"  avg rejected_quantile_reward: {avg_rejected_q:.4f}")

    print(f"\nFinal columns: {dataset.column_names}")
    print(f"Final size: {len(dataset)}")

    print(f"\nSaving to {args.output_dir}")
    DatasetDict({"train_split": dataset}).save_to_disk(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute quantile rewards and prepare training dataset"
    )
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Combined annotations dataset (chosen, rejected, ref_completion_rewards, ...)")
    parser.add_argument("--rewards-dataset-path", type=str, required=True,
                        help="Original filtered dataset with chosen_score/rejected_score")
    parser.add_argument("--logprobs-dataset-path", type=str, required=True,
                        help="Logprobs dataset with ref_chosen_logprob, ref_rejected_logprob")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-proc", type=int, default=288)
    args = parser.parse_args()
    main(args)
