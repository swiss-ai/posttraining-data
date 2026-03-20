"""
Finalize the alignment dataset by combining:
  - Combined annotations with ref completion rewards (from combine_annotations.py)
  - Reference log-probabilities (from stage 3 logprobs computation)

Produces the final training dataset with columns:
  - chosen, rejected
  - chosen_reward, rejected_reward (from chosen_score / rejected_score)
  - chosen_model, rejected_model
  - chosen_ref_logprobs, rejected_ref_logprobs
  - chosen_qrpo_rewards, rejected_qrpo_rewards

The QRPO rewards are derived from the judge-annotated reference completions:
  chosen_qrpo_rewards  = max(ref_completion_rewards)
  rejected_qrpo_rewards = min(ref_completion_rewards)

Usage:
    python finalize_dataset.py \
        --annotations-path /path/to/combined_annotations \
        --logprobs-path /path/to/MaxMin-Filtered-Logprobs/merged \
        --output-dir /path/to/final_dataset
"""

import argparse
from pathlib import Path

from datasets import DatasetDict, load_from_disk


def main(args):
    # Load datasets
    print("Loading combined annotations...")
    ann_ds = load_from_disk(args.annotations_path)
    if hasattr(ann_ds, "keys"):
        split = "train_split" if "train_split" in ann_ds else list(ann_ds.keys())[0]
        ann_ds = ann_ds[split]
    print(f"  Annotations: {len(ann_ds)} rows, columns: {ann_ds.column_names}")

    print("Loading logprobs...")
    lp_ds = load_from_disk(args.logprobs_path)
    if hasattr(lp_ds, "keys"):
        split = "train_split" if "train_split" in lp_ds else list(lp_ds.keys())[0]
        lp_ds = lp_ds[split]
    print(f"  Logprobs: {len(lp_ds)} rows, columns: {lp_ds.column_names}")

    # Validate row counts match
    assert len(ann_ds) == len(lp_ds), (
        f"Row count mismatch: annotations={len(ann_ds)}, logprobs={len(lp_ds)}. "
        f"Both must come from the same base dataset."
    )

    # Extract QRPO rewards: max → chosen, min → rejected
    print("Computing QRPO rewards from ref completion rewards...")
    chosen_qrpo_rewards = [max(r) for r in ann_ds["ref_completion_rewards"]]
    rejected_qrpo_rewards = [min(r) for r in ann_ds["ref_completion_rewards"]]

    # Build final dataset
    result = ann_ds.select_columns(["chosen", "rejected"])
    result = result.add_column("chosen_reward", [float(s) for s in ann_ds["chosen_score"]])
    result = result.add_column("rejected_reward", [float(s) for s in ann_ds["rejected_score"]])
    result = result.add_column("chosen_model", ann_ds["chosen_model"])
    result = result.add_column("rejected_model", ann_ds["rejected_model"])
    result = result.add_column("chosen_ref_logprobs", lp_ds["ref_chosen_logprob"])
    result = result.add_column("rejected_ref_logprobs", lp_ds["ref_rejected_logprob"])
    result = result.add_column("chosen_qrpo_rewards", chosen_qrpo_rewards)
    result = result.add_column("rejected_qrpo_rewards", rejected_qrpo_rewards)

    print(f"\nFinal dataset: {len(result)} rows")
    print(f"Columns: {result.column_names}")

    # Stats
    avg_chosen_r = sum(chosen_qrpo_rewards) / len(chosen_qrpo_rewards)
    avg_rejected_r = sum(rejected_qrpo_rewards) / len(rejected_qrpo_rewards)
    print(f"Avg chosen QRPO reward:   {avg_chosen_r:.4f}")
    print(f"Avg rejected QRPO reward: {avg_rejected_r:.4f}")

    none_logprobs = sum(1 for x in lp_ds["ref_chosen_logprob"] if x is None)
    if none_logprobs > 0:
        print(
            f"WARNING: {none_logprobs} rows have None logprobs (exceeded max_seq_len)"
        )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train_split": result}).save_to_disk(str(output_dir))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finalize alignment dataset with logprobs and QRPO rewards"
    )
    parser.add_argument(
        "--annotations-path",
        type=str,
        required=True,
        help="Path to combined annotations dataset (output of combine_annotations.py)",
    )
    parser.add_argument(
        "--logprobs-path",
        type=str,
        required=True,
        help="Path to merged logprobs dataset (MaxMin-Filtered-Logprobs/merged)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save final dataset",
    )
    args = parser.parse_args()
    main(args)
