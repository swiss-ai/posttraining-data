"""
Combine N annotated completion splits into a single dataset with per-completion rewards.

Loads each completion_i/ directory, computes a scalar reward from the judge annotation
(mean expected score across selected aspects), and produces one dataset with a
ref_completion_rewards column (list of N floats per row).

Usage:
    python combine_annotations.py \
        --annotations-dir /path/to/MaxMin-Filtered-Ref-Completions-Annotated \
        --output-dir /path/to/MaxMin-Filtered-Ref-Completions-Combined \
        --aspects helpfulness honesty instruction_following \
        --num-cpus 288
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm

from datasets import DatasetDict, load_from_disk

ALL_ASPECTS = ["helpfulness", "honesty", "instruction_following", "truthfulness"]


def expected_score(dist: dict) -> float:
    """Compute E[score] = sum(k * P(k)) for k in 1..5."""
    return sum(float(k) * float(v) for k, v in dist.items())


def _load_one(comp_dir: str) -> "Dataset":
    """Load a single completion directory (top-level for pickling)."""
    return load_from_disk(comp_dir)


def _compute_reward_column(args_tuple):
    """Compute reward column for one completion split (top-level for pickling).

    Returns (index, rewards_list, avg_reward).
    """
    idx, comp_dir_str, aspects = args_tuple
    ds = load_from_disk(comp_dir_str)

    def _annotation_to_reward(annotation: dict) -> float:
        scores = [expected_score(annotation[asp]) for asp in aspects if asp in annotation]
        return sum(scores) / len(scores) if scores else 0.0

    rewards = ds.map(
        lambda row: {"_reward": _annotation_to_reward(row["annotation"])},
        num_proc=1,  # already parallelised at the outer level
        desc=f"Rewards completion_{idx}",
    )["_reward"]

    avg = sum(rewards) / len(rewards)
    return idx, rewards, avg


def main(args):
    annotations_dir = Path(args.annotations_dir)
    aspects = args.aspects
    num_cpus = args.num_cpus

    print(f"Using aspects: {aspects}")

    # Find all completion directories, sorted by index
    completion_dirs = sorted(
        [
            d
            for d in annotations_dir.iterdir()
            if d.is_dir() and d.name.startswith("completion_")
        ],
        key=lambda p: int(p.name.split("_")[1]),
    )
    n_completions = len(completion_dirs)
    print(f"Found {n_completions} completion directories")

    # --- Parallel load + reward computation ---
    # Each worker loads one completion split and computes its rewards.
    # With 30 completions and 288 CPUs this saturates I/O and compute nicely.
    work_items = [
        (i, str(d), aspects) for i, d in enumerate(completion_dirs)
    ]

    workers = min(n_completions, num_cpus)
    print(f"Computing rewards in parallel with {workers} workers...")

    all_rewards = [None] * n_completions
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for idx, rewards, avg in pool.map(_compute_reward_column, work_items):
            all_rewards[idx] = rewards
            print(f"  completion_{idx}: avg reward = {avg:.4f}")

    # --- Load base dataset (only need shared columns) ---
    print("Loading base dataset for shared columns...")
    base_ds = load_from_disk(str(completion_dirs[0]))
    num_rows = len(base_ds)
    print(f"All datasets have {num_rows} rows")

    # --- Assemble ref_completion_rewards directly in Python ---
    print("Zipping rewards into per-row lists...")
    ref_completion_rewards = [
        [all_rewards[comp_idx][row_idx] for comp_idx in range(n_completions)]
        for row_idx in tqdm(range(num_rows), desc="Zipping rewards")
    ]

    # Drop completion-specific columns, then add the combined reward list
    columns_to_drop = [
        c for c in base_ds.column_names if c in ("response", "annotation")
    ]
    result_ds = base_ds.remove_columns(columns_to_drop)
    result_ds = result_ds.add_column("ref_completion_rewards", ref_completion_rewards)

    print(f"\nResult dataset: {len(result_ds)} rows")
    print(f"Columns: {result_ds.column_names}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train_split": result_ds}).save_to_disk(str(output_dir))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine annotated completion splits into single dataset with rewards"
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        required=True,
        help="Directory containing completion_0/, completion_1/, ... subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save combined dataset",
    )
    parser.add_argument(
        "--aspects",
        nargs="+",
        default=ALL_ASPECTS,
        choices=ALL_ASPECTS,
        help=f"Aspects to include in reward computation (default: all). "
             f"Choices: {ALL_ASPECTS}",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=288,
        help="Number of CPUs for parallel processing (default: 288)",
    )
    args = parser.parse_args()
    main(args)
