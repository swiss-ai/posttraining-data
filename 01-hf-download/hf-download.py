#!/usr/bin/env python3
"""
HuggingFace Dataset Downloader

Downloads datasets from HuggingFace Hub with compatibility checks.
Requires datasets version 3.3.2 for compatibility with current docker images.
"""

import argparse
import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

try:
    import datasets
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not found. Please install with: pip install datasets==3.3.2")
    sys.exit(1)


def download_dataset(dataset_name, download_folder, subset=None, split=None, nest_subset=False, trust_remote_code=False):
    """Download a dataset from HuggingFace Hub"""
    try:
        # Extract dataset name without organization
        dataset_short_name = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
        # Each subset is saved as its own dataset, e.g. OpenCodeReasoning-split_0
        if subset and not nest_subset:
            dataset_short_name = f"{dataset_short_name}-{subset}"
        dataset_folder = os.path.join(download_folder, dataset_short_name)
        if subset and nest_subset:
            # All subsets of one dataset live side by side: <name>/<subset>
            dataset_folder = os.path.join(dataset_folder, subset)

        print(f"Downloading {dataset_name} to {dataset_folder}")

        # Create dataset folder if it doesn't exist
        Path(dataset_folder).mkdir(parents=True, exist_ok=True)

        # Set cache directory to dataset folder
        cache_dir = os.path.join(dataset_folder, ".cache")

        # Download dataset
        load_kwargs = {"split": split, "cache_dir": cache_dir}
        if trust_remote_code:
            # Only for script-based datasets, and only with datasets<4
            load_kwargs["trust_remote_code"] = True
        if subset:
            dataset = load_dataset(dataset_name, subset, **load_kwargs)
        else:
            dataset = load_dataset(dataset_name, **load_kwargs)

        print(f"✓ Successfully downloaded {dataset_name}")

        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "subset": subset,
            "split": split,
            "download_date": datetime.now().isoformat(),
            "datasets_library_version": datasets.__version__
        }

        # Extract dataset info
        info = None
        if hasattr(dataset, 'info'):
            info = dataset.info
        elif isinstance(dataset, dict) and len(dataset) > 0:
            # For DatasetDict, get info from first split
            first_split = list(dataset.keys())[0]
            if hasattr(dataset[first_split], 'info'):
                info = dataset[first_split].info

        if info:
            metadata["dataset_info"] = {
                "version": str(info.version) if hasattr(info, 'version') else None,
                "description": info.description,
                "features": str(info.features),
                "splits": {name: {"num_examples": split_info.num_examples, "num_bytes": split_info.num_bytes}
                           for name, split_info in info.splits.items()} if hasattr(info, 'splits') else None,
                "download_size": info.download_size,
                "dataset_size": info.dataset_size
            }

            # Extract commit hash from download checksums if available
            if hasattr(info, 'download_checksums') and info.download_checksums:
                for url in info.download_checksums.keys():
                    if '@' in url and '/' in url:
                        # Extract commit hash from URL like "hf://datasets/tatsu-lab/alpaca@dce01c9b08f87459cf36a430d809084718273017/..."
                        parts = url.split('@')
                        if len(parts) > 1:
                            commit_hash = parts[1].split('/')[0]
                            metadata["commit_hash"] = commit_hash
                            break

        # Save metadata to JSON file
        metadata_path = os.path.join(dataset_folder, "dataset_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_path}")

        # Save dataset using HuggingFace format
        print("Saving dataset...")
        dataset.save_to_disk(dataset_folder)
        print(f"✓ Saved HuggingFace dataset to {dataset_folder}")

        # Print dataset info
        if hasattr(dataset, '__len__'):
            print(f"Number of examples: {len(dataset)}")
        elif isinstance(dataset, dict):
            for split_name, split_dataset in dataset.items():
                print(f"Number of {split_name} examples: {len(split_dataset)}")

        return dataset

    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        # Show the full chain: datasets wraps the real cause in a generic
        # "error occurred while generating the dataset" message.
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets from HuggingFace Hub with compatibility checks"
    )
    parser.add_argument(
        "dataset_name",
        help="Name of the dataset to download (e.g., 'allenai/tulu-3-sft-mixture')"
    )
    parser.add_argument(
        "--download-folder",
        required=True,
        help="Folder to download the dataset to"
    )
    parser.add_argument(
        "--subset",
        help="Dataset subset/configuration name (if applicable)"
    )
    parser.add_argument(
        "--split",
        help="Dataset split to download (e.g., 'train', 'test', 'validation')"
    )

    parser.add_argument(
        "--nest-subset",
        action="store_true",
        help="Save the subset in <dataset>/<subset> instead of <dataset>-<subset>"
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Run the dataset's loading script (script-based datasets; needs datasets<4)"
    )

    args = parser.parse_args()

    # Download the dataset
    download_dataset(
        args.dataset_name,
        args.download_folder,
        args.subset,
        args.split,
        args.nest_subset,
        args.trust_remote_code
    )


if __name__ == "__main__":
    main()
