#!/usr/bin/env python3
"""
Download all languages of bigcode/commitpackft into one folder.

commitpackft has one config per programming language (277 of them). Each is saved
as its own HuggingFace dataset in <download-folder>/commitpackft/<language>/, with a
dataset_metadata.json (repo, config, commit hash, features, split sizes, download
date). The metadata file is written last, so a language that has one is complete.

This is a script-based dataset, so it needs `datasets` < 4 (4.0 removed dataset
scripts) and it runs the dataset's loading script from the Hub.

Languages that are already complete are skipped, so the script can be rerun to
resume. A summary is written to commitpackft/_all_configs.json. The exit code is 1
if any language failed.

Usage:
  python hf-download-commitpackft.py --download-folder data/01-hf-data
  python hf-download-commitpackft.py --download-folder data/01-hf-data --configs python rust
"""

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

try:
    import datasets
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import HfApi
except ImportError:
    print("Error: datasets library not found. Please install with: pip install datasets==3.3.2")
    sys.exit(1)

DATASET = "bigcode/commitpackft"


def download_config(config, root, revision):
    """Download one language, at a fixed commit, and save it to <root>/<config>/."""
    folder = root / config
    folder.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        DATASET, config, revision=revision,
        cache_dir=str(folder / ".cache"), trust_remote_code=True,
    )

    metadata = {
        "dataset_name": DATASET,
        "subset": config,
        "split": None,
        "commit_hash": revision,
        "download_date": datetime.now().isoformat(),
        "datasets_library_version": datasets.__version__,
    }

    # DatasetDict: take the info of the first split
    info = None
    if hasattr(dataset, "info"):
        info = dataset.info
    elif isinstance(dataset, dict) and len(dataset) > 0:
        info = dataset[next(iter(dataset))].info

    if info:
        metadata["dataset_info"] = {
            "version": str(info.version) if hasattr(info, "version") else None,
            "description": info.description,
            "features": str(info.features),
            "splits": {
                name: {"num_examples": s.num_examples, "num_bytes": s.num_bytes}
                for name, s in info.splits.items()
            } if hasattr(info, "splits") and info.splits else None,
            "download_size": info.download_size,
            "dataset_size": info.dataset_size,
        }

    dataset.save_to_disk(str(folder))

    # Written last: its presence means the language is complete.
    with open(folder / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Download all languages of bigcode/commitpackft into one folder"
    )
    parser.add_argument("--download-folder", required=True, help="Folder to download into")
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Only these languages (default: all of them)",
    )
    args = parser.parse_args()

    if int(datasets.__version__.split(".")[0]) >= 4:
        print(f"Error: datasets {datasets.__version__} cannot run dataset scripts; "
              "use datasets<4 (e.g. 3.3.2)")
        sys.exit(1)

    root = Path(args.download_folder) / "commitpackft"
    root.mkdir(parents=True, exist_ok=True)

    # Pin the whole run to one commit; script-based downloads do not carry it in their URLs.
    revision = HfApi().dataset_info(DATASET).sha
    all_configs = get_dataset_config_names(DATASET, revision=revision, trust_remote_code=True)
    configs = args.configs or all_configs
    unknown = [c for c in configs if c not in all_configs]
    if unknown:
        print(f"Error: unknown configs {unknown}")
        sys.exit(1)
    print(f"{DATASET}@{revision}: {len(configs)} of {len(all_configs)} configs -> {root}", flush=True)

    done, skipped, failed = [], [], []
    for i, config in enumerate(configs, 1):
        if (root / config / "dataset_metadata.json").exists():
            skipped.append(config)
            continue
        print(f"[{i}/{len(configs)}] {config}", flush=True)
        try:
            download_config(config, root, revision)
            done.append(config)
        except Exception as e:
            failed.append(config)
            print(f"!! {config} failed: {e}", flush=True)
            traceback.print_exc()

    summary = {
        "dataset_name": DATASET,
        "commit_hash": revision,
        "configs": len(configs),
        "downloaded": len(done),
        "skipped_already_downloaded": len(skipped),
        "failed": failed,
        "finished": datetime.now().isoformat(),
    }
    # Only a full run writes the summary a rerun can trust
    if not args.configs:
        with open(root / "_all_configs.json", "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
