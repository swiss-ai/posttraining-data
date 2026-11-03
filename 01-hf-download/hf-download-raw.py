#!/usr/bin/env python3
"""
HuggingFace raw-file dataset downloader

Copies the files of a dataset repo (or a part of it) exactly as they are on the
Hub, without loading them through `datasets`. Use it when `load_dataset` cannot
build the dataset, e.g. irregular nested columns or a single column too large
for pyarrow to convert in one chunk.

The folder gets a dataset_metadata.json with the repo name, the commit hash and
the list of files, so that later stages can record the exact source revision.
Standardisation accepts such a folder of parquet/jsonl files; pass
--repo and --revision from dataset_metadata.json if it cannot find them.

Example:
  python hf-download-raw.py nvidia/Nemotron-Terminal-Corpus \
      --download-folder data/01-hf-data --subset dataset_adapters \
      --allow-patterns 'dataset_adapters/*'
"""

import argparse
import fnmatch
import json
import sys
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download the raw files of a HuggingFace dataset repo"
    )
    parser.add_argument("dataset_name", help="e.g. 'nvidia/Nemotron-Terminal-Corpus'")
    parser.add_argument("--download-folder", required=True, help="Folder to download into")
    parser.add_argument(
        "--subset",
        help="Name added to the folder (<dataset>-<subset>), same as hf-download.py",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="+",
        help="Only download files matching these glob patterns (default: everything)",
    )
    parser.add_argument(
        "--nest-subset",
        action="store_true",
        help="Save in <dataset>/<subset> instead of <dataset>-<subset>",
    )
    args = parser.parse_args()

    short = args.dataset_name.split("/")[-1]
    folder = Path(args.download_folder) / short
    if args.subset and args.nest_subset:
        folder = folder / args.subset
    elif args.subset:
        folder = Path(args.download_folder) / f"{short}-{args.subset}"
    folder.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.dataset_info(args.dataset_name, files_metadata=True)
    revision = info.sha
    print(f"Downloading {args.dataset_name}@{revision} to {folder}")
    if args.allow_patterns:
        print(f"Patterns: {args.allow_patterns}")

    # Pin the download to the revision we record in the metadata.
    snapshot_download(
        args.dataset_name,
        repo_type="dataset",
        revision=revision,
        local_dir=str(folder),
        allow_patterns=args.allow_patterns,
    )

    # Verify against the file list and sizes the Hub reports for this commit.
    hub = {s.rfilename: s.size for s in info.siblings}
    patterns = args.allow_patterns or ["*"]
    files = sorted(f for f in hub if any(fnmatch.fnmatch(f, p) for p in patterns))
    if not files:
        print(f"Error: no files matched {args.allow_patterns} in {args.dataset_name}")
        sys.exit(1)
    problems = []
    for f in files:
        path = folder / f
        if not path.is_file() or path.is_symlink():
            problems.append(f"{f}: missing or a symlink")
        elif path.stat().st_size != hub[f]:
            problems.append(f"{f}: {path.stat().st_size} bytes, Hub says {hub[f]}")
    if problems:
        print("Error: download does not match the Hub listing:")
        for line in problems:
            print("  " + line)
        sys.exit(1)

    metadata = {
        "dataset_name": args.dataset_name,
        "subset": args.subset,
        "download_method": "snapshot_download (raw files, no datasets conversion)",
        "verified_against_hub_sizes": True,
        "allow_patterns": args.allow_patterns,
        "commit_hash": revision,
        "download_date": datetime.now().isoformat(),
        "files": [
            {"path": f, "bytes": hub[f]} for f in files
        ],
    }
    metadata_path = folder / "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {len(files)} files, metadata in {metadata_path}")


if __name__ == "__main__":
    main()
