#!/usr/bin/env python3
"""Pin and download public coding evaluation data without executing dataset scripts."""

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import requests
import yaml
from huggingface_hub import hf_hub_url

from catalog_all import resolve
from standardize import atomic_json, safe_name, write_partition


def adapt(row, entry, source_file):
    from preprocessors import get_processor

    return get_processor(entry["repo"]).prepare(
        row, dict(entry, category="benchmarks"), {"path": source_file}
    )


def download_file(entry, spec, output):
    name = spec["path"]
    config, split = spec["source_config"], spec["source_split"]
    # Unmapped raw files have an explicit evaluation split, never an inferred train split.
    if split == "unknown":
        prefix = Path(name).name.split("-")[0]
        split = (
            "validation"
            if prefix in ("valid", "validation")
            else entry.get("raw_split", "test")
        )
    partition = (
        output
        / "benchmarks"
        / entry["repo"]
        / ("config=" + safe_name(config))
        / ("split=" + safe_name(split))
        / ("file=" + hashlib.sha256(name.encode()).hexdigest()[:16])
    )
    if (partition / "_SUCCESS.json").exists():
        return json.loads((partition / "_SUCCESS.json").read_text())
    url = hf_hub_url(
        entry["repo"], name, repo_type="dataset", revision=entry["revision"]
    )
    headers = (
        {"Authorization": "Bearer " + os.environ["HF_TOKEN"]}
        if os.environ.get("HF_TOKEN")
        else {}
    )
    # Temporary raw files avoid loading whole Parquet shards into RAM; removed on completion.
    with tempfile.TemporaryDirectory(prefix="benchmark-") as tmp:
        path = Path(tmp) / Path(name).name
        with requests.get(
            url, headers=headers, stream=True, timeout=(30, 300)
        ) as response:
            response.raise_for_status()
            with path.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)

        def rows():
            if name.endswith(".parquet"):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                    for row in batch.to_pylist():
                        yield adapt(row, entry, name)
            else:
                with path.open() as handle:
                    for line in handle:
                        if line.strip():
                            yield adapt(json.loads(line), entry, name)

        return write_partition(rows(), partition, entry, config, split, 2000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("benchmarks.yaml")
    )
    parser.add_argument(
        "--catalog", type=Path, default=Path(__file__).with_name("catalog_v2")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Resolve the manifest and register pinned benchmarks in the catalog",
    )
    args = parser.parse_args()
    entries = yaml.safe_load(args.manifest.read_text())["datasets"]
    lock = args.manifest.with_suffix(".lock.json")
    if args.resolve:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            entries = list(
                pool.map(lambda entry: resolve(entry, args.catalog), entries)
            )
        atomic_json(lock, {"datasets": entries})
        catalog_path = args.catalog / "catalog.json"
        catalog = (
            json.loads(catalog_path.read_text())
            if catalog_path.exists()
            else {"datasets": [], "unresolved": []}
        )
        repos = {e["repo"] for e in entries}
        catalog["datasets"] = sorted(
            [e for e in catalog["datasets"] if e["repo"] not in repos] + entries,
            key=lambda e: e["repo"],
        )
        catalog["unresolved"] = [
            e for e in catalog.get("unresolved", []) if e["repo"] not in repos
        ]
        catalog["requested_count"] = len(catalog["datasets"]) + len(
            catalog["unresolved"]
        )
        atomic_json(catalog_path, catalog)
    else:
        entries = json.loads(lock.read_text())["datasets"]
    if not args.output:
        return
    failures = []
    results = {e["repo"]: [] for e in entries}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {}
        for entry in entries:
            root = args.output / "benchmarks" / entry["repo"]
            root.mkdir(parents=True, exist_ok=True)
            source = root / "source.json"
            if source.exists() and json.loads(source.read_text()) != entry:
                raise ValueError(
                    f"Source changed: use a new output directory for {entry['repo']}"
                )
            atomic_json(source, entry)
            (root / "_SUCCESS.json").unlink(missing_ok=True)
            for spec in entry["files"]:
                tasks[pool.submit(download_file, entry, spec, args.output)] = (
                    entry,
                    spec,
                )
        for future in concurrent.futures.as_completed(tasks):
            entry, spec = tasks[future]
            try:
                result = future.result()
                results[entry["repo"]].append(result)
                print(
                    f"DONE {entry['repo']} {spec['path']}: {result['rows']}", flush=True
                )
            except Exception as exc:
                failures.append(
                    {"repo": entry["repo"], "file": spec["path"], "error": str(exc)}
                )
                print(f"FAILED {failures[-1]}", flush=True)
    for entry in entries:
        root = args.output / "benchmarks" / entry["repo"]
        errors = [f for f in failures if f["repo"] == entry["repo"]]
        if errors:
            atomic_json(root / "_FAILED.json", errors)
        else:
            atomic_json(
                root / "_SUCCESS.json",
                {
                    "dataset": entry["repo"],
                    "revision": entry["revision"],
                    "rows": sum(s["rows"] for s in results[entry["repo"]]),
                    "partitions": results[entry["repo"]],
                },
            )
            (root / "_FAILED.json").unlink(missing_ok=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
