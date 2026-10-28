"""Pin a manifest to dataset revisions and enumerate supported source files."""

import argparse
import concurrent.futures
from pathlib import Path
import yaml
from catalog_all import resolve
from standardize import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    entries = yaml.safe_load(args.manifest.read_text())["datasets"]
    if len({e["repo"] for e in entries}) != len(entries):
        parser.error("Duplicate repository in manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(resolve, e, args.output): e for e in entries}
        for task in concurrent.futures.as_completed(tasks):
            try:
                results.append(task.result())
            except Exception as error:
                failures.append(dict(repo=tasks[task]["repo"], error=str(error)))
    atomic_json(
        args.output / "catalog.json",
        dict(datasets=sorted(results, key=lambda e: e["repo"]), unresolved=failures),
    )
    if failures:
        raise SystemExit(
            f"{len(failures)} sources could not be resolved; see catalog.json"
        )


if __name__ == "__main__":
    main()
