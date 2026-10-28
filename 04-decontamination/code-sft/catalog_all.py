"""Resolve a user-supplied dataset manifest to pinned source files."""

import argparse
import concurrent.futures
import fnmatch
import json
from pathlib import Path
import requests
import yaml
from huggingface_hub import HfApi, hf_hub_url
from standardize import atomic_json
from source_layout import is_training_source_file

SUFFIXES = (".parquet", ".jsonl", ".jsonl.gz", ".json", ".json.gz", ".arrow")


def resolve(entry, output):
    entry = dict(entry)
    api = HfApi()
    info = api.dataset_info(
        entry["repo"], revision=entry.get("revision"), files_metadata=True
    )
    entry["revision"] = info.sha
    card = info.card_data.to_dict() if info.card_data else {}
    declared = card.get("license")
    entry["license"] = (
        declared
        if isinstance(declared, str)
        else json.dumps(declared)
        if declared
        else "unknown"
    )
    notes = entry.get("license_notes", "")
    entry["license_status"] = (
        "conflicting"
        if any(s in notes.lower() for s in ("conflict", "ambiguous"))
        else "declared"
        if declared
        else "unknown"
    )
    entry["license_notes"] = (
        notes
        or "Publisher metadata declaration; underlying source licenses may also apply."
    )
    entry["metadata_source"] = (
        f"https://huggingface.co/datasets/{entry['repo']}/blob/{info.sha}/README.md"
    )
    entry["gated"] = info.gated
    entry.setdefault("synthetic_status", "unknown")
    entry.setdefault(
        "synthetic_notes",
        "Dataset-level classification from supplied research; original row provenance preserved. Not independently audited.",
    )
    files = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if not name.endswith(SUFFIXES):
            continue
        if Path(name).name in ("dataset_info.json", "dataset_infos.json", "state.json"):
            continue
        if not is_training_source_file(name, card):
            continue
        if entry.get("data_files") and not any(
            fnmatch.fnmatch(name, p) for p in entry["data_files"]
        ):
            continue
        if entry.get("path_contains_any") and not any(
            p in name for p in entry["path_contains_any"]
        ):
            continue
        config, split = str(Path(name).parent), "unknown"
        configs = card.get("configs", [])
        if isinstance(configs, list):
            for cfg in configs:
                for spec in (
                    cfg.get("data_files", [])
                    if isinstance(cfg.get("data_files"), list)
                    else []
                ):
                    if not isinstance(spec, dict):
                        continue
                    paths = spec.get("path", [])
                    if isinstance(paths, str):
                        paths = [paths]
                    if any(fnmatch.fnmatch(name, p) for p in paths):
                        config, split = (
                            cfg.get("config_name", "default"),
                            spec.get("split", "unknown"),
                        )
        if entry["category"] == "benchmarks" and split == "unknown":
            prefix = Path(name).name.split("-")[0]
            split = (
                "validation"
                if prefix in ("valid", "validation")
                else entry.get("raw_split", "test")
            )
        files.append(
            dict(
                path=name, bytes=sibling.size, source_config=config, source_split=split
            )
        )
    if not files:
        raise ValueError("No supported source data files matched the declared scope")
    entry["files"] = files
    entry["source_bytes"] = sum(f["bytes"] or 0 for f in files)
    dest = output / "cards" / entry["repo"]
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(dest / "metadata.json", dict(revision=info.sha, card=card, files=files))
    if any(s.rfilename == "README.md" for s in info.siblings):
        response = requests.get(
            hf_hub_url(
                entry["repo"], "README.md", repo_type="dataset", revision=info.sha
            ),
            timeout=60,
        )
        response.raise_for_status()
        (dest / "README.md").write_text(response.text)
    return entry
