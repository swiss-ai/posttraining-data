#!/usr/bin/env python3
"""Materialize a deterministic, auditable JSONL subset from Parquet SFT shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SUBSET_BUILDER_VERSION = "0.1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _record_identity(row: dict[str, Any]) -> str:
    for key in ("id", "record_id", "conversation_id"):
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value):
            return f"{key}:{value}"
    fields = [
        row.get("source_dataset"),
        row.get("source_revision"),
        row.get("source_split"),
        row.get("source_row_index"),
    ]
    if any(value is not None for value in fields):
        return "source:" + "\0".join("" if value is None else str(value) for value in fields)
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "content:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parquet_files(paths: Iterable[Path]) -> tuple[Path, list[Path]]:
    roots = [path.resolve() for path in paths]
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("this initial subset builder requires exactly one source directory")
    root = roots[0]
    files = sorted(path for path in root.rglob("*.parquet") if path.is_file())
    if not files:
        raise ValueError(f"no Parquet files found under {root}")
    return root, files


def _license_allowed(row: dict[str, Any], allowed: set[str]) -> bool:
    if not allowed:
        return True
    status = str(row.get("license_status") or "").strip().lower()
    license_name = str(row.get("license") or "").strip().lower()
    return status == "declared" and license_name in allowed


def materialize(
    source: Path,
    output: Path,
    *,
    records: int,
    records_per_file: int,
    seed: str,
    allowed_licenses: set[str],
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("subset preparation requires pyarrow") from exc

    if records <= 0 or records_per_file <= 0:
        raise ValueError("records and records-per-file must be positive")
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise ValueError(f"output or manifest already exists: {output}")

    root, files = _parquet_files([source])
    ranked_files = sorted(
        files,
        key=lambda path: (_score(seed, str(path.relative_to(root))), str(path.relative_to(root))),
    )
    selected: list[tuple[str, str, dict[str, Any], Path]] = []
    used_files: list[Path] = []
    considered_files = 0

    for path in ranked_files:
        considered_files += 1
        relative = str(path.relative_to(root))
        file_score = _score(seed, relative)
        candidates: list[tuple[str, str, dict[str, Any]]] = []
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if not _license_allowed(row, allowed_licenses):
                    continue
                identity = _record_identity(row)
                candidates.append((_score(seed, identity), identity, row))
        candidates.sort(key=lambda item: (item[0], item[1]))
        if not candidates:
            continue
        used_files.append(path)
        selected.extend(
            (file_score, record_score, row, path)
            for record_score, _identity, row in candidates[:records_per_file]
        )
        if len(selected) >= records:
            break

    selected.sort(key=lambda item: (item[0], item[1], _record_identity(item[2])))
    selected = selected[:records]
    if len(selected) != records:
        raise ValueError(f"requested {records} records but only selected {len(selected)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for _file_score, _record_score, row, _path in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output)

    source_revisions = sorted(
        {
            str(row.get("source_revision"))
            for _file_score, _record_score, row, _path in selected
            if row.get("source_revision")
        }
    )
    source_datasets = sorted(
        {
            str(row.get("source_dataset"))
            for _file_score, _record_score, row, _path in selected
            if row.get("source_dataset")
        }
    )
    manifest = {
        "subset_builder_version": SUBSET_BUILDER_VERSION,
        "subset_builder_sha256": _sha256(Path(__file__).resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "method": "sha256_ranked_files_and_records",
            "seed": seed,
            "records": records,
            "records_per_file": records_per_file,
            "allowed_licenses": sorted(allowed_licenses),
            "require_license_status": "declared" if allowed_licenses else None,
        },
        "source": {
            "root": str(root),
            "datasets": source_datasets,
            "revisions": source_revisions,
            "available_parquet_files": len(files),
            "considered_parquet_files": considered_files,
            "used_files": [
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(set(used_files))
            ],
        },
        "selected_records": [
            {
                "identity": _record_identity(row),
                "source_file": str(path),
                "selection_sha256": record_score,
            }
            for _file_score, record_score, row, path in selected
        ],
        "output": {
            "path": str(output.resolve()),
            "size": output.stat().st_size,
            "sha256": _sha256(output),
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing Parquet shards")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL subset path")
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--records-per-file", type=int, default=4)
    parser.add_argument("--seed", required=True)
    parser.add_argument(
        "--allow-license",
        action="append",
        default=[],
        help="Require declared rows with this normalized license (repeatable)",
    )
    args = parser.parse_args()
    manifest = materialize(
        args.source,
        args.output.resolve(),
        records=args.records,
        records_per_file=args.records_per_file,
        seed=args.seed,
        allowed_licenses={value.strip().lower() for value in args.allow_license},
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
