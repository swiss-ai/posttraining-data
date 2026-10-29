"""Resumable streaming driver for deterministic mixture annotation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .deterministic import CLASSIFIER_VERSION, annotate_record
from .io import atomic_json, discover_inputs, input_identity, iter_records, write_shard
from .judge import SemanticJudge
from .report import Summary
from .taxonomy import Taxonomy


PIPELINE_VERSION = "0.2.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha256() -> str:
    """Fingerprint executable pipeline source so dirty-tree runs remain reproducible."""
    package = Path(__file__).resolve().parent
    paths = sorted(package.glob("*.py"))
    paths.append(package.parent / "quantify_sft_mixture.py")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _identity(
    inputs: list[Path],
    taxonomy: Taxonomy,
    output_format: str,
    rows_per_shard: int,
    limit: int | None,
    judge: SemanticJudge | None,
    num_input_shards: int,
    input_shard_index: int,
    runtime_metadata: dict[str, str],
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "classifier_code_sha256": _code_sha256(),
        "code_commit": _git_commit(),
        "taxonomy": {
            "id": taxonomy.taxonomy_id,
            "version": taxonomy.version,
            "path": str(taxonomy.path),
            "sha256": _sha256(taxonomy.path),
        },
        "inputs": input_identity(inputs),
        "output_format": output_format,
        "rows_per_shard": rows_per_shard,
        "limit": limit,
        "input_sharding": {
            "num_shards": num_input_shards,
            "shard_index": input_shard_index,
            "method": "global_row_index_modulo",
        },
        "runtime_metadata": dict(sorted(runtime_metadata.items())),
        "judge": judge.run_identity() if judge is not None else None,
    }


def _load_existing_annotations(output: Path, output_format: str) -> Iterable[dict[str, Any]]:
    """Read flattened completed shards to restore summary counters on resume."""
    suffix = ".jsonl" if output_format == "jsonl" else ".parquet"
    for row in iter_records(sorted(output.glob(f"part-*{suffix}"))):
        classifications = json.loads(row.pop("classifications_json"))
        yield {
            **{
                key: row.get(key)
                for key in (
                    "record_id",
                    "source_dataset",
                    "source_revision",
                    "source_split",
                    "source_row_index",
                    "taxonomy_version",
                    "classifier_version",
                    "input_hash",
                    "source_tokens",
                    "training_loss_tokens",
                    "judge_provider",
                    "judge_model",
                    "judge_model_revision",
                    "judge_prompt_version",
                    "judge_prompt_tokens",
                    "judge_completion_tokens",
                    "judge_response_hash",
                    "judge_error",
                )
            },
            "classifications": classifications,
        }


def run(
    input_paths: list[str | Path],
    output: str | Path,
    taxonomy: Taxonomy,
    *,
    output_format: str = "parquet",
    rows_per_shard: int = 50_000,
    input_batch_size: int = 1024,
    limit: int | None = None,
    judge: SemanticJudge | None = None,
    num_input_shards: int = 1,
    input_shard_index: int = 0,
    runtime_metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if num_input_shards <= 0:
        raise ValueError("num_input_shards must be positive")
    if not 0 <= input_shard_index < num_input_shards:
        raise ValueError("input_shard_index must be in [0, num_input_shards)")
    runtime_metadata = runtime_metadata or {}
    if not all(
        isinstance(key, str) and key and isinstance(value, str) and value
        for key, value in runtime_metadata.items()
    ):
        raise ValueError("runtime_metadata must contain non-empty string keys and values")
    inputs = discover_inputs(input_paths)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = _identity(
        inputs,
        taxonomy,
        output_format,
        rows_per_shard,
        limit,
        judge,
        num_input_shards,
        input_shard_index,
        runtime_metadata,
    )
    run_path = output / "run.json"
    if run_path.exists():
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        existing.pop("created_at_utc", None)
        if existing != identity:
            raise ValueError("existing output was created with different inputs or options")
    else:
        atomic_json(
            run_path,
            {"created_at_utc": datetime.now(timezone.utc).isoformat(), **identity},
        )

    success_path = output / "_SUCCESS.json"
    if success_path.exists():
        return json.loads(success_path.read_text(encoding="utf-8"))
    progress_path = output / "_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {
        "rows": 0,
        "next_shard": 0,
        "shards": [],
    }
    resume_at = int(progress["rows"])
    summary = Summary()
    if resume_at:
        for annotation in _load_existing_annotations(output, output_format):
            summary.add(annotation)
        if summary.records != resume_at:
            raise ValueError(
                f"checkpoint says {resume_at} rows but completed shards contain {summary.records}"
            )

    shard_rows: list[dict[str, Any]] = []
    pending_records: list[dict[str, Any]] = []
    pending_annotations: list[dict[str, Any]] = []
    processed = resume_at

    def flush_shard() -> None:
        nonlocal shard_rows
        if not shard_rows:
            return
        shard = write_shard(shard_rows, output, int(progress["next_shard"]), output_format)
        progress["shards"].append(shard)
        progress["next_shard"] += 1
        progress["rows"] += len(shard_rows)
        atomic_json(progress_path, progress)
        shard_rows = []

    def accept(annotation: dict[str, Any]) -> None:
        summary.add(annotation)
        shard_rows.append(annotation)
        if len(shard_rows) >= rows_per_shard:
            flush_shard()

    def process_judge_batch() -> None:
        nonlocal pending_records, pending_annotations
        if not pending_annotations:
            return
        assert judge is not None
        for judged in judge.annotate_batch(pending_records, pending_annotations):
            accept(judged)
        pending_records = []
        pending_annotations = []

    selected_index = 0
    for source_index, record in enumerate(iter_records(inputs, batch_size=input_batch_size)):
        if source_index % num_input_shards != input_shard_index:
            continue
        if selected_index < resume_at:
            selected_index += 1
            continue
        if limit is not None and processed >= limit:
            break
        try:
            annotation = annotate_record(record, taxonomy)
        except Exception as exc:
            raise ValueError(f"failed to annotate input row {source_index}: {exc}") from exc
        processed += 1
        selected_index += 1
        if judge is None:
            accept(annotation)
        else:
            pending_records.append(record)
            pending_annotations.append(annotation)
            if len(pending_annotations) >= judge.settings.batch_size:
                process_judge_batch()
    if judge is not None:
        process_judge_batch()
    flush_shard()

    summary_value = {
        "pipeline_version": PIPELINE_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "taxonomy_id": taxonomy.taxonomy_id,
        "taxonomy_version": taxonomy.version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **summary.as_dict(),
    }
    atomic_json(output / "summary.json", summary_value)
    success = {
        "status": "complete",
        "rows": processed,
        "shards": progress["shards"],
        "summary": "summary.json",
    }
    atomic_json(success_path, success)
    progress_path.unlink(missing_ok=True)
    return success
