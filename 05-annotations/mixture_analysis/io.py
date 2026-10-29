"""Streaming input and sharded sidecar output for mixture annotations."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any, Iterable, Iterator


class DataFormatError(ValueError):
    """Raised for unsupported or malformed pipeline input/output."""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def discover_inputs(paths: Iterable[str | Path]) -> list[Path]:
    discovered: list[Path] = []
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_dir():
            files = sorted(
                file
                for file in path.rglob("*")
                if file.is_file()
                and not any(part.startswith(".") for part in file.relative_to(path).parts)
                and file.suffix.lower() in (".parquet", ".json", ".jsonl")
            )
            if not files:
                raise DataFormatError(f"no Parquet or JSON inputs found under {path}")
            discovered.extend(files)
        elif path.is_file():
            discovered.append(path)
        else:
            raise DataFormatError(f"input does not exist: {path}")
    unique = list(dict.fromkeys(discovered))
    if not unique:
        raise DataFormatError("no inputs were provided")
    return unique


def input_identity(paths: Iterable[Path]) -> list[dict[str, Any]]:
    identity = []
    for path in paths:
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(64 * 1024))
            if stat.st_size > 64 * 1024:
                handle.seek(max(0, stat.st_size - 64 * 1024))
                digest.update(handle.read(64 * 1024))
        identity.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sample_sha256": digest.hexdigest(),
            }
        )
    return identity


def _json_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        if path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataFormatError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise DataFormatError(f"expected an object at {path}:{line_number}")
                yield value
            return
        try:
            value = json.load(handle)
        except json.JSONDecodeError as exc:
            raise DataFormatError(f"invalid JSON in {path}: {exc}") from exc
        if isinstance(value, dict):
            yield value
        elif isinstance(value, list):
            for index, row in enumerate(value):
                if not isinstance(row, dict):
                    raise DataFormatError(f"expected an object at {path}[{index}]")
                yield row
        else:
            raise DataFormatError(f"expected an object or object list in {path}")


def _parquet_records(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DataFormatError(
            "Parquet input requires pyarrow; install the repository requirements"
        ) from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def iter_records(paths: Iterable[Path], *, batch_size: int = 1024) -> Iterator[dict[str, Any]]:
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            yield from _parquet_records(path, batch_size)
        elif suffix in (".json", ".jsonl"):
            yield from _json_records(path)
        else:
            raise DataFormatError(f"unsupported input: {path}")


DIMENSION_COLUMNS = {
        "expert_domains": "expert_domains",
        "task_domains": "task_domains",
        "programming_languages": "programming_languages",
        "work_context": "work_context",
        "interaction_styles": "interaction_styles",
        "swe_change_scope": "swe.change_scope",
        "swe_artifact_types": "swe.artifact_types",
        "terminal_activities": "terminal.activities",
        "terminal_command_names": "terminal.command_names",
        "tool_types": "tool_use.tool_types",
        "available_tool_names": "tool_use.available_tool_names",
        "invoked_tool_names": "tool_use.invoked_tool_names",
        "tool_trajectory_properties": "tool_use.trajectory_properties",
        "swe_categories": "swe_categories",
        "terminal_categories": "terminal_categories",
        "tool_use_categories": "tool_use_categories",
}


def _column_for_dimension(dimension: str) -> str:
    for column, registered_dimension in DIMENSION_COLUMNS.items():
        if registered_dimension == dimension:
            return column
    return dimension.replace(".", "_")


def _flatten_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    classifications = annotation["classifications"]
    row = {key: value for key, value in annotation.items() if key != "classifications"}
    columns = {
        _column_for_dimension(dimension): dimension
        for dimension in classifications
    }
    for column, dimension in columns.items():
        classification = classifications[dimension]
        row[column] = classification["labels"]
        row[f"{column}_status"] = classification["status"]
    row["classifications_json"] = json.dumps(
        classifications, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return row


def _parquet_schema(pa, dimensions: list[str]):
    list_string = pa.list_(pa.string())
    fields = [
        ("record_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_revision", pa.string()),
        ("source_split", pa.string()),
        ("source_row_index", pa.int64()),
        ("taxonomy_version", pa.string()),
        ("classifier_version", pa.string()),
        ("input_hash", pa.string()),
        ("source_tokens", pa.int64()),
        ("training_loss_tokens", pa.int64()),
        ("judge_provider", pa.string()),
        ("judge_model", pa.string()),
        ("judge_model_revision", pa.string()),
        ("judge_prompt_version", pa.string()),
        ("judge_prompt_tokens", pa.int64()),
        ("judge_completion_tokens", pa.int64()),
        ("judge_response_hash", pa.string()),
        ("judge_error", pa.string()),
    ]
    for column in (_column_for_dimension(dimension) for dimension in dimensions):
        fields.extend(((column, list_string), (f"{column}_status", pa.string())))
    fields.append(("classifications_json", pa.large_string()))
    return pa.schema(fields)


def write_shard(rows: list[dict[str, Any]], output: Path, index: int, output_format: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    dimensions = list(rows[0]["classifications"])
    if any(list(row["classifications"]) != dimensions for row in rows[1:]):
        raise DataFormatError("all rows in a shard must have identical classification dimensions")
    flattened = [_flatten_annotation(row) for row in rows]
    if output_format == "jsonl":
        name = f"part-{index:06d}.jsonl"
        path = output / name
        temporary = output / f".{name}.{os.getpid()}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            for row in flattened:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
    elif output_format == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise DataFormatError(
                "Parquet output requires pyarrow; install the repository requirements"
            ) from exc
        name = f"part-{index:06d}.parquet"
        path = output / name
        temporary = output / f".{name}.{os.getpid()}.tmp"
        table = pa.Table.from_pylist(flattened, schema=_parquet_schema(pa, dimensions))
        pq.write_table(table, temporary, compression="zstd", compression_level=6)
        temporary.replace(path)
    else:
        raise DataFormatError(f"unsupported output format: {output_format}")
    return {"name": name, "rows": len(rows)}
