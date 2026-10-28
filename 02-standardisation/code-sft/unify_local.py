"""Unify local JSONL or Parquet columns without discarding the source record."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from standardize import atomic_json, dumps
from unified import SCHEMA, normalize_v2


def rename(row, mapping):
    result = dict(row)
    for source, target in mapping.items():
        if source not in row:
            continue
        if target in result and result[target] != row[source]:
            raise ValueError(f"Conflicting values for {source!r} -> {target!r}")
        result[target] = row[source]
    return result


def records(path):
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    else:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def unify(inputs, output, repo, revision, mapping, category="code_sft"):
    if len(repo.split("/")) != 2 or any(
        part in {"", ".", ".."} for part in repo.split("/")
    ):
        raise ValueError("repo must be an owner/name identifier")
    if not inputs or any(not path.is_file() for path in inputs):
        raise ValueError("All input files must exist")
    if any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()
    ):
        raise ValueError("Column map must contain string source and target names")
    entry = dict(
        repo=repo,
        revision=revision,
        category=category,
        code_policy="code_source",
        license="unknown",
        license_status="unknown",
        license_notes="Not declared",
        synthetic_status="unknown",
        synthetic_notes="Not declared",
        metadata_source="local input",
    )
    directory = output / category / repo
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "source.json", entry)
    count = 0
    for path in inputs:
        spec = dict(
            path=str(path.resolve()), source_config="local", source_split="train"
        )
        partition = directory / (
            "file=" + hashlib.sha256(spec["path"].encode()).hexdigest()[:16]
        )
        partition.mkdir()
        pending = []
        with pq.ParquetWriter(
            partition / "part-00000.parquet", SCHEMA, compression="zstd"
        ) as writer:
            for index, original in enumerate(records(path)):
                if not isinstance(original, dict):
                    raise ValueError("Each input row must be an object")
                row = normalize_v2(rename(original, mapping), entry, spec, index)
                row["source_record_json"] = dumps(original)
                pending.append(row)
                count += 1
                if len(pending) == 256:
                    writer.write_table(pa.Table.from_pylist(pending, schema=SCHEMA))
                    pending.clear()
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=SCHEMA))
    atomic_json(
        directory / "_SUCCESS.json",
        dict(rows=count, revision=revision, files=len(inputs)),
    )
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--column-map", type=Path)
    parser.add_argument(
        "--category", choices=["code_sft", "code_agentic"], default="code_sft"
    )
    args = parser.parse_args()
    mapping = json.loads(args.column_map.read_text()) if args.column_map else {}
    if not isinstance(mapping, dict):
        parser.error("--column-map must be a JSON object")
    print(
        f"Unified {unify(args.input, args.output, args.repo, args.revision, mapping, args.category)} rows"
    )


if __name__ == "__main__":
    main()
