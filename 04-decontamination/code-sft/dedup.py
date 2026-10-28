#!/usr/bin/env python3
"""Parallel exact SFT deduplication, then public benchmark decontamination."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import re
import time
import importlib.metadata
from pathlib import Path

import ahocorasick
import pyarrow as pa
import pyarrow.parquet as pq

from standardize import atomic_json
from sft_selection import user_exclusions

VERSION = 1
MATCHER = None
EXACT = {}
FUZZY_ENABLED = False
FP_SCHEMA = pa.schema(
    [
        ("file_index", pa.int64()),
        ("row_index", pa.int64()),
        ("id", pa.string()),
        ("source_dataset", pa.string()),
        ("sample_hash", pa.string()),
        ("benchmark_match", pa.string()),
        ("match_kind", pa.string()),
        ("benchmark_similarity", pa.float64()),
        ("fuzzy_bands", pa.list_(pa.binary())),
        ("fuzzy_size", pa.int64()),
        ("fuzzy_guard", pa.string()),
    ]
)


def compact(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canonical(value):
    """Preserve code whitespace and case; normalize only platform line endings."""
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    return value


def parsed(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def messages(row):
    raw = parsed(row.get("messages_json"))
    if not isinstance(raw, list):
        return raw
    result = []
    aliases = {"human": "user", "gpt": "assistant", "bot": "assistant"}
    for item in raw:
        if not isinstance(item, dict):
            result.append(item)
            continue
        item = dict(item)
        for key in ("from", "speaker", "author"):
            if key in item:
                item.setdefault("role", item[key])
                if aliases.get(item[key], item[key]) == aliases.get(
                    item.get("role"), item.get("role")
                ):
                    item.pop(key)
        item["role"] = aliases.get(item.get("role"), item.get("role"))
        for key in ("value", "text", "message", "observation"):
            if key in item:
                item.setdefault("content", item[key])
                if item[key] == item.get("content"):
                    item.pop(key)
        result.append(item)
    return result


def sample_hash(row):
    turns = messages(row)
    if not turns:
        turns = [
            {"role": role, "content": row[key]}
            for role, key in [("user", "prompt"), ("assistant", "response")]
            if row.get(key)
        ]
    # Preserve explicit prompt/response if they carry information absent from messages.
    represented = (
        {compact(m.get("content")) for m in turns if isinstance(m, dict)}
        if isinstance(turns, list)
        else set()
    )
    extra = {
        k: row[k]
        for k in ("prompt", "response")
        if row.get(k) and compact(row[k]) not in represented
    }
    payload = {
        "messages": turns,
        **extra,
        "reasoning": row.get("reasoning") or None,
        "tools": parsed(row.get("tools_json")),
        "tests": parsed(row.get("tests_json")),
    }
    if (
        not turns
        and not extra
        and not any(payload[k] for k in ("reasoning", "tools", "tests"))
    ):
        # Never merge opaque, unadapted examples just because their training views are empty.
        payload["opaque_metadata"] = parsed(row.get("metadata_json"))
    return hashlib.sha256(compact(canonical(payload)).encode()).hexdigest()


def match_text(text):
    # Only the contamination view folds whitespace. Original text and exact hashes do not.
    return " ".join(text.split())


def string_leaves(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from string_leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from string_leaves(item)


def build_matcher(files, min_chars, include_solutions=False):
    global MATCHER, EXACT
    EXACT = {}
    patterns = {}
    rows = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=512,
            columns=[
                "id",
                "source_dataset",
                "source_config",
                "source_split",
                "category",
                "prompt",
                "response",
                "metadata_json",
            ],
        ):
            for row in batch.to_pylist():
                rows += 1
                if row["category"] != "benchmarks":
                    raise ValueError(f"Non-benchmark reference row in {path}")
                metadata = parsed(row.get("metadata_json")) or {}
                texts = metadata.get("benchmark_prompts", []) + [row.get("prompt")]
                if include_solutions:
                    texts.append(row.get("response"))
                for text in texts:
                    if not isinstance(text, str) or not text.strip():
                        continue
                    text = match_text(text)
                    reference = compact(
                        {
                            "dataset": row["source_dataset"],
                            "id": row["id"],
                            "config": row["source_config"],
                            "split": row["source_split"],
                            "file": metadata.get(
                                "benchmark_source_file", row.get("source_file")
                            ),
                        }
                    )
                    # Store one deterministic witness; overlapping benchmark variants remain on disk.
                    EXACT.setdefault(text, reference)
                    if len(text) >= min_chars:
                        patterns.setdefault(text, reference)
    if not rows or not EXACT:
        raise ValueError("No usable benchmark prompts")
    MATCHER = ahocorasick.Automaton()
    for pattern, reference in sorted(patterns.items()):
        MATCHER.add_word(pattern, reference)
    if patterns:
        MATCHER.make_automaton()
    else:
        MATCHER = None
    return {
        "rows": rows,
        "unique_texts": len(EXACT),
        "substring_patterns": len(patterns),
    }


def contamination(row):
    # Scan all conversation roles and reasoning, not provenance metadata or generic test boilerplate.
    texts = [row.get("prompt"), row.get("response"), row.get("reasoning")]
    texts.extend(string_leaves(messages(row)))
    if row.get("conversation_json"):
        native = json.loads(row["conversation_json"])
        texts.extend(
            item["payload"]
            for item in native["items"]
            if item.get("type")
            in {"user", "reply", "think", "call", "result", "document", "host"}
            and isinstance(item.get("payload"), str)
        )
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        text = match_text(text)
        if text in EXACT:
            return EXACT[text], "exact_text"
        if MATCHER is not None:
            for _, reference in MATCHER.iter(text):
                return reference, "contained_text"
    return None, None


def fingerprint(task):
    index, path, target, signature, batch_size = task
    target = Path(target)
    marker = target.with_suffix(".json")
    if (
        target.exists()
        and marker.exists()
        and json.loads(marker.read_text()).get("signature") == signature
    ):
        return str(target)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    temp = target.with_suffix(".tmp")
    count = 0
    with pq.ParquetWriter(temp, FP_SCHEMA, compression="zstd") as writer:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=batch_size,
            columns=[
                "id",
                "source_dataset",
                "category",
                "messages_json",
                "prompt",
                "response",
                "reasoning",
                "tools_json",
                "tests_json",
                "metadata_json",
            ]
            + (
                ["conversation_json"]
                if "conversation_json" in pq.ParquetFile(path).schema_arrow.names
                else []
            ),
        ):
            result = []
            for row in batch.to_pylist():
                if row["category"] == "benchmarks":
                    raise ValueError("Benchmark row found in SFT input")
                reference, kind = contamination(row)
                similarity = None
                fuzzy_bands, fuzzy_size, fuzzy_guard = [], 0, None
                if FUZZY_ENABLED:
                    from fuzzy import signature as fuzzy_signature, benchmark_match

                    fuzzy_bands, fuzzy_size, fuzzy_guard = fuzzy_signature(row)
                    if reference is None:
                        reference, kind, similarity = benchmark_match(row)
                result.append(
                    dict(
                        file_index=index,
                        row_index=count,
                        id=row["id"],
                        source_dataset=row["source_dataset"],
                        sample_hash=sample_hash(row),
                        benchmark_match=reference,
                        match_kind=kind,
                        benchmark_similarity=similarity,
                        fuzzy_bands=fuzzy_bands,
                        fuzzy_size=fuzzy_size,
                        fuzzy_guard=fuzzy_guard,
                    )
                )
                count += 1
            writer.write_table(pa.Table.from_pylist(result, schema=FP_SCHEMA))
    temp.replace(target)
    atomic_json(marker, {"signature": signature, "rows": count})
    return str(target)


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def classify(
    fingerprints,
    output,
    threads,
    memory_limit,
    files=None,
    fuzzy_threshold=0.9,
    max_candidates=50_000_000,
):
    import duckdb

    duckdb.close()  # Release the unused machine-wide default pool; use the bounded connection below.
    db = duckdb.connect(
        str(output / "_work" / "index.duckdb"), config={"threads": threads}
    )
    try:
        db.execute(f"SET threads={int(threads)}")
        db.execute("SET memory_limit=" + sql_string(memory_limit))
        db.execute("SET temp_directory=" + sql_string(output / "_work" / "spill"))
        db.read_parquet(fingerprints).create_view("fingerprints")
        db.execute("""CREATE OR REPLACE TABLE decisions AS
          WITH ranked AS (
            SELECT *, row_number() OVER w AS duplicate_rank,
              first_value(file_index) OVER w AS keeper_file_index,
              first_value(row_index) OVER w AS keeper_row_index,
              first_value(id) OVER w AS keeper_id,
              bool_or(benchmark_match IS NOT NULL) OVER (PARTITION BY sample_hash) AS group_benchmark_match
            FROM fingerprints
            WINDOW w AS (PARTITION BY sample_hash ORDER BY file_index, row_index)
          ) SELECT *, CASE WHEN duplicate_rank > 1 THEN 'duplicate'
                           WHEN group_benchmark_match THEN 'benchmark'
                           ELSE 'keep' END AS decision FROM ranked""")
        if files is not None:
            from fuzzy import apply_training

            fuzzy_report = apply_training(
                db, files, output, fuzzy_threshold, max_candidates, threads
            )
            atomic_json(output / "fuzzy_report.json", fuzzy_report)
        decisions = output / "decisions.parquet"
        db.execute(
            "COPY (SELECT * FROM decisions ORDER BY file_index, row_index) TO "
            + sql_string(str(decisions) + ".tmp")
            + " (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        Path(str(decisions) + ".tmp").replace(decisions)
        # Partition compact keep offsets once, avoiding a full index scan per output shard.
        db.execute(
            "COPY (SELECT file_index, row_index FROM decisions WHERE decision='keep') TO "
            + sql_string(output / "_work" / "keep")
            + " (FORMAT PARQUET, PARTITION_BY (file_index), OVERWRITE_OR_IGNORE)"
        )
        summary = [
            dict(zip(("dataset", "decision", "rows"), record))
            for record in db.execute(
                "SELECT source_dataset, decision, count(*) FROM decisions GROUP BY ALL ORDER BY ALL"
            ).fetchall()
        ]
        totals = dict(
            db.execute(
                "SELECT decision, count(*) FROM decisions GROUP BY decision"
            ).fetchall()
        )
        return summary, totals
    finally:
        db.close()


def materialize(task):
    index, path, relative, output, batch_size = task
    output = Path(output)
    target = output / "clean" / relative
    marker = target.with_suffix(".success.json")
    if marker.exists() and target.exists():
        return json.loads(marker.read_text())
    keep = set()
    for mask in (output / "_work" / "keep" / f"file_index={index}").glob("*.parquet"):
        keep.update(
            pq.ParquetFile(mask)
            .read(columns=["row_index"])
            .column("row_index")
            .to_pylist()
        )
    source = pq.ParquetFile(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    offset = count = 0
    with pq.ParquetWriter(
        temp, source.schema_arrow, compression="zstd", compression_level=1
    ) as writer:
        for batch in source.iter_batches(batch_size=batch_size):
            selected = [i for i in range(batch.num_rows) if offset + i in keep]
            if selected:
                writer.write_table(
                    pa.Table.from_batches([batch]).take(
                        pa.array(selected, type=pa.int64())
                    )
                )
                count += len(selected)
            offset += batch.num_rows
    temp.replace(target)
    result = {"input_rows": offset, "kept_rows": count, "file": relative}
    atomic_json(marker, result)
    return result


def snapshot(paths):
    return [
        {"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in paths
    ]


def benchmark_directory(root):
    nested = root / "benchmarks"
    return nested if nested.is_dir() else root


def completed_files(root, category, excluded_repos=()):
    files = []
    category_root = (
        benchmark_directory(root) if category == "benchmarks" else root / category
    )
    for source in sorted(category_root.glob("*/*/source.json")):
        repo = source.parent
        if str(repo.relative_to(category_root)) in excluded_repos:
            continue
        if not (repo / "_SUCCESS.json").exists() or (repo / "_FAILED.json").exists():
            raise ValueError(f"Incomplete input dataset: {repo}")
        state = json.loads((repo / "_SUCCESS.json").read_text())
        shards = sorted(repo.rglob("part-*.parquet"))
        if not shards:
            if state.get("rows") == 0:
                continue
            raise ValueError(f"No shards: {repo}")
        if (
            "rows" in state
            and sum(pq.ParquetFile(p).metadata.num_rows for p in shards)
            != state["rows"]
        ):
            raise ValueError(
                f"Shard row count disagrees with completion marker: {repo}"
            )
        files.extend(shards)
    if not files:
        raise ValueError(f"No completed datasets under {category_root}")
    return files


def unavailable_repos(root, categories, allowed, excluded_repos=()):
    """Explicitly report named unavailable sources; completed sources are never excluded."""
    catalog_path = root / "catalog.json"
    if not catalog_path.exists():
        if allowed:
            raise ValueError(
                "An input catalog is required for unavailable-source exceptions"
            )
        return []
    entries = [
        e
        for e in json.loads(catalog_path.read_text())["datasets"]
        if e["category"] in categories and e["repo"] not in excluded_repos
    ]
    unknown = set(allowed) - {e["repo"] for e in entries}
    if unknown:
        raise ValueError(
            f"Unavailable-source exceptions not in input catalog: {sorted(unknown)}"
        )
    excluded = []
    for entry in entries:
        repo = root / entry["category"] / entry["repo"]
        failure = repo / "_FAILED.json"
        if (repo / "_SUCCESS.json").exists() and not failure.exists():
            continue
        if entry["repo"] not in allowed:
            raise ValueError(f"Missing completed training dataset: {entry['repo']}")
        excluded.append(
            {
                "repo": entry["repo"],
                "category": entry["category"],
                "reason": "Explicitly allowed unavailable source after ingestion retry",
                "failure": json.loads(failure.read_text())
                if failure.exists()
                else "No completion marker",
            }
        )
    return excluded


def main():
    global FUZZY_ENABLED
    pa.set_cpu_count(1)
    pa.set_io_thread_count(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--exclusions", type=Path, help="Optional explicit repository-exclusion JSON"
    )
    parser.add_argument(
        "--benchmarks",
        type=Path,
        help="Benchmark directory or parent containing benchmarks/; defaults to --input",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["code_sft"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--memory-limit",
        default="16GB",
        help="DuckDB memory cap, separate from scan worker RAM",
    )
    parser.add_argument("--min-benchmark-chars", type=int, default=80)
    parser.add_argument("--include-benchmark-solutions", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--fuzzy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.9)
    parser.add_argument("--benchmark-fuzzy-threshold", type=float, default=0.8)
    parser.add_argument("--max-fuzzy-candidates", type=int, default=50_000_000)
    parser.add_argument(
        "--allow-unavailable-repo",
        action="append",
        default=[],
        help="Allow this named training source to remain unavailable; record its omission. Completed sources are included.",
    )
    args = parser.parse_args()
    FUZZY_ENABLED = args.fuzzy
    if (
        not 0 < args.fuzzy_threshold <= 1
        or not 0 < args.benchmark_fuzzy_threshold <= 1
        or args.max_fuzzy_candidates < 1
    ):
        parser.error(
            "similarity thresholds must be in (0,1], and candidate budget positive"
        )
    if args.workers < 1 or args.batch_size < 1 or args.min_benchmark_chars < 1:
        parser.error("workers, batch-size and min-benchmark-chars must be positive")
    if "benchmarks" in args.categories:
        parser.error("benchmarks cannot be a training category")
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    benchmark_root = (args.benchmarks or args.input).resolve()
    if (
        args.output == args.input
        or args.output.is_relative_to(args.input)
        or args.input.is_relative_to(args.output)
    ):
        parser.error("output must be separate from the input tree")
    if (
        args.output == benchmark_root
        or args.output.is_relative_to(benchmark_root)
        or benchmark_root.is_relative_to(args.output)
    ):
        parser.error("output must be separate from the benchmark tree")
    started = time.monotonic()
    explicit_exclusions = user_exclusions(args.exclusions)
    user_excluded_repos = {entry["repo"] for entry in explicit_exclusions}
    unavailable = unavailable_repos(
        args.input, args.categories, args.allow_unavailable_repo, user_excluded_repos
    )
    excluded_repos = user_excluded_repos | {entry["repo"] for entry in unavailable}
    for entry in explicit_exclusions:
        print("USER_EXCLUDED_SOURCE " + compact(entry), flush=True)
    for entry in unavailable:
        print("UNAVAILABLE_SOURCE " + compact(entry), flush=True)
    files = sorted(
        set(
            p
            for c in args.categories
            for p in completed_files(args.input, c, excluded_repos)
        )
    )
    benchmarks = completed_files(benchmark_root, "benchmarks")
    # Refuse a silently partial benchmark download against the registered manifest.
    lock = Path(__file__).with_name("benchmarks.lock.json")
    if lock.exists():
        for entry in json.loads(lock.read_text())["datasets"]:
            marker = (
                benchmark_directory(benchmark_root) / entry["repo"] / "_SUCCESS.json"
            )
            if not marker.exists():
                raise ValueError(f"Missing benchmark: {entry['repo']}")
            if json.loads(marker.read_text())["revision"] != entry["revision"]:
                raise ValueError(
                    f"Benchmark revision differs from the lock: {entry['repo']}"
                )
    run = {
        "version": VERSION,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fuzzy_code_sha256": hashlib.sha256(
            Path(__file__).with_name("fuzzy.py").read_bytes()
        ).hexdigest(),
        "rensa_version": importlib.metadata.version("rensa"),
        "fuzzy": args.fuzzy,
        "fuzzy_threshold": args.fuzzy_threshold,
        "benchmark_fuzzy_threshold": args.benchmark_fuzzy_threshold,
        "input": snapshot(files),
        "benchmarks": snapshot(benchmarks),
        "unavailable_sources": unavailable,
        "user_excluded_sources": explicit_exclusions,
        "min_benchmark_chars": args.min_benchmark_chars,
        "solutions": args.include_benchmark_solutions,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != run:
        raise ValueError(
            "Inputs or matching settings changed; use a fresh output directory"
        )
    atomic_json(manifest, run)
    work = args.output / "_work" / "fingerprints"
    work.mkdir(parents=True, exist_ok=True)
    # A single writer per output; interrupted processes release the advisory lock automatically.
    import fcntl

    with (args.output / "_run.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        coverage = build_matcher(
            benchmarks, args.min_benchmark_chars, args.include_benchmark_solutions
        )
        if args.fuzzy:
            from fuzzy import build_benchmarks

            coverage["fuzzy_patterns"] = build_benchmarks(
                EXACT, args.benchmark_fuzzy_threshold
            )
        print("BENCHMARK_INDEX " + compact(coverage), flush=True)
        indexed_at = time.monotonic()
        signature = hashlib.sha256(compact(run).encode()).hexdigest()
        tasks = [
            (i, str(p), str(work / f"{i:07d}.parquet"), signature, args.batch_size)
            for i, p in enumerate(files)
        ]
        context = multiprocessing.get_context(
            "fork"
        )  # Linux HPC: share the read-only automaton via copy-on-write.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context
        ) as pool:
            fingerprints = []
            for i, result in enumerate(pool.map(fingerprint, tasks), 1):
                fingerprints.append(result)
                if i % 25 == 0 or i == len(tasks):
                    print(f"SCAN {i}/{len(tasks)} shards", flush=True)
        if snapshot(files) != run["input"] or snapshot(benchmarks) != run["benchmarks"]:
            raise ValueError(
                "Input files changed during scan; use a fresh output directory"
            )
        scanned_at = time.monotonic()
        summary, totals = classify(
            fingerprints,
            args.output,
            args.workers,
            args.memory_limit,
            files if args.fuzzy else None,
            args.fuzzy_threshold,
            args.max_fuzzy_candidates,
        )
        report = {
            "policy": "global exact samples; optional verified fuzzy samples; then benchmark equality/containment and optional fuzzy prompts",
            "benchmark_coverage": coverage,
            "totals": totals,
            "by_dataset": summary,
            "unavailable_sources": unavailable,
            "user_excluded_sources": explicit_exclusions,
            "all_requested_sources_included": not unavailable,
            "input_rows": sum(totals.values()),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "timing_seconds": {
                "benchmark_index": round(indexed_at - started, 2),
                "scan": round(scanned_at - indexed_at, 2),
                "classification": round(time.monotonic() - scanned_at, 2),
            },
            "limitations": "LSH may miss similar pairs. Fuzzy checks use 5-token Jaccard, minimum 30 tokens; no semantic guarantee. Fuzzy benchmark checks compare whole text fields, so long wrappers may reduce recall. One benchmark witness per row.",
            "materialized": False,
        }
        atomic_json(args.output / "report.json", report)
        print("DECISIONS " + compact(totals), flush=True)
        if not args.report_only:
            tasks = [
                (
                    i,
                    str(p),
                    str(p.relative_to(args.input)),
                    str(args.output),
                    args.batch_size,
                )
                for i, p in enumerate(files)
            ]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=context
            ) as pool:
                written = list(pool.map(materialize, tasks))
            if sum(w["kept_rows"] for w in written) != totals.get("keep", 0):
                raise ValueError("Materialized row count disagrees with decisions")
            report["materialized"] = True
            report["elapsed_seconds"] = round(time.monotonic() - started, 2)
            atomic_json(args.output / "report.json", report)
            atomic_json(args.output / "_SUCCESS.json", report)


if __name__ == "__main__":
    main()
