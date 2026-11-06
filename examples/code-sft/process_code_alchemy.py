"""Process all rehydrated Code Alchemy training subsets; retain completions separately."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
STANDARD = ROOT / "02-standardisation/code-sft"
FILTER = ROOT / "04-decontamination/code-sft"
sys.path[:0] = [str(STANDARD), str(FILTER)]

import pyarrow as pa
import pyarrow.parquet as pq
import dedup
from code_alchemy import CompletionOnly, PLACEHOLDER, REPO, REVISION
from standardise_native import SCHEMA, checked_row, digest
from standardize import SCHEMA as BENCH_SCHEMA, atomic_json, dumps, normalize

SUBSETS = ("code-qa", "code-dev", "code-dialogue", "code-trace", "code-enhance")
PRETRAIN_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.large_string()),
        ("source_dataset", pa.string()),
        ("source_config", pa.string()),
        ("source_revision", pa.string()),
        ("source_record_json", pa.large_string()),
    ]
)
STATE = {}


def init_worker(state=None, exact=None, matcher=None):
    global STATE
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    if state is not None:
        STATE = state
    if exact is not None:
        dedup.EXACT = exact
    if matcher is not None:
        dedup.MATCHER = matcher


def materialize_checked(task):
    result = dedup.materialize(task)
    path = Path(task[3]) / "clean" / task[2]
    if (
        sum(
            batch.num_rows for batch in pq.ParquetFile(path).iter_batches(batch_size=16)
        )
        != result["kept_rows"]
    ):
        raise ValueError("Clean shard readback count mismatch")
    result = dict(result, sha256=digest(path))
    atomic_json(path.with_suffix(".audit.json"), result)
    return result


def native_hash(conversation):
    payload = dedup.compact(dedup.canonical(json.loads(conversation)))
    return hashlib.sha256(("native\0" + payload).encode()).hexdigest()


def fp(index, offset, row_id, subset, hash_value, witness=None, kind=None):
    return dict(
        file_index=index,
        row_index=offset,
        id=row_id,
        source_dataset=subset,
        sample_hash=hash_value,
        benchmark_match=witness,
        match_kind=kind,
        benchmark_similarity=None,
        fuzzy_bands=None,
        fuzzy_size=0,
        fuzzy_guard=None,
    )


def verify_marker(marker):
    saved = json.loads(marker.read_text())
    for name, checksum in saved["sha256"].items():
        if digest(marker.parent / name) != checksum:
            raise ValueError(f"Output checksum mismatch: {marker.parent / name}")
    return saved


def worker(task):
    kind, index, path_text, subset = task
    path = Path(path_text)
    output = Path(STATE["output"])
    directory = (
        output
        / "work"
        / ("reference" if kind == "reference" else subset)
        / f"{abs(index):07d}"
    )
    marker = directory / "_SUCCESS.json"
    if marker.exists():
        return verify_marker(marker)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    fingerprint = directory / "fingerprints.parquet"
    counts = Counter(read=0, native=0, pretraining=0, quarantined=0)
    fingerprints = []
    buffers = [[], []]
    offsets = [0, 0]
    schemas = [SCHEMA, PRETRAIN_SCHEMA]
    writers = []
    paths = [directory / "native.parquet", directory / "pretraining.parquet"]
    if kind != "reference":
        writers = [
            pq.ParquetWriter(p, s, compression="zstd") for p, s in zip(paths, schemas)
        ]

    def flush_payload(slot):
        if buffers[slot]:
            writers[slot].write_table(
                pa.Table.from_pylist(buffers[slot], schema=schemas[slot])
            )
            buffers[slot].clear()

    try:
        with (
            pq.ParquetWriter(
                fingerprint, dedup.FP_SCHEMA, compression="zstd"
            ) as fp_writer,
            gzip.open(
                directory / "rejected.jsonl.gz", "wt", compresslevel=1
            ) as rejected,
        ):
            columns = ["id", "conversation_json"] if kind == "reference" else None
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=16, columns=columns
            ):
                for original in batch.to_pylist():
                    source_index = counts["read"]
                    counts["read"] += 1
                    if kind == "reference":
                        fingerprints.append(
                            fp(
                                -1 - index,
                                source_index,
                                original["id"],
                                "existing_native_reference",
                                native_hash(original["conversation_json"]),
                            )
                        )
                    else:
                        spec = dict(
                            path=str(path), source_config=subset, source_split="train"
                        )
                        try:
                            row = checked_row(
                                original,
                                STATE["entry"],
                                spec,
                                source_index,
                                {},
                                "think-first",
                            )
                            slot = 0
                            hash_value = native_hash(row["conversation_json"])
                        except CompletionOnly:
                            text = original.get("text")
                            if (
                                not isinstance(text, str)
                                or not text.strip()
                                or PLACEHOLDER in text
                            ):
                                counts["quarantined"] += 1
                                rejected.write(
                                    json.dumps(
                                        dict(
                                            row=source_index,
                                            reason="Invalid completion text",
                                            source_record=json.loads(dumps(original)),
                                        ),
                                        ensure_ascii=True,
                                    )
                                    + "\n"
                                )
                                continue
                            slot = 1
                            row = dict(
                                id=hashlib.sha256(
                                    f"{path}\0{source_index}".encode()
                                ).hexdigest(),
                                text=text,
                                source_dataset=REPO,
                                source_config=subset,
                                source_revision=REVISION,
                                source_record_json=dumps(original),
                            )
                            hash_value = hashlib.sha256(
                                ("pretraining\0" + dedup.canonical(text)).encode()
                            ).hexdigest()
                        except Exception as error:
                            counts["quarantined"] += 1
                            rejected.write(
                                json.dumps(
                                    dict(
                                        row=source_index,
                                        reason=f"{type(error).__name__}: {error}",
                                        source_record=json.loads(dumps(original)),
                                    ),
                                    ensure_ascii=True,
                                )
                                + "\n"
                            )
                            continue
                        # Code Alchemy evaluation source blobs are held out across all training subsets.
                        witness = STATE["heldout"].get(original.get("blob_id"))
                        match_kind = "evaluation_blob" if witness else None
                        if not witness:
                            view = row if slot == 0 else dict(prompt=row["text"])
                            witness, match_kind = dedup.contamination(view)
                        buffers[slot].append(row)
                        fingerprints.append(
                            fp(
                                index * 2 + slot,
                                offsets[slot],
                                row["id"],
                                subset + ("/native" if slot == 0 else "/pretraining"),
                                hash_value,
                                witness,
                                match_kind,
                            )
                        )
                        offsets[slot] += 1
                        counts["native" if slot == 0 else "pretraining"] += 1
                        if len(buffers[slot]) >= 16:
                            flush_payload(slot)
                    if len(fingerprints) >= 256:
                        fp_writer.write_table(
                            pa.Table.from_pylist(fingerprints, schema=dedup.FP_SCHEMA)
                        )
                        fingerprints.clear()
            if fingerprints:
                fp_writer.write_table(
                    pa.Table.from_pylist(fingerprints, schema=dedup.FP_SCHEMA)
                )
            if writers:
                flush_payload(0)
                flush_payload(1)
    finally:
        for writer in writers:
            writer.close()
    if counts["read"] != pq.ParquetFile(path).metadata.num_rows:
        raise ValueError("Input row count changed")
    files = []
    if kind != "reference":
        for slot, payload in enumerate(paths):
            if (
                sum(b.num_rows for b in pq.ParquetFile(payload).iter_batches())
                != offsets[slot]
            ):
                raise ValueError("Output readback count mismatch")
            branch = "datasets" if slot == 0 else "pretraining"
            relative = (
                f"{branch}/code_sft/{REPO}/{subset}/{path.stem}/part-000000.parquet"
            )
            files.append(
                dict(
                    index=index * 2 + slot,
                    path=str(payload),
                    relative=relative,
                    rows=offsets[slot],
                )
            )
        if (
            counts["read"]
            != counts["native"] + counts["pretraining"] + counts["quarantined"]
        ):
            raise ValueError("Source row accounting mismatch")
    result = dict(
        kind=kind,
        subset=subset,
        counts=dict(counts),
        fingerprint=str(fingerprint),
        files=files,
        sha256={p.name: digest(p) for p in directory.iterdir() if p.is_file()},
    )
    atomic_json(marker, result)
    return result


def eval_references(source, output):
    heldout = {}
    rows = []
    missing = 0
    for subset in ("dev-eval", "trace-eval"):
        for path in sorted((source / subset / "test").glob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                for index, row in enumerate(batch.to_pylist()):
                    if row.get("blob_id"):
                        heldout[row["blob_id"]] = dedup.compact(
                            dict(dataset=REPO, config=subset, blob_id=row["blob_id"])
                        )
                    prompt = row.get("prompt")
                    if not isinstance(prompt, str) or not prompt.strip():
                        missing += 1
                        continue
                    if PLACEHOLDER in prompt:
                        if row.get("source_code_status") != "hydrated" or not row.get(
                            "source_code"
                        ):
                            missing += 1
                            continue
                        prompt = prompt.replace(PLACEHOLDER, row["source_code"].strip())
                    rows.append(
                        normalize(
                            dict(prompt=prompt, benchmark_source_file=str(path)),
                            repo=REPO,
                            revision=REVISION,
                            config=subset,
                            split="test",
                            index=len(rows),
                            category="benchmarks",
                        )
                    )
    if not rows or not heldout:
        raise ValueError(
            "Both evaluation references and source-blob holdouts are required"
        )
    path = output / "code-alchemy-evaluation.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=BENCH_SCHEMA), path, compression="zstd"
    )
    return (
        path,
        heldout,
        dict(
            prompt_rows=len(rows),
            heldout_blobs=len(heldout),
            unavailable_prompts=missing,
        ),
    )


def run(args):
    global STATE
    source = args.input.resolve()
    output = args.output.resolve()
    if (
        source == output
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("Separate source/output trees required")
    success = json.loads((source / "_SUCCESS.json").read_text())
    if success.get("revision") != REVISION or not success.get("validated"):
        raise ValueError(
            "A validated rehydrated source at the pinned revision is required"
        )
    tasks = []
    original_rows = {}
    inputs = []
    for subset in SUBSETS:
        files = sorted((source / subset / "train").glob("*.parquet"))
        if not files:
            raise ValueError(f"Missing training subset: {subset}")
        if args.limit_files:
            files = files[: args.limit_files]
        original_rows[subset] = sum(pq.ParquetFile(p).metadata.num_rows for p in files)
        for path in files:
            tasks.append(("training", len(tasks), str(path), subset))
    references = (
        sorted((args.reference.resolve() / "datasets").rglob("*.parquet"))
        if args.reference
        else []
    )
    if args.reference and not references:
        raise ValueError("No existing native reference shards")
    reference_expected = None
    if args.reference:
        reference_success = json.loads((args.reference / "_SUCCESS.json").read_text())
        reference_expected = reference_success.get(
            "rows", reference_success.get("counts", {}).get("kept")
        )
    for task in tasks:
        p = Path(task[2])
        s = p.stat()
        inputs.append(dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns))
    reference_tasks = [
        ("reference", i, str(p), "existing") for i, p in enumerate(references)
    ]
    benchmark_files = dedup.completed_files(args.benchmarks.resolve(), "benchmarks")
    benchmark_root = dedup.benchmark_directory(args.benchmarks.resolve())
    for entry in json.loads(args.benchmark_lock.read_text())["datasets"]:
        marker = benchmark_root / entry["repo"] / "_SUCCESS.json"
        if (
            not marker.exists()
            or json.loads(marker.read_text()).get("revision") != entry["revision"]
        ):
            raise ValueError(
                f"Incomplete or unpinned benchmark reference: {entry['repo']}"
            )
    code_files = [
        Path(__file__),
        *STANDARD.rglob("*.py"),
        *FILTER.glob("*.py"),
        *STANDARD.glob("*.json"),
    ]
    code_files = [p for p in code_files if "tests" not in p.parts]
    config = dict(
        revision=REVISION,
        source=str(source),
        source_success=digest(source / "_SUCCESS.json"),
        source_materialization=digest(source / "_MATERIALIZED.json"),
        inputs=inputs,
        native_library=importlib.metadata.distribution("apertus-common").read_text(
            "direct_url.json"
        ),
        benchmark_lock=digest(args.benchmark_lock),
        reference_success=digest(args.reference / "_SUCCESS.json")
        if args.reference
        else None,
        reference=[
            dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in references
        ],
        benchmark_inputs=[dict(path=str(p), sha256=digest(p)) for p in benchmark_files],
        code={str(p.relative_to(ROOT)): digest(p) for p in code_files},
        limit_files=args.limit_files,
        policy="native SFT where released turns exist; completion-only pretraining separate; exact native/text dedup against existing reference; benchmark equality/containment and Code Alchemy evaluation blob holdout",
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "run.json").exists() and json.loads(
            (output / "run.json").read_text()
        ) != config:
            raise ValueError("Run identity changed; choose a fresh output")
        atomic_json(output / "run.json", config)
        if (output / "_SUCCESS.json").exists():
            return json.loads((output / "_SUCCESS.json").read_text())
        evaluation, heldout, coverage = eval_references(source, output)
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        coverage["registered_benchmarks"] = dedup.build_matcher(
            [*benchmark_files, evaluation], 80, False
        )
        STATE = dict(
            output=str(output),
            heldout=heldout,
            entry=dict(
                repo=REPO,
                revision=REVISION,
                category="code_sft",
                license="see-notice",
                license_status="unknown",
                license_notes="Code Alchemy NOTICE; source rights remain source-specific",
                synthetic_status="synthetic",
                synthetic_notes="Code Alchemy release",
                metadata_source=f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
            ),
        )
        results = []
        totals = Counter()
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=init_worker,
            initargs=(STATE, dedup.EXACT, dedup.MATCHER),
        ) as pool:
            for i, result in enumerate(
                pool.map(worker, [*tasks, *reference_tasks], chunksize=1), 1
            ):
                results.append(result)
                if result["kind"] == "training":
                    totals.update(result["counts"])
                if i % 10 == 0 or i == len(tasks) + len(reference_tasks):
                    progress = dict(
                        tasks_done=i,
                        tasks_total=len(tasks) + len(reference_tasks),
                        counts=dict(totals),
                    )
                    atomic_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        if totals["read"] != sum(original_rows.values()):
            raise ValueError("Collection row count mismatch")
        reference_read = sum(
            r["counts"]["read"] for r in results if r["kind"] == "reference"
        )
        if reference_expected is not None and reference_read != reference_expected:
            raise ValueError("Existing native reference row count mismatch")
        filtered = output / "decontaminated"
        (filtered / "_work").mkdir(parents=True, exist_ok=True)
        fingerprints = [r["fingerprint"] for r in results]
        _, all_counts = dedup.classify(
            fingerprints, filtered, args.threads, args.memory_limit
        )
        files = [f for result in results for f in result["files"]]
        materialize_tasks = [
            (f["index"], f["path"], f["relative"], str(filtered), 16) for f in files
        ]
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context, initializer=init_worker
        ) as pool:
            written = list(pool.map(materialize_checked, materialize_tasks))
        import duckdb

        db = duckdb.connect(
            str(filtered / "_work/index.duckdb"),
            read_only=True,
            config={"threads": args.threads},
        )
        try:
            new_counts = dict(
                db.execute(
                    "SELECT decision,count(*) FROM decisions WHERE file_index>=0 GROUP BY decision"
                ).fetchall()
            )
            by_subset = [
                dict(subset=s, decision=d, rows=n)
                for s, d, n in db.execute(
                    "SELECT source_dataset,decision,count(*) FROM decisions WHERE file_index>=0 GROUP BY ALL ORDER BY ALL"
                ).fetchall()
            ]
        finally:
            db.close()
        if sum(w["kept_rows"] for w in written) != new_counts.get("keep", 0):
            raise ValueError("Filtered output count mismatch")
        report = dict(
            ready=not bool(args.limit_files),
            source_rows=original_rows,
            standardisation=dict(totals),
            decisions=new_counts,
            by_subset=by_subset,
            evaluation=coverage,
            existing_reference_shards=len(references),
            existing_reference_rows=reference_read,
            output_shards=len(written),
            training_path=str(filtered / "clean/datasets"),
            pretraining_path=str(filtered / "clean/pretraining"),
        )
        atomic_json(output / "summary.json", report)
        if not args.limit_files:
            atomic_json(output / "_SUCCESS.json", report)
        print(json.dumps(report), flush=True)
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmarks", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument(
        "--benchmark-lock", type=Path, default=FILTER / "benchmarks.lock.json"
    )
    p.add_argument("--workers", type=int, default=96)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--memory-limit", default="600GB")
    p.add_argument("--limit-files", type=int)
    a = p.parse_args()
    if (
        a.workers < 1
        or a.threads < 1
        or (a.limit_files is not None and a.limit_files < 1)
    ):
        p.error("Worker/thread/file limits must be positive")
    run(a)


if __name__ == "__main__":
    main()
