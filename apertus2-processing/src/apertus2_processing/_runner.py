"""Spawn workers, process bounded batches, publish complete shards atomically."""

import fcntl
import json
import multiprocessing
import os
import shutil
import tempfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from apertus_common import Conversation, TokenizerError, checkers_for, load_encoding
from pydantic import ValidationError

from ._context import decode_context
from ._guards import verify_inputs, verify_runtime
from ._input import iter_rows
from ._output import TokenWriter
from ._report import Report, findings, structural
from ._util import atomic_json, inventory, read_json, verify_files

_STATE = {}


def initialize(config, run_dir, batch_rows, batch_bytes, tokenizer_threads):
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenizer_threads > 1 else "false"
    os.environ["RAYON_NUM_THREADS"] = str(tokenizer_threads)
    verify_runtime(config)
    _STATE.clear()
    _STATE.update(config=config, run_dir=run_dir, batch_rows=batch_rows, batch_bytes=batch_bytes)
    _STATE["checkers"] = checkers_for("full", allow_loss_weights=config["allow_loss_weights"])
    _STATE["context"] = decode_context(config.get("context"))
    _STATE["encoding"] = (
        load_encoding(config["tokenizer"]) if config["operation"] == "tokenize" else None
    )


@contextmanager
def shard_lock(run_dir, index):
    locks = Path(run_dir) / "locks"
    locks.mkdir(exist_ok=True)
    with (locks / f"{index:06d}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"shard {index} is already running") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def inspect_row(row):
    location, raw, context_raw, wrapper_error = row
    if wrapper_error:
        return location, None, [structural(wrapper_error)]
    try:
        value = Conversation.from_json(raw)
    except (ValidationError, UnicodeError) as error:
        return location, None, [structural(error)]
    try:
        context = decode_context(context_raw) if context_raw is not None else _STATE["context"]
    except (ValueError, TypeError, UnicodeError) as error:
        return location, None, [structural(error, "context/invalid")]
    config = _STATE["config"]
    issues = []
    if config["check"] == "full":
        try:
            report = value.check(checkers=_STATE["checkers"], context=context)
        except ValueError as error:
            # These are the library's documented invalid-evidence conditions.
            if str(error) not in {
                "conflicting system configuration in checking context",
                "host closure evidence must identify a retained host item",
            }:
                raise
            return location, None, [structural(error, "context/invalid")]
        issues = findings(report)
        if not (report.ok and report.complete):
            return location, None, issues
    return location, value, issues


def process_batch(batch, report, sink, writer):
    inspected = [inspect_row(row) for row in batch]
    good = [(location, value, issues) for location, value, issues in inspected if value is not None]
    encoded = {}
    if writer is not None and good:
        encoding = _STATE["encoding"]
        kwargs = {"return_loss_weights": _STATE["config"]["emit_loss_weights"]}
        try:
            results = encoding.encode_conversations([value for _, value, _ in good], **kwargs)
            encoded = {
                id(value): result for (_, value, _), result in zip(good, results, strict=True)
            }
        except TokenizerError:
            # Preserve record attribution when an isolated input fails encoding.
            for _, value, issues in good:
                try:
                    encoded[id(value)] = encoding.encode_conversations([value], **kwargs)[0]
                except TokenizerError as error:
                    issues.append(structural(error, "encoding/invalid"))
    output = []
    for location, value, issues in inspected:
        accepted = value is not None and (writer is None or id(value) in encoded)
        report.add(location, issues, sink, accepted=accepted)
        if writer is not None and accepted:
            result = encoded[id(value)]
            token_ids = result.token_ids if _STATE["config"]["emit_loss_weights"] else result
            record = {**location, "token_ids": token_ids, "token_count": len(token_ids)}
            if _STATE["config"]["emit_loss_weights"]:
                if result.loss_weights is None or len(result.loss_weights) != len(token_ids):
                    raise ValueError("encoder returned mismatched loss weights")
                record["loss_weights"] = result.loss_weights
            output.append(record)
    if writer is not None:
        writer.write(output)


def validate_shard(run_dir, shard, fingerprint, operation):
    directory = Path(run_dir) / "shards" / f"{shard['index']:06d}"
    manifest = read_json(directory / "manifest.json")
    if (
        manifest["fingerprint"] != fingerprint
        or manifest["plan"] != shard
        or manifest["index"] != shard["index"]
    ):
        raise ValueError(f"shard {shard['index']} identity mismatch")
    if manifest["summary"]["rows"].get("processed", 0) != shard["end"] - shard["start"]:
        raise ValueError(f"shard {shard['index']} has incomplete row coverage")
    verify_files(directory, manifest["files"])
    if read_json(directory / "summary.json") != manifest["summary"]:
        raise ValueError("shard manifest disagrees with checksum-protected summary")
    file_paths = {entry["path"] for entry in manifest["files"]}
    token_paths = [entry["path"] for entry in manifest["token_files"]]
    inventory_tokens = {path for path in file_paths if Path(path).suffix in {".arrow", ".parquet"}}
    if len(set(token_paths)) != len(token_paths) or set(token_paths) != inventory_tokens:
        raise ValueError("token descriptors must cover each verified token file exactly once")
    for entry in manifest["token_files"]:
        if entry["path"] not in file_paths or Path(entry["path"]).name != entry["path"]:
            raise ValueError("token file is not a verified shard file")
    if operation == "tokenize":
        import pyarrow as pa
        import pyarrow.parquet as pq
        from pyarrow import ipc

        total = 0
        for entry in manifest["token_files"]:
            path = directory / entry["path"]
            count = 0
            if path.suffix == ".parquet":
                batches = pq.ParquetFile(path).iter_batches(columns=["split"], batch_size=1024)
                for batch in batches:
                    if any(value != entry["split"] for value in batch.column("split").to_pylist()):
                        raise ValueError("token descriptor split differs from data")
                    count += batch.num_rows
            else:
                with pa.memory_map(str(path), "r") as stream:
                    for batch in ipc.open_stream(stream):
                        if any(
                            value != entry["split"] for value in batch.column("split").to_pylist()
                        ):
                            raise ValueError("token descriptor split differs from data")
                        count += batch.num_rows
            if count != entry["rows"]:
                raise ValueError("token file row count mismatch")
            total += count
        if total != manifest["summary"]["rows"].get("accepted", 0):
            raise ValueError("token shard accepted count mismatch")
    return manifest


def run_shard(shard):
    state = _STATE
    config = state["config"]
    run_dir = Path(state["run_dir"])
    target = run_dir / "shards" / f"{shard['index']:06d}"
    target.parent.mkdir(exist_ok=True)
    with shard_lock(run_dir, shard["index"]):
        source_indices = {part["source"] for part in shard["slices"]}
        verify_inputs(config, source_indices=source_indices)
        if target.exists():
            return validate_shard(run_dir, shard, config["fingerprint"], config["operation"])
        temporary = Path(tempfile.mkdtemp(prefix=f".{shard['index']:06d}-", dir=target.parent))
        writer = None
        try:
            report = Report()
            if config["operation"] == "tokenize":
                writer = TokenWriter(temporary, config["format"], config["emit_loss_weights"])
            with (temporary / "issues.jsonl").open("w", encoding="utf-8") as sink:
                batch = []
                size = 0
                for row in iter_rows(
                    config["sources"], shard, config["column"], config["context_column"]
                ):
                    row_size = len(
                        row[1].encode("utf-8", errors="surrogatepass")
                        if isinstance(row[1], str)
                        else row[1]
                    )
                    if row[2] is not None:
                        context_text = (
                            row[2]
                            if isinstance(row[2], (str, bytes))
                            else json.dumps(row[2], ensure_ascii=True)
                        )
                        row_size += len(
                            context_text.encode("utf-8", errors="surrogatepass")
                            if isinstance(context_text, str)
                            else context_text
                        )
                    if batch and (
                        len(batch) >= state["batch_rows"] or size + row_size > state["batch_bytes"]
                    ):
                        process_batch(batch, report, sink, writer)
                        batch, size = [], 0
                    batch.append(row)
                    size += row_size
                if batch:
                    process_batch(batch, report, sink, writer)
            token_files = writer.close() if writer is not None else []
            writer = None
            summary = report.as_dict()
            atomic_json(temporary / "summary.json", summary)
            manifest = {
                "version": 1,
                "fingerprint": config["fingerprint"],
                "index": shard["index"],
                "plan": shard,
                "summary": summary,
                "token_files": token_files,
                "files": inventory(temporary),
            }
            verify_inputs(config, source_indices=source_indices)
            atomic_json(temporary / "manifest.json", manifest)
            os.rename(temporary, target)
            return manifest
        finally:
            if writer is not None:
                writer.close()
            if temporary.exists():
                shutil.rmtree(temporary)


def execute(config, run_dir, shards, workers, batch_rows, batch_bytes, tokenizer_threads):
    args = (config, str(run_dir), batch_rows, batch_bytes, tokenizer_threads)
    if workers == 1:
        initialize(*args)
        for shard in shards:
            yield run_shard(shard)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=args,
        ) as pool:
            pending = deque()
            remaining = iter(shards)
            for _ in range(workers):
                if (shard := next(remaining, None)) is not None:
                    pending.append(pool.submit(run_shard, shard))
            while pending:
                yield pending.popleft().result()
                if (shard := next(remaining, None)) is not None:
                    pending.append(pool.submit(run_shard, shard))
