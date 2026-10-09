"""Prepared runs: workers on any node claim shards; merge combines the results."""

import importlib.metadata
import json
import multiprocessing
import os
import shutil
import sys
import uuid
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import _indexed, _inputs

CONFIG = "run.json"
BATCH = 64
DTYPES = {"tokens": np.int32, "loss_weights": np.float32}


def write_json(path, value):
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def library():
    distribution = importlib.metadata.distribution("apertus-common")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    return {"version": distribution.version, "commit": origin.get("vcs_info", {}).get("commit_id")}


def load(run):
    run = Path(run).resolve()
    if not (run / CONFIG).exists():
        raise ValueError(f"{run} is not prepared; run 'apertus-data prepare' first")
    config = read_json(run / CONFIG)
    if config["library"] != library():
        raise ValueError(
            "apertus-common version or revision changed since prepare; use a new run directory"
        )
    return run, config


def verify_inputs(config):
    roots = [config["input"], *filter(None, [config["tokenizer"]])]
    if _inputs.snapshot(*roots) != config["files"]:
        raise ValueError("input or tokenizer files changed since prepare; use a new run directory")


def prepare(mode, source, run, *, column, split, shard_rows, loss_weights, tokenizer):
    """Plan a run once; later calls only verify that options and inputs are unchanged."""
    source, run = Path(source).resolve(), Path(run).resolve()
    if run == source or run.is_relative_to(source):
        raise ValueError("the run directory must be outside the input")
    if shard_rows < 1:
        raise ValueError("--shard-rows must be positive")
    if mode == "tokenize" and not (tokenizer and Path(tokenizer).is_dir()):
        raise ValueError("tokenize needs --tokenizer pointing to a tokenizer directory")
    options = {
        "mode": mode,
        "input": str(source),
        "column": column,
        "split": split,
        "shard_rows": shard_rows,
        "loss_weights": loss_weights,
        "tokenizer": str(Path(tokenizer).resolve()) if mode == "tokenize" else None,
        "library": library(),
    }
    if (run / CONFIG).exists():
        config = read_json(run / CONFIG)
        if {key: config[key] for key in options} != options:
            raise ValueError(f"{run} was prepared with other options; use a new run directory")
        verify_inputs(config)
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError(f"{run} is not empty; use a new run directory")
        if mode == "tokenize":
            from apertus_common import load_encoding

            load_encoding(options["tokenizer"])  # fail here rather than on every worker
        files = _inputs.snapshot(source, *filter(None, [options["tokenizer"]]))
        inputs, shards = _inputs.plan(source, column, split, shard_rows)
        config = options | {"files": files, "sources": inputs, "shards": shards}
        verify_inputs(config)  # the input did not change while it was planned
        run.mkdir(parents=True, exist_ok=True)
        write_json(run / CONFIG, config)
    # Claims and partial shards belong to earlier jobs, which have ended.
    shutil.rmtree(run / "claims", ignore_errors=True)
    for partial in (run / "shards").glob(".tmp-*"):
        shutil.rmtree(partial)
    return config


def _issue(rule, severity, location, message):
    return {"rule": rule, "severity": severity, "location": location, "message": message}


def _issue_counts(counts):
    return [{"rule": r, "severity": s, "count": n} for (r, s), n in sorted(counts.items())]


class Check:
    def __init__(self, config):
        from apertus_common import checkers_for

        self.checkers = checkers_for(
            "full", allow_loss_weights=config["loss_weights"], training=True
        )
        self.column = config["column"]

    def inspect(self, raw):
        from apertus_common import Conversation

        try:
            conversation = Conversation.from_json(raw)
        except Exception as error:  # noqa: BLE001 - an unparseable sample is reported, not fatal
            return False, [
                _issue("structure/invalid", "error", None, f"{type(error).__name__}: {error}")
            ]
        try:
            report = conversation.check(checkers=self.checkers)
        except Exception as error:  # noqa: BLE001 - checker failures fail the sample, not the job
            return False, [
                _issue("check/exception", "unevaluated", None, f"{type(error).__name__}: {error}")
            ]
        found = [
            _issue(v.rule, str(v.severity), asdict(v.location), v.message)
            for v in report.violations
        ] + [
            _issue(v.rule, "unevaluated", asdict(v.location), v.message) for v in report.unevaluated
        ]
        return report.ok and report.complete, found

    def __call__(self, source, shard, directory):
        rows = failed = with_issues = 0
        counts = Counter()
        with open(directory / "issues.jsonl", "w", encoding="utf-8") as sink:
            for row, raw in enumerate(_inputs.rows(source, shard, self.column), shard["start"]):
                ok, found = self.inspect(raw)
                rows, failed, with_issues = rows + 1, failed + (not ok), with_issues + bool(found)
                for issue in found:
                    line = {"source": source["path"], "row": row} | issue
                    sink.write(json.dumps(line, ensure_ascii=False) + "\n")
                    counts[issue["rule"], issue["severity"]] += 1
        return {
            "rows": rows,
            "failed_rows": failed,
            "rows_with_issues": with_issues,
            "issues": _issue_counts(counts),
        }


class Tokenize:
    def __init__(self, config):
        from apertus_common import load_encoding

        self.encoding = load_encoding(config["tokenizer"])
        self.column = config["column"]
        self.weights = config["loss_weights"]

    def encode(self, batch, source, tokens, weights):
        try:
            results = self.encoding.encode_conversations(
                [conversation for _, conversation in batch], return_loss_weights=self.weights
            )
        except Exception as error:
            rows = f"rows {batch[0][0]}-{batch[-1][0]}"
            raise ValueError(f"{source['path']} {rows} could not be encoded") from error
        for result in results:
            tokens.add(result.token_ids if self.weights else result)
            if weights is not None:
                weights.add(result.loss_weights)

    def __call__(self, source, shard, directory):
        from apertus_common import Conversation

        weights_writer = (
            _indexed.Writer(directory / "loss_weights", DTYPES["loss_weights"])
            if self.weights
            else nullcontext()
        )
        batch, rows = [], 0
        tokens_writer = _indexed.Writer(directory / "tokens", DTYPES["tokens"])
        with tokens_writer as tokens, weights_writer as weights:
            for row, raw in enumerate(_inputs.rows(source, shard, self.column), shard["start"]):
                try:
                    batch.append((row, Conversation.from_json(raw)))
                except Exception as error:
                    raise ValueError(
                        f"{source['path']} row {row} is not a native conversation; run check first"
                    ) from error
                rows += 1
                if len(batch) == BATCH:
                    self.encode(batch, source, tokens, weights)
                    batch = []
            if batch:
                self.encode(batch, source, tokens, weights)
        return {"rows": rows, "tokens": int(sum(tokens.lengths))}


def _run_shard(run, config, index, process):
    """Process one shard in a temporary directory and publish it with one rename."""
    shard = config["shards"][index]
    partial = run / "shards" / f".tmp-{index:06d}-{uuid.uuid4().hex[:8]}"
    partial.mkdir()
    try:
        summary = process(config["sources"][shard["source"]], shard, partial)
        write_json(partial / "summary.json", summary)
        for path in partial.iterdir():  # on disk before the shard counts as complete
            with open(path, "rb") as stream:
                os.fsync(stream.fileno())
        partial.rename(run / "shards" / f"{index:06d}")
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    return summary


def _worker(run, config, job, threads, rank, world):
    """Claim and process free shards until none are left."""
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if threads > 1 else "false"
    process = Check(config) if config["mode"] == "check" else Tokenize(config)
    claims = run / "claims" / job
    (run / "shards").mkdir(exist_ok=True)
    claims.mkdir(parents=True, exist_ok=True)
    taken = set(os.listdir(run / "shards"))  # completed shards, then also claimed ones
    count = len(config["shards"])
    first = rank * count // world  # start apart from the other workers
    for index in ((first + step) % count for step in range(count)):
        name = f"{index:06d}"
        if name in taken:
            continue
        try:
            (claims / name).mkdir()  # atomic on shared filesystems: exactly one worker wins
        except FileExistsError:
            taken.update(os.listdir(claims))  # skip whatever was claimed meanwhile
            continue
        summary = _run_shard(run, config, index, process)
        print(
            f"shard {index + 1}/{count} done: {summary['rows']} rows", file=sys.stderr, flush=True
        )


def encode(run, *, workers, threads):
    """Run `workers` processes; the tasks of one srun step share the shard pool."""
    run, config = load(run)
    verify_inputs(config)  # once per node, not per worker
    step = [os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")]
    job = ".".join(filter(None, step)) or f"local-{uuid.uuid4().hex[:8]}"
    task, tasks = int(os.environ.get("SLURM_PROCID", "0")), int(os.environ.get("SLURM_NTASKS", "1"))
    ranks = [(task * workers + local, tasks * workers) for local in range(workers)]
    if workers == 1:
        _worker(run, config, job, threads, *ranks[0])
        return
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(run, config, job, threads, rank, world))
        for rank, world in ranks
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = sum(process.exitcode != 0 for process in processes)
    if failed:
        raise RuntimeError(f"{failed} of {workers} workers failed; see their errors above")


def merge(run):
    """Verify that every shard is complete, then write the final report or dataset."""
    run, config = load(run)
    verify_inputs(config)
    names = [f"{index:06d}" for index in range(len(config["shards"]))]
    missing = [name for name in names if not (run / "shards" / name).is_dir()]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(names)} shards are incomplete (first: {missing[0]}); "
            "resubmit the same job to resume"
        )
    summaries = [read_json(run / "shards" / name / "summary.json") for name in names]
    damaged = "shards/{} is damaged; delete it and resubmit the same job"
    for name, shard, summary in zip(names, config["shards"], summaries, strict=True):
        if summary["rows"] != shard["end"] - shard["start"]:
            raise ValueError(damaged.format(name))
    rows = sum(summary["rows"] for summary in summaries)
    if config["mode"] == "check":
        counts = Counter()
        with open(run / "issues.jsonl.tmp", "wb") as target:
            for name, summary in zip(names, summaries, strict=True):
                lines = 0
                with open(run / "shards" / name / "issues.jsonl", "rb") as source:
                    while chunk := source.read(1 << 24):
                        target.write(chunk)
                        lines += chunk.count(b"\n")
                if lines != sum(issue["count"] for issue in summary["issues"]):
                    raise ValueError(damaged.format(name))
                counts.update({(i["rule"], i["severity"]): i["count"] for i in summary["issues"]})
        os.replace(run / "issues.jsonl.tmp", run / "issues.jsonl")
        result = {
            "mode": "check",
            "rows": rows,
            "failed_rows": sum(summary["failed_rows"] for summary in summaries),
            "rows_with_issues": sum(summary["rows_with_issues"] for summary in summaries),
            "issues": _issue_counts(counts),
        }
    else:
        for output in ["tokens"] + ["loss_weights"] * config["loss_weights"]:
            partial = run / f".{output}.tmp"
            prefixes = [run / "shards" / name / output for name in names]
            if _indexed.merge(prefixes, partial, DTYPES[output]) != rows:
                raise ValueError(f"{output} shards do not hold one document per input row")
            os.replace(f"{partial}.bin", run / f"{output}.bin")
            os.replace(f"{partial}.idx", run / f"{output}.idx")
        result = {
            "mode": "tokenize",
            "documents": rows,
            "tokens": sum(summary["tokens"] for summary in summaries),
            "loss_weights": config["loss_weights"],
        }
    write_json(run / "summary.json", result)
    return result
