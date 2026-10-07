"""Plan identity, policy completion and verified output merging."""

import os
import shutil
import tempfile
from pathlib import Path

from ._guards import snapshot, verify_inputs
from ._input import discover, make_plan
from ._output import token_schema
from ._report import Report
from ._runner import validate_shard
from ._util import (
    atomic_json,
    checksum,
    digest,
    inventory,
    library_identity,
    read_json,
    runtime_identity,
    verify_files,
)


def prepare(args):
    root = Path(args.output).resolve()
    source_path = Path(args.input).resolve()
    if root == source_path or (source_path.is_dir() and root.is_relative_to(source_path)):
        raise ValueError("output must be outside the input dataset")
    sources = discover(source_path, args.input_format, args.split)
    if (
        args.context_column
        and not args.conversation_column
        and any(source["kind"] == "jsonl" for source in sources)
    ):
        raise ValueError("--context-column requires --conversation-column for JSONL")
    tokenizer = str(Path(args.tokenizer).resolve()) if args.tokenizer else None
    if tokenizer is not None and not Path(tokenizer).is_dir():
        raise ValueError("--tokenizer must name a local artifact directory")
    context = read_json(args.context_file) if args.context_file else None
    config = {
        "version": 1,
        "operation": args.command,
        "sources": sources,
        "plan": make_plan(sources, args.num_shards),
        "column": args.conversation_column,
        "context_column": args.context_column,
        "context": context,
        "check": "full" if args.command == "check" else args.check,
        "allow_loss_weights": args.allow_loss_weights or args.emit_loss_weights,
        "emit_loss_weights": args.emit_loss_weights,
        "on_error": args.on_error,
        "format": args.format,
        "tokenizer": tokenizer,
        "artifact": snapshot(tokenizer) if tokenizer else None,
        "library": library_identity(),
        "runtime": runtime_identity(),
        "pipeline": [
            {"path": p.name, "sha256": checksum(p)}
            for p in sorted(Path(__file__).parent.glob("*.py"))
        ],
    }
    verify_inputs(config)
    config["fingerprint"] = digest(config)
    root.mkdir(parents=True, exist_ok=True)
    try:
        atomic_json(root / "run.json", config, exclusive=True)
    except FileExistsError:
        existing = read_json(root / "run.json")
        if existing != config:
            raise ValueError(
                "run fingerprint mismatch: inputs, artifact, code or semantic options changed"
            ) from None
    selected = config["plan"] if args.shard_index is None else [config["plan"][args.shard_index]]
    if not args.resume and any((root / "shards" / f"{s['index']:06d}").exists() for s in selected):
        raise ValueError("completed shard exists; use --resume to validate and reuse it")
    return root, config, selected


def policy_pass(config, report):
    return not report["rows"].get("rejected", 0) or (
        config["operation"] == "tokenize" and config["on_error"] == "skip"
    )


def load_run(root):
    config = read_json(Path(root) / "run.json")
    fingerprint = config.pop("fingerprint")
    if digest(config) != fingerprint:
        raise ValueError("run manifest fingerprint mismatch")
    config["fingerprint"] = fingerprint
    return config


def verified_manifests(root, config):
    planned = {f"{p['index']:06d}" for p in config["plan"]}
    directory = Path(root) / "shards"
    actual = (
        {p.name for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")}
        if directory.exists()
        else set()
    )
    if actual != planned:
        raise ValueError(
            f"incomplete or unexpected shards: missing={sorted(planned - actual)}, extra={sorted(actual - planned)}"
        )
    return [
        validate_shard(root, shard, config["fingerprint"], config["operation"])
        for shard in config["plan"]
    ]


def merge(root, *, verify_sources=True):
    root = Path(root).resolve()
    config = load_run(root)
    if verify_sources:
        verify_inputs(config, full=True)
    manifests = verified_manifests(root, config)
    report = Report()
    for manifest in manifests:
        report.merge(manifest["summary"])
    summary = report.as_dict()
    summary.update(
        fingerprint=config["fingerprint"],
        complete=True,
        expected_shards=len(config["plan"]),
        completed_shards=len(manifests),
        policy_pass=policy_pass(config, report.as_dict()),
    )
    atomic_json(root / "summary.json", summary)
    if config["operation"] == "tokenize" and summary["policy_pass"]:
        materialize(root, config, manifests)
    return summary


def materialize(root, config, manifests):
    target = root / "dataset"
    if target.exists():
        existing = read_json(target / "manifest.json")
        if existing["fingerprint"] != config["fingerprint"]:
            raise ValueError("merged dataset fingerprint mismatch")
        verify_files(target, existing["files"])
        return
    temporary = Path(tempfile.mkdtemp(prefix=".dataset-", dir=root))
    try:
        by_split = {source["split"]: [] for source in config["sources"]}
        for manifest in manifests:
            for entry in manifest["token_files"]:
                by_split[entry["split"]].append(
                    root / "shards" / f"{manifest['index']:06d}" / entry["path"]
                )
        if config["format"] == "hf":
            from datasets import Dataset, DatasetDict, Features, concatenate_datasets

            splits = {}
            features = Features.from_arrow_schema(token_schema(config["emit_loss_weights"]))
            for split, files in by_split.items():
                splits[split] = (
                    concatenate_datasets([Dataset.from_file(str(path)) for path in files])
                    if files
                    else Dataset.from_dict({name: [] for name in features}, features=features)
                )
            DatasetDict(splits).save_to_disk(
                str(temporary), num_shards={name: max(1, len(by_split[name])) for name in splits}
            )
        else:
            split_files = {}
            for split, files in by_split.items():
                name = digest(split)[:16]
                (temporary / name).mkdir()
                paths = []
                for i, path in enumerate(files):
                    relative = f"{name}/{i:06d}.parquet"
                    os.link(path, temporary / relative)
                    paths.append(relative)
                if not paths:
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    relative = f"{name}/000000.parquet"
                    pq.write_table(
                        pa.Table.from_pylist([], schema=token_schema(config["emit_loss_weights"])),
                        temporary / relative,
                    )
                    paths.append(relative)
                split_files[split] = paths
            atomic_json(temporary / "splits.json", split_files)
        atomic_json(
            temporary / "manifest.json",
            {
                "fingerprint": config["fingerprint"],
                "format": config["format"],
                "files": inventory(temporary),
            },
        )
        try:
            os.rename(temporary, target)
        except OSError:
            if not target.exists():
                raise
            existing = read_json(target / "manifest.json")
            if existing["fingerprint"] != config["fingerprint"]:
                raise ValueError("concurrent merge fingerprint mismatch") from None
            verify_files(target, existing["files"])
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
