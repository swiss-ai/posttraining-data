"""Optional export through the consumer's own indexed dataset implementation."""

import importlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._output import iter_token_rows
from ._run import load_run, merge, verified_manifests
from ._util import atomic_json, canonical, checksum, digest, inventory, read_json, verify_files

TESTED_MEGATRON_REVISION = "04edde000b189470836d6fe85cda9a9f3c0b3161"


def load_writer(path):
    revision = None
    if path:
        path = Path(path).resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != TESTED_MEGATRON_REVISION:
            raise ValueError(
                f"Megatron revision {revision} differs from supported {TESTED_MEGATRON_REVISION}"
            )
        sys.path.insert(0, str(path))
    try:
        module = importlib.import_module("megatron.core.datasets.indexed_dataset")
    except ImportError as error:
        raise ValueError(
            "Megatron export requires its optional runtime and IndexedDatasetBuilder; install the consumer's dependencies in a separate environment"
        ) from error
    return module.IndexedDatasetBuilder, {
        "revision": revision,
        "indexed_dataset_sha256": checksum(module.__file__),
    }


def export(args):
    if args.documents_per_shard < 1 or args.tokens_per_shard < 1:
        raise ValueError("Megatron shard bounds must be positive")
    source = Path(args.token_output).resolve()
    output = Path(args.output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Megatron output must be outside the token run")
    config = load_run(source)
    if config["operation"] != "tokenize":
        raise ValueError("Megatron input must be a tokenization run")
    summary = merge(source, verify_sources=False)
    if not summary["policy_pass"]:
        raise ValueError("cannot export a token run that failed its acceptance policy")
    manifests = verified_manifests(source, config)
    builder, writer_identity = load_writer(args.megatron_path)
    identity = {
        "version": 1,
        "input_fingerprint": config["fingerprint"],
        "documents_per_shard": args.documents_per_shard,
        "tokens_per_shard": args.tokens_per_shard,
        "writer": writer_identity,
        "emit_loss_weights": config["emit_loss_weights"],
        "token_dtype": "int32",
        "loss_weight_dtype": "float64" if config["emit_loss_weights"] else None,
    }
    fingerprint = digest(identity)
    output.mkdir(parents=True, exist_ok=True)
    try:
        atomic_json(
            output / "export.json", {**identity, "fingerprint": fingerprint}, exclusive=True
        )
    except FileExistsError:
        if not args.resume:
            raise ValueError("Megatron export exists; use --resume") from None
        if read_json(output / "export.json") != {**identity, "fingerprint": fingerprint}:
            raise ValueError("Megatron resume fingerprint mismatch") from None
    outputs = []
    rows = []
    token_count = 0
    for row in iter_token_rows(source, manifests):
        if rows and (
            len(rows) >= args.documents_per_shard
            or token_count + row["token_count"] > args.tokens_per_shard
            or rows[0]["split"] != row["split"]
        ):
            outputs.append(
                export_shard(
                    output, len(outputs), rows, builder, fingerprint, config["emit_loss_weights"]
                )
            )
            rows, token_count = [], 0
        validate_row(row, config["emit_loss_weights"])
        rows.append(row)
        token_count += row["token_count"]
    if rows:
        outputs.append(
            export_shard(
                output, len(outputs), rows, builder, fingerprint, config["emit_loss_weights"]
            )
        )
    actual = {p.name for p in output.glob("shard-*") if p.is_dir()}
    expected = {f"shard-{i:06d}" for i in range(len(outputs))}
    if actual != expected:
        raise ValueError("unexpected Megatron shards")
    atomic_json(
        output / "manifest.json",
        {**identity, "fingerprint": fingerprint, "complete": True, "shards": outputs},
    )
    print(
        canonical(
            {
                "complete": True,
                "documents": sum(v["documents"] for v in outputs),
                "shards": len(outputs),
                "output": str(output),
            }
        )
    )


def validate_row(row, weights):
    ids = row["token_ids"]
    if (
        row["token_count"] != len(ids)
        or len(ids) > 2**31 - 1
        or any(type(i) is not int or not 0 <= i <= 2**31 - 1 for i in ids)
    ):
        raise ValueError("token IDs or sequence length cannot be represented as int32")
    if weights:
        values = row["loss_weights"]
        if len(values) != len(ids) or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("invalid or misaligned loss weights")


def export_shard(output, index, rows, builder, fingerprint, weights):
    name = f"shard-{index:06d}"
    target = output / name
    batch_digest = digest(rows)
    if target.exists():
        existing = read_json(target / "manifest.json")
        if existing["fingerprint"] != fingerprint or existing["records_sha256"] != batch_digest:
            raise ValueError("Megatron shard resume mismatch")
        verify_files(target, existing["files"])
        return existing
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=output))
    token_builder = weight_builder = None
    try:
        token_builder = builder(str(temporary / "tokens.bin"), dtype=np.int32)
        if weights:
            weight_builder = builder(str(temporary / "loss_weights.bin"), dtype=np.float64)
        with (temporary / "records.jsonl").open("w", encoding="utf-8") as mapping:
            for document, row in enumerate(rows):
                lengths = [len(row["token_ids"])]
                token_builder.add_document(np.asarray(row["token_ids"], dtype=np.int32), lengths)
                if weight_builder is not None:
                    weight_builder.add_document(
                        np.asarray(row["loss_weights"], dtype=np.float64), lengths
                    )
                mapping.write(
                    canonical(
                        {k: row[k] for k in ("source", "split", "row", "record_id", "token_count")}
                        | {"document": document}
                    )
                    + "\n"
                )
        token_builder.finalize(str(temporary / "tokens.idx"))
        if weight_builder is not None:
            weight_builder.finalize(str(temporary / "loss_weights.idx"))
        manifest = {
            "fingerprint": fingerprint,
            "records_sha256": batch_digest,
            "directory": name,
            "documents": len(rows),
            "sequences": len(rows),
            "split": rows[0]["split"],
            "token_prefix": f"{name}/tokens",
            "loss_weight_prefix": f"{name}/loss_weights" if weights else None,
            "files": inventory(temporary),
        }
        atomic_json(temporary / "manifest.json", manifest)
        os.rename(temporary, target)
        return manifest
    finally:
        for obj in (token_builder, weight_builder):
            if obj is not None and not obj.data_file.closed:
                obj.data_file.close()
        if temporary.exists():
            shutil.rmtree(temporary)
