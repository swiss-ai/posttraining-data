"""Command line interface; data failures and operational failures are distinct."""

import argparse
import sys
from pathlib import Path

from ._report import Report
from ._run import load_run, merge, policy_pass, prepare
from ._runner import execute
from ._util import canonical


def parser():
    root = argparse.ArgumentParser(description="Check and tokenize native Apertus corpora")
    commands = root.add_subparsers(dest="command", required=True)
    for operation in ("check", "tokenize"):
        cmd = commands.add_parser(operation)
        cmd.add_argument("input")
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--input-format", choices=("auto", "hf", "jsonl"), default="auto")
        cmd.add_argument(
            "--conversation-column",
            help="HF defaults to conversation_json; JSONL defaults to bare records",
        )
        cmd.add_argument(
            "--context-column", help="Optional per-record serialized CheckContext in wrapped input"
        )
        cmd.add_argument("--context-file", help="Default CheckContext JSON for each record")
        cmd.add_argument("--split", help="Select an HF split, or label JSONL (default train)")
        cmd.add_argument("--workers", type=int, default=1)
        cmd.add_argument("--tokenizer-threads", type=int, default=1)
        cmd.add_argument("--num-shards", type=int, default=1)
        cmd.add_argument(
            "--shard-index",
            type=int,
            help="Process only this zero-based shard for external schedulers",
        )
        cmd.add_argument("--batch-rows", type=int, default=64)
        cmd.add_argument("--batch-bytes", type=int, default=4 * 1024 * 1024)
        cmd.add_argument("--resume", action="store_true")
        cmd.add_argument(
            "--prepare-only",
            action="store_true",
            help="Fingerprint inputs and plan shards without processing",
        )
        if operation == "check":
            cmd.add_argument("--allow-loss-weights", action="store_true")
            cmd.set_defaults(
                tokenizer=None, check="full", on_error="fail", format=None, emit_loss_weights=False
            )
        else:
            cmd.add_argument("--tokenizer", required=True)
            cmd.add_argument("--format", choices=("hf", "parquet"), default="hf")
            cmd.add_argument("--check", choices=("none", "full"), default="none")
            cmd.add_argument("--on-error", choices=("fail", "skip"), default="fail")
            cmd.add_argument("--emit-loss-weights", action="store_true")
            cmd.set_defaults(allow_loss_weights=False)
    cmd = commands.add_parser("run-shard", help="Execute one shard from a prepared immutable run")
    cmd.add_argument("run_dir")
    cmd.add_argument("--shard-index", type=int, required=True)
    cmd.add_argument("--workers", type=int, default=1)
    cmd.add_argument("--batch-rows", type=int, default=64)
    cmd.add_argument("--batch-bytes", type=int, default=4 * 1024 * 1024)
    cmd.add_argument("--tokenizer-threads", type=int, default=1)
    cmd.add_argument("--resume", action="store_true")
    cmd = commands.add_parser("merge", help="Verify complete shard coverage and merge a run")
    cmd.add_argument("run_dir")
    cmd = commands.add_parser(
        "export-megatron", help="Export validated token shards through Megatron's indexed writer"
    )
    cmd.add_argument("token_output")
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--megatron-path", help="Optional Megatron checkout to import")
    cmd.add_argument("--documents-per-shard", type=int, default=10000)
    cmd.add_argument("--tokens-per-shard", type=int, default=10_000_000)
    cmd.add_argument("--resume", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "merge":
            summary = merge(args.run_dir)
            print(canonical(summary))
            return 0 if summary["policy_pass"] else 1
        if args.command == "export-megatron":
            from ._megatron import export

            export(args)
            return 0
        for name in ("workers", "batch_rows", "batch_bytes", "tokenizer_threads"):
            if getattr(args, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.command == "run-shard":
            root = Path(args.run_dir).resolve()
            config = load_run(root)
            if not 0 <= args.shard_index < len(config["plan"]):
                raise ValueError("--shard-index is outside the prepared plan")
            shards = [config["plan"][args.shard_index]]
            if not args.resume and (root / "shards" / f"{args.shard_index:06d}").exists():
                raise ValueError("completed shard exists; use --resume")
        else:
            if args.num_shards < 1:
                raise ValueError("--num-shards must be positive")
            if args.shard_index is not None and not 0 <= args.shard_index < args.num_shards:
                raise ValueError("--shard-index must be in [0, --num-shards)")
            root, config, shards = prepare(args)
            if args.prepare_only:
                print(
                    canonical(
                        {
                            "prepared": True,
                            "run": str(root),
                            "fingerprint": config["fingerprint"],
                            "shards": len(shards),
                        }
                    )
                )
                return 0
        report = Report()
        for manifest in execute(
            config,
            root,
            shards,
            args.workers,
            args.batch_rows,
            args.batch_bytes,
            args.tokenizer_threads,
        ):
            report.merge(manifest["summary"])
            print(
                f"Shard {manifest['index'] + 1}/{len(config['plan'])}: {manifest['summary']['rows']}",
                file=sys.stderr,
            )
        if args.shard_index is None:
            summary = merge(root)
        else:
            summary = report.as_dict()
            summary.update(
                complete=False, policy_pass=policy_pass(config, summary), run=str(Path(root))
            )
        print(canonical(summary))
        return 0 if summary["policy_pass"] else 1
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - CLI exit contract
        print(f"Operational failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
