"""apertus-data: prepare, encode and merge distributed check/tokenize runs; inspect tokens.

Exit codes: 0 success, 1 check found failing samples, 2 error.
"""

import argparse
import json
import os
import sys
import traceback

from ._inspect import render
from ._run import encode, merge, prepare


def parser():
    root = argparse.ArgumentParser(prog="apertus-data", description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="command", required=True)
    formatter = argparse.ArgumentDefaultsHelpFormatter
    cmd = commands.add_parser(
        "prepare", help="Plan the shards of a check or tokenize run", formatter_class=formatter
    )
    cmd.add_argument("mode", choices=("check", "tokenize"))
    cmd.add_argument("input", help="Parquet file or directory of them, or a saved HF dataset")
    cmd.add_argument("run", help="New or empty run directory for shards and final outputs")
    cmd.add_argument(
        "--column", default="conversation_json", help="Column with the native JSON strings"
    )
    cmd.add_argument("--split", default="train", help="Split of a saved HF DatasetDict")
    cmd.add_argument("--tokenizer", help="Apertus 2 tokenizer directory (used by tokenize)")
    cmd.add_argument(
        "--loss-weights",
        action="store_true",
        help="check: accept loss_weight annotations; tokenize: also write loss_weights.bin/.idx",
    )
    cmd.add_argument("--shard-rows", type=int, default=10_000, help="Rows per shard")
    cmd = commands.add_parser(
        "encode", help="Check or tokenize unclaimed shards with local worker processes"
    )
    cmd.add_argument("run", help="Prepared run directory")
    cmd.add_argument("--workers", type=int, help="Worker processes (default: CPUs / threads)")
    cmd.add_argument("--threads", type=int, default=1, help="Tokenizer threads per worker")
    cmd = commands.add_parser("merge", help="Verify all shards and write the final outputs")
    cmd.add_argument("run", help="Prepared run directory")
    cmd = commands.add_parser(
        "inspect", help="Print documents of a merged tokenize run", formatter_class=formatter
    )
    cmd.add_argument("run", help="Merged tokenize run directory")
    cmd.add_argument(
        "--index", type=int, action="append", help="Zero-based document to show (repeatable)"
    )
    cmd.add_argument("--sample", type=int, default=3, help="Random documents when no --index")
    cmd.add_argument("--seed", type=int, help="Seed for --sample")
    cmd.add_argument("--tokenizer", help="Override the run's tokenizer directory")
    cmd.add_argument(
        "--source", action="store_true", help="Also print the source row's native JSON"
    )
    cmd.add_argument(
        "--tokens",
        action="store_true",
        help="One line per token: position, ID, [loss weight,] text (partial for multi-byte)",
    )
    cmd.add_argument(
        "--weights", action="store_true", help="Also show loss weights (run needs --loss-weights)"
    )
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            config = prepare(
                args.mode,
                args.input,
                args.run,
                column=args.column,
                split=args.split,
                shard_rows=args.shard_rows,
                loss_weights=args.loss_weights,
                tokenizer=args.tokenizer,
            )
            print(f"prepared {len(config['shards'])} shards in {args.run}", file=sys.stderr)
        elif args.command == "encode":
            if args.threads < 1 or (args.workers is not None and args.workers < 1):
                raise ValueError("--workers and --threads must be positive")
            workers = args.workers or max(1, (os.process_cpu_count() or 1) // args.threads)
            encode(args.run, workers=workers, threads=args.threads)
        elif args.command == "merge":
            summary = merge(args.run)
            print(json.dumps(summary, indent=1))
            return 1 if summary.get("failed_rows") else 0
        else:
            blocks = render(
                args.run,
                indices=args.index or (),
                sample=args.sample,
                seed=args.seed,
                tokenizer=args.tokenizer,
                source=args.source,
                tokens=args.tokens,
                weights=args.weights,
            )
            try:
                for block in blocks:
                    print(block, flush=True)
            except BrokenPipeError:  # e.g. piped into head; silence the flush at exit too
                os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except ValueError as error:
        cause = f" ({error.__cause__})" if error.__cause__ else ""
        print(f"error: {error}{cause}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - every failure maps to exit code 2
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
