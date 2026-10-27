"""apertus-data: check or tokenize native data in shards; inspect, cap, bucket or split tokens.

Exit codes: 0 success, 1 check found failing samples, 2 error.
"""

import argparse
import json
import os
import sys
import traceback

from ._inspect import documents, overview, overview_text, render, write_report
from ._run import encode, merge, prepare
from ._selection import bucket, cap, split


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
        "inspect",
        help="Print or report documents of a merged tokenize run",
        formatter_class=formatter,
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
        help="Text output: one line per token with position, ID, loss weight, and text",
    )
    cmd.add_argument(
        "--weights",
        action="store_true",
        help="Show the written loss_weights instead of the trainer defaults (needs --loss-weights)",
    )
    cmd.add_argument(
        "--output",
        metavar="DIR",
        help="Write document-NNNNNN.txt files and a standalone report.html to DIR, not stdout",
    )
    cmd = commands.add_parser(
        "cap", help="Keep the documents of a merged tokenize run up to a token length"
    )
    cmd.add_argument("run", help="Merged tokenize run directory")
    cmd.add_argument("output", help="New or empty directory for the capped run")
    cmd.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        metavar="TOKENS",
        help="Longest document to keep, in tokens",
    )
    cmd = commands.add_parser("bucket", help="Split a merged tokenize run into length buckets")
    cmd.add_argument("run", help="Merged tokenize run directory")
    cmd.add_argument("output", help="New or empty directory for the buckets")
    cmd.add_argument(
        "--bounds",
        type=int,
        nargs="+",
        required=True,
        metavar="TOKENS",
        help="Increasing upper bounds in tokens; a final bucket holds documents above the last "
        "(left out when the run is capped at or below it)",
    )
    cmd = commands.add_parser("split", help="Split a merged tokenize run into train and validation")
    cmd.add_argument("run", help="Merged tokenize run directory")
    cmd.add_argument("output", help="New or empty directory for train/ and validation/")
    cmd.add_argument(
        "--validation",
        type=float,
        required=True,
        metavar="FRACTION",
        help="Approximate share of documents (or groups) for validation, 0 < FRACTION < 1",
    )
    cmd.add_argument(
        "--seed", type=int, default=0, help="Seed of the side-picking hash (default: %(default)s)"
    )
    cmd.add_argument(
        "--group-by",
        metavar="COLUMN",
        help="Column of the tokenized input; documents sharing its value stay on one side",
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
        elif args.command in ("cap", "bucket", "split"):
            if args.command == "cap":
                result = cap(args.run, args.output, args.max_tokens)
            elif args.command == "bucket":
                result = bucket(args.run, args.output, args.bounds)
            else:
                result = split(
                    args.run, args.output, args.validation, seed=args.seed, group_by=args.group_by
                )
            print(json.dumps(result, indent=1))
        else:
            records = documents(
                args.run,
                indices=args.index or (),
                sample=args.sample,
                seed=args.seed,
                tokenizer=args.tokenizer,
                source=args.source,
                weights=args.weights,
            )
            if args.output:
                paths, report = write_report(records, args.output, run=args.run, tokens=args.tokens)
                print(
                    f"wrote {len(paths)} documents and {report.name} to {report.parent}",
                    file=sys.stderr,
                )
            else:
                try:
                    stats = overview(args.run)
                    print(f"=== run {stats['run']}: {overview_text(stats)}", flush=True)
                    for record in records:
                        print(render(record, tokens=args.tokens), flush=True)
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
