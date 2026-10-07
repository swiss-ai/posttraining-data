"""Measure incremental Python allocations while processing a generated JSONL corpus.

The corpus is generated on disk before measurement. Peaks exclude earlier
imports, but include run initialization and lazy imports. Native allocations
are excluded; use system RSS monitoring for total process RAM.
"""

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from tempfile import TemporaryDirectory

from apertus_common import Conversation, SystemPrompt, User

from apertus2_processing.cli import main as process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=5000)
    parser.add_argument("--payload-bytes", type=int, default=8192)
    parser.add_argument("--tokenizer", help="Also tokenize to Parquet with aligned weights")
    options = parser.parse_args()
    if options.records < 1 or options.payload_bytes < 1:
        parser.error("record and payload sizes must be positive")
    record = (
        Conversation(
            system=SystemPrompt.build(), items=[User(payload="x" * options.payload_bytes)]
        ).to_json()
        + "\n"
    )
    with TemporaryDirectory(prefix="apertus-memory-") as temporary:
        root = Path(temporary)
        source = root / "source.jsonl"
        with source.open("w", encoding="utf-8") as stream:
            for _ in range(options.records):
                stream.write(record)
        tracemalloc.start()
        started = time.perf_counter()
        command = [
            "tokenize" if options.tokenizer else "check",
            str(source),
            "--output",
            str(root / "report"),
            "--batch-rows",
            "32",
            "--batch-bytes",
            "262144",
            "--num-shards",
            "4",
        ]
        if options.tokenizer:
            command.extend(
                [
                    "--tokenizer",
                    options.tokenizer,
                    "--format",
                    "parquet",
                    "--check",
                    "full",
                    "--emit-loss-weights",
                ]
            )
        result = process(command)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if result:
            raise RuntimeError(f"processing failed with exit status {result}")
        print(
            json.dumps(
                {
                    "records": options.records,
                    "input_bytes": source.stat().st_size,
                    "python_peak_bytes": peak,
                    "elapsed_seconds": elapsed,
                    "batch_rows": 32,
                    "batch_bytes": 262144,
                }
            )
        )


if __name__ == "__main__":
    main()
