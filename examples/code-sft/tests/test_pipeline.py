"""Exercise stage boundaries using local fixtures; no network or production data."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from apertus_common import Conversation, checkers_for
from standardize import SCHEMA, normalize

ROOT = Path(__file__).resolve().parents[3]
DISPATCHER = ROOT / "examples/code-sft/pipeline.py"


class PipelineTests(unittest.TestCase):
    def test_local_unification_and_dedup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def run(stage, *args):
                result = subprocess.run(
                    [sys.executable, str(DISPATCHER), stage, *map(str, args)],
                    capture_output=True,
                    text=True,
                    env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            source = root / "source.jsonl"
            first = {
                "question": "Write a unique greeting function.",
                "answer": 'def greet(): return "hi"',
            }
            source.write_text(
                "".join(
                    json.dumps(r) + "\n"
                    for r in [
                        first,
                        first,
                        {
                            "question": "Reference evaluation problem",
                            "answer": "evaluation solution",
                        },
                    ]
                )
            )
            run(
                "unify",
                "--input",
                source,
                "--output",
                root / "unified",
                "--repo",
                "example/source",
                "--revision",
                "abc",
            )
            for path in (root / "unified").rglob("part-*.parquet"):
                for row in pq.ParquetFile(path).read().to_pylist():
                    conversation = Conversation.from_json(row["conversation_json"])
                    self.assertTrue(
                        conversation.check(
                            checkers=checkers_for("full", training=True)
                        ).ok
                    )
                    self.assertIn("<|out|>reply<|hdr|>", row["native_text"])
            entries = json.loads(
                (ROOT / "04-decontamination/code-sft/benchmarks.lock.json").read_text()
            )["datasets"]
            for entry in entries:
                directory = root / "evaluation" / entry["repo"]
                directory.mkdir(parents=True)
                (directory / "source.json").write_text(json.dumps(entry))
                (directory / "_SUCCESS.json").write_text(
                    json.dumps(dict(rows=1, revision=entry["revision"]))
                )
                row = normalize(
                    {
                        "prompt": "Reference evaluation problem",
                        "response": "evaluation solution",
                    },
                    repo=entry["repo"],
                    revision=entry["revision"],
                    config="default",
                    split="test",
                    index=0,
                    category="benchmarks",
                )
                pq.write_table(
                    pa.Table.from_pylist([row], schema=SCHEMA),
                    directory / "part-00000.parquet",
                )
            run(
                "dedup",
                "--input",
                root / "unified",
                "--output",
                root / "dedup",
                "--benchmarks",
                root / "evaluation",
                "--workers",
                "1",
                "--memory-limit",
                "256MB",
                "--no-fuzzy",
            )
            report = json.loads((root / "dedup/report.json").read_text())
            self.assertEqual(report["totals"]["keep"], 1)
            self.assertEqual(report["input_rows"], 3)
            self.assertEqual(
                sum(
                    pq.ParquetFile(p).metadata.num_rows
                    for p in (root / "dedup/clean").rglob("*.parquet")
                ),
                1,
            )
