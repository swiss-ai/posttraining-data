import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import dedup
import fuzzy
from standardize import SCHEMA
from test_dedup import row


class FuzzyTests(unittest.TestCase):
    def setUp(self):
        self.prompt = " ".join(f"problem{i}" for i in range(150))
        self.answer = " ".join(f"solution{i}" for i in range(150))

    def test_sketch_and_verified_fuzzy_benchmark(self):
        reference = json.dumps({"dataset": "benchmark", "id": "1"})
        fuzzy.build_benchmarks({self.prompt: reference}, 0.8)
        match = fuzzy.benchmark_match(row(self.prompt.replace("problem80", "changed")))
        self.assertEqual(match[0], reference)
        self.assertGreater(match[2], 0.8)
        self.assertIsNone(fuzzy.benchmark_match(row("Unrelated short prompt."))[0])
        a = fuzzy.bands(fuzzy.shingles(self.prompt))
        b = fuzzy.bands(fuzzy.shingles(self.prompt.replace("problem80", "changed")))
        self.assertTrue(any(x == y for x, y in zip(a, b)))

    def test_guard_different_answers_and_short_samples(self):
        a = row(self.prompt, self.answer)
        b = row(
            self.prompt.replace("problem80", "changed"),
            self.answer.replace("solution80", "changed"),
        )
        self.assertGreater(fuzzy.verify_pair(a, b, 0.9), 0.9)
        self.assertIsNone(
            fuzzy.verify_pair(
                row(self.prompt, "return True"), row(self.prompt, "return False"), 0.9
            )
        )
        self.assertIsNone(
            fuzzy.verify_pair(a, row(self.prompt, self.answer, tools=["shell"]), 0.9)
        )
        self.assertEqual(fuzzy.signature(row("a", "b"))[0], [])
        a = row(
            self.prompt,
            self.answer,
            messages=[
                {
                    "role": "assistant",
                    "content": self.answer,
                    "tool_calls": [{"name": "shell", "arguments": "ls"}],
                }
            ],
        )
        b = row(
            self.prompt,
            self.answer,
            messages=[
                {
                    "role": "assistant",
                    "content": self.answer,
                    "tool_calls": [{"name": "shell", "arguments": "rm x"}],
                }
            ],
        )
        self.assertIsNone(fuzzy.verify_pair(a, b, 0.9))

    def test_disk_lsh_removal_and_false_candidate_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "_work").mkdir()
            source = root / "samples.parquet"
            samples = [
                row(self.prompt, self.answer),
                row(self.prompt.replace("problem80", "changed"), self.answer, index=1),
                row(self.prompt, "return False", index=2),
                row(self.prompt, "return True", index=3),
            ]
            pq.write_table(pa.Table.from_pylist(samples, schema=SCHEMA), source)
            records = []
            for i, sample in enumerate(samples):
                bands, size, guard = fuzzy.signature(sample)
                records.append(
                    dict(
                        file_index=0,
                        row_index=i,
                        id=sample["id"],
                        source_dataset=sample["source_dataset"],
                        sample_hash=dedup.sample_hash(sample),
                        benchmark_match=None,
                        match_kind=None,
                        fuzzy_bands=bands,
                        fuzzy_size=size,
                        fuzzy_guard=guard,
                    )
                )
            path = root / "fingerprints.parquet"
            pq.write_table(pa.Table.from_pylist(records, schema=dedup.FP_SCHEMA), path)
            _, totals = dedup.classify([str(path)], root, 2, "256MB", [source], 0.9)
            self.assertEqual(totals, {"keep": 3, "fuzzy_duplicate": 1})
            decisions = pq.read_table(root / "decisions.parquet").to_pylist()
            self.assertEqual(decisions[1]["keeper_row_index"], 0)
            self.assertGreaterEqual(decisions[1]["fuzzy_similarity"], 0.9)
            # Re-running the same stage must not reuse stale candidate tables or fail schema updates.
            _, repeated = dedup.classify([str(path)], root, 2, "256MB", [source], 0.9)
            self.assertEqual(repeated, totals)

    def test_parallel_verification_preserves_candidate_order(self):
        left = json.dumps(row(self.prompt, self.answer))
        right = json.dumps(
            row(self.prompt.replace("problem80", "changed"), self.answer)
        )
        batch = pa.record_batch(
            {"a": [0, 1], "b": [2, 3], "left": [left, left], "right": [right, right]}
        )
        expected = list(fuzzy.ordered_verification(iter([batch]), 0.9, 1, 2048))
        actual = list(fuzzy.ordered_verification(iter([batch]), 0.9, 2, 2048))
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
