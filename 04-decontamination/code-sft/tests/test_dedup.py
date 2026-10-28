import copy
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(2)
pa.set_io_thread_count(2)

import dedup
from benchmarks import adapt
from standardize import SCHEMA, normalize


def row(
    prompt="Implement a novel example.",
    answer="pass",
    repo="org/train",
    index=0,
    **extra,
):
    return normalize(
        dict(prompt=prompt, response=answer, **extra),
        repo=repo,
        revision="abc",
        config="default",
        split="test",
        index=index,
        category="benchmarks" if repo == "org/benchmark" else "code_sft",
    )


class DedupTests(unittest.TestCase):
    def test_benchmark_directory_accepts_direct_and_parent_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmarks = root / "benchmarks"
            repo = benchmarks / "org" / "benchmark"
            repo.mkdir(parents=True)
            (repo / "source.json").write_text("{}")
            (repo / "_SUCCESS.json").write_text(json.dumps({"rows": 1}))
            shard = repo / "part-00000.parquet"
            pq.write_table(
                pa.Table.from_pylist([row(repo="org/benchmark")], schema=SCHEMA), shard
            )
            self.assertEqual(dedup.completed_files(root, "benchmarks"), [shard])
            self.assertEqual(dedup.completed_files(benchmarks, "benchmarks"), [shard])
            self.assertEqual(dedup.benchmark_directory(root), benchmarks)
            self.assertEqual(dedup.benchmark_directory(benchmarks), benchmarks)

    def test_duplicates_cross_sources_and_aliases(self):
        a = row()
        b = row(repo="other/train", index=100)
        self.assertEqual(dedup.sample_hash(a), dedup.sample_hash(b))
        b = row(
            messages=[
                {"from": "human", "value": "Implement a novel example."},
                {"from": "gpt", "value": "pass"},
            ]
        )
        self.assertEqual(dedup.sample_hash(a), dedup.sample_hash(b))
        b["messages_json"] = json.dumps(
            [
                {
                    "role": "user",
                    "from": "human",
                    "content": "Implement a novel example.",
                    "value": "Implement a novel example.",
                },
                {
                    "role": "assistant",
                    "from": "gpt",
                    "content": "pass",
                    "value": "pass",
                },
            ]
        )
        self.assertEqual(dedup.sample_hash(a), dedup.sample_hash(b))

    def test_preserve_distinct_answers_tools_tests_and_code(self):
        original = row()
        for changed in [
            row(answer="return 1"),
            row(prompt="implement a novel example."),
            row(tools=[{"name": "shell"}]),
            row(test="assert f() == 2"),
            row(reasoning="Because..."),
        ]:
            self.assertNotEqual(dedup.sample_hash(original), dedup.sample_hash(changed))
        self.assertNotEqual(
            dedup.sample_hash(row(answer="if True:\n    x=1")),
            dedup.sample_hash(row(answer="if True:\nx=1")),
        )

    def test_empty_views_do_not_collapse_opaque_rows(self):
        a = row(prompt=None, answer=None, opaque="a")
        b = row(prompt=None, answer=None, opaque="b")
        self.assertNotEqual(dedup.sample_hash(a), dedup.sample_hash(b))

    def test_benchmark_adapters_and_file_ids(self):
        from code_policy import configure

        self.assertEqual(
            configure({"repo": "org/benchmark", "category": "benchmarks"})["category"],
            "benchmarks",
        )
        entry = {
            "repo": "org/benchmark",
            "prompt_fields": ["problem_statement"],
            "response_fields": ["patch"],
        }
        a = adapt(
            {"problem_statement": "Fix issue", "patch": "diff", "instance_id": "same"},
            entry,
            "a.jsonl",
        )
        b = adapt(
            {"problem_statement": "Fix issue", "patch": "diff", "instance_id": "same"},
            entry,
            "b.jsonl",
        )
        kwargs = dict(
            repo="org/benchmark",
            revision="abc",
            config="default",
            split="test",
            index=0,
            category="benchmarks",
        )
        self.assertNotEqual(normalize(a, **kwargs)["id"], normalize(b, **kwargs)["id"])
        self.assertEqual(a["prompt"], "Fix issue")
        self.assertEqual(a["benchmark_prompts"], ["Fix issue"])
        with self.assertRaises(ValueError):
            adapt({"irrelevant": 1}, entry, "bad.jsonl")

    def test_matching_classification_materialization_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark = root / "benchmark.parquet"
            text = "Write a function that selects all pairs whose sum is divisible by seven and returns them in sorted order."
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        row(text, repo="org/benchmark"),
                        row("Short task.", repo="org/benchmark", index=1),
                    ],
                    schema=SCHEMA,
                ),
                benchmark,
            )
            dedup.build_matcher([benchmark], 80)
            self.assertEqual(dedup.contamination(row("Short task."))[1], "exact_text")
            self.assertIsNone(dedup.contamination(row("Prefix Short task. suffix"))[0])
            contaminated = row(
                "Please solve:\n" + text.replace(" ", "  ") + "\nThanks.", index=2
            )
            self.assertEqual(dedup.contamination(contaminated)[1], "contained_text")
            later_turn = row(messages=[{"role": "assistant", "content": text}])
            self.assertIsNotNone(dedup.contamination(later_turn)[0])
            source = root / "input.parquet"
            samples = [
                row(index=0),
                row(repo="other/train", index=1),
                contaminated,
                row(answer="different", index=3),
            ]
            pq.write_table(pa.Table.from_pylist(samples, schema=SCHEMA), source)
            work = root / "output" / "_work"
            work.mkdir(parents=True)
            target = work / "fp.parquet"
            task = (0, str(source), str(target), "test", 2)
            dedup.fingerprint(task)
            mtime = target.stat().st_mtime_ns
            dedup.fingerprint(task)
            self.assertEqual(mtime, target.stat().st_mtime_ns)
            summary, totals = dedup.classify([str(target)], work.parent, 2, "128MB")
            self.assertEqual(totals, {"keep": 2, "duplicate": 1, "benchmark": 1})
            decision_rows = pq.read_table(work.parent / "decisions.parquet").to_pylist()
            self.assertEqual(decision_rows[1]["keeper_id"], samples[0]["id"])
            result = dedup.materialize(
                (0, str(source), "input.parquet", str(work.parent), 2)
            )
            self.assertEqual(result["kept_rows"], 2)
            actual = pq.read_table(work.parent / "clean" / "input.parquet")
            self.assertEqual(actual.to_pylist(), [samples[0], samples[3]])
            self.assertTrue(actual.schema.equals(SCHEMA))

    def test_incomplete_inputs_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "code_sft" / "org" / "train"
            repo.mkdir(parents=True)
            (repo / "source.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                dedup.completed_files(root, "code_sft")

    def test_completed_zero_row_source_does_not_block_other_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, count in [("empty", 0), ("populated", 1)]:
                repo = root / "code_sft" / "org" / name
                repo.mkdir(parents=True)
                (repo / "source.json").write_text("{}")
                (repo / "_SUCCESS.json").write_text(json.dumps({"rows": count}))
                if count:
                    pq.write_table(
                        pa.Table.from_pylist([row()], schema=SCHEMA),
                        repo / "part-00000.parquet",
                    )
            self.assertEqual(len(dedup.completed_files(root, "code_sft")), 1)

    def test_unavailable_sources_are_explicit_and_completed_sources_included(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = [
                {"repo": "org/ready", "category": "code_sft"},
                {"repo": "org/gated", "category": "code_sft"},
            ]
            (root / "catalog.json").write_text(json.dumps({"datasets": entries}))
            for entry in entries:
                repo = root / "code_sft" / entry["repo"]
                repo.mkdir(parents=True)
                (repo / "source.json").write_text(json.dumps(entry))
            ready = root / "code_sft/org/ready"
            (ready / "_SUCCESS.json").write_text(json.dumps({"rows": 1}))
            pq.write_table(
                pa.Table.from_pylist([row()], schema=SCHEMA),
                ready / "part-00000.parquet",
            )
            gated = root / "code_sft/org/gated"
            (gated / "_FAILED.json").write_text(
                json.dumps({"missing_files": ["data.jsonl"]})
            )
            with self.assertRaisesRegex(ValueError, "Missing completed"):
                dedup.unavailable_repos(root, ["code_sft"], [])
            missing = dedup.unavailable_repos(
                root, ["code_sft"], ["org/gated", "org/ready"]
            )
            self.assertEqual([e["repo"] for e in missing], ["org/gated"])
            files = dedup.completed_files(
                root, "code_sft", {e["repo"] for e in missing}
            )
            self.assertEqual(len(files), 1)
            with self.assertRaisesRegex(ValueError, "not in input catalog"):
                dedup.unavailable_repos(root, ["code_sft"], ["org/typo"])
            (gated / "_FAILED.json").unlink()
            (gated / "_SUCCESS.json").write_text(json.dumps({"rows": 0}))
            self.assertEqual(
                dedup.unavailable_repos(root, ["code_sft"], ["org/gated"]), []
            )


if __name__ == "__main__":
    unittest.main()
