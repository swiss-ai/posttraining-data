import json
import unittest
from pathlib import Path
from preprocessors import PROCESSORS, get_processor
from unified import normalize_v2
from test_unified import ENTRY, FILE


class PreprocessorTests(unittest.TestCase):
    def test_every_catalog_dataset_has_distinct_class(self):
        catalog = json.loads(
            (Path(__file__).resolve().parents[1] / "adapter-policies.json").read_text()
        )
        repos = {e["repo"] for e in catalog["datasets"]}
        self.assertTrue(repos.issubset(PROCESSORS))
        self.assertEqual(len(repos), len({type(PROCESSORS[r]) for r in repos}))

    def test_terminal_conversation_aliases_and_serialization(self):
        processor = get_processor("Lite-Coder/LiteCoder-Terminal-SFT")
        entry = dict(ENTRY, repo=processor.repo, code_policy="labels")
        for key in ("messages", "conversation", "conversations", "trajectory"):
            for encoded in (False, True):
                messages = [
                    {"from": "human", "value": "Fix the program"},
                    {"from": "gpt", "value": "A" * 4000},
                ]
                row = {key: json.dumps(messages) if encoded else messages}
                self.assertEqual(processor.select(row, entry, FILE), "keep")
                result = normalize_v2(row, entry, FILE, 0)
                self.assertEqual(
                    json.loads(result["messages_json"])[1]["content"], "A" * 4000
                )
                self.assertEqual(json.loads(result["source_record_json"]), row)
        self.assertEqual(
            processor.select({"conversations": []}, entry, FILE), "excluded_unknown"
        )
        self.assertEqual(
            processor.select(
                {"conversation": [{"role": "user", "content": "x"}]}, entry, FILE
            ),
            "excluded_unknown",
        )

    def test_dataset_specific_sql_and_instruction_mapping(self):
        result = normalize_v2(
            {
                "question": "Count rows",
                "context": "CREATE TABLE x(a int)",
                "answer": "SELECT COUNT(*) FROM x",
            },
            dict(ENTRY, repo="b-mc2/sql-create-context"),
            FILE,
            0,
        )
        self.assertEqual(result["prompt"], "Count rows\n\nCREATE TABLE x(a int)")
        result = normalize_v2(
            {
                "instruction": "Use this",
                "input": "Ignore this seed",
                "response": "done",
            },
            dict(ENTRY, repo="bigcode/self-oss-instruct-sc2-exec-filter-50k"),
            FILE,
            0,
        )
        self.assertEqual(result["prompt"], "Use this")

    def test_benchmark_mapping_keeps_tests_and_alternative_prompts(self):
        processor = get_processor("openai/openai_humaneval")
        entry = dict(ENTRY, repo=processor.repo, category="benchmarks")
        row = {
            "prompt": "def f():",
            "canonical_solution": " return 1",
            "test": "assert f()==1",
        }
        result = processor.prepare(row, entry, FILE)
        self.assertEqual(result["response"], row["canonical_solution"])
        self.assertEqual(result["tests"], {"test": row["test"]})
        self.assertEqual(result["benchmark_prompts"], [row["prompt"]])

    def test_partition_selection_stays_in_dataset_class(self):
        processor = get_processor("open-r1/Mixture-of-Thoughts")
        entry = processor.configure(
            dict(
                ENTRY,
                files=[
                    dict(FILE, path="code/train.parquet"),
                    dict(FILE, path="math/train.parquet"),
                ],
            )
        )
        self.assertEqual([f["path"] for f in entry["files"]], ["code/train.parquet"])


if __name__ == "__main__":
    unittest.main()
