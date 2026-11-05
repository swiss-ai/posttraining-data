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


class OpenCodeReasoning2Tests(unittest.TestCase):
    REPO = "nvidia/OpenCodeReasoning-2"

    def setUp(self):
        import os
        import tempfile
        import pyarrow as pa
        import pyarrow.parquet as pq
        from preprocessors.nvidia_opencodereasoning_2 import questions

        self.directory = tempfile.TemporaryDirectory()
        path = os.path.join(self.directory.name, "ocr2_questions.parquet")
        pq.write_table(
            pa.Table.from_pylist(
                [dict(dataset="taco", split="train", index=7, question="Sum two numbers.", labels_agree=True)]
            ),
            path,
        )
        self.previous = os.environ.get("OCR2_QUESTIONS")
        os.environ["OCR2_QUESTIONS"] = path
        questions.cache_clear()

    def tearDown(self):
        import os
        from preprocessors.nvidia_opencodereasoning_2 import questions

        if self.previous is None:
            os.environ.pop("OCR2_QUESTIONS", None)
        else:
            os.environ["OCR2_QUESTIONS"] = self.previous
        questions.cache_clear()
        self.directory.cleanup()

    def row(self, **fields):
        return dict(
            question="-", r1_generation="<think>\nadd\n</think>\nprint(sum(map(int, input().split())))",
            dataset="taco", split="train", index="7", **fields,
        )

    def test_question_is_restored_as_the_prompt(self):
        result = normalize_v2(self.row(), dict(ENTRY, repo=self.REPO), FILE, 0)
        self.assertEqual(result["prompt"], "Sum two numbers.")
        self.assertEqual(json.loads(result["source_record_json"])["question"], "-")

    def test_benchmark_split_questions_are_rejected(self):
        for dataset, split in (("apps", "test"), ("taco", "test"), ("code_contests", "valid")):
            with self.assertRaisesRegex(ValueError, "benchmark_split_question"):
                normalize_v2(
                    dict(self.row(), dataset=dataset, split=split), dict(ENTRY, repo=self.REPO), FILE, 0
                )

    def test_missing_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "question_not_restored"):
            normalize_v2(dict(self.row(), index="8"), dict(ENTRY, repo=self.REPO), FILE, 0)
