import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from unified import SCHEMA, normalize_v2, process_file, file_partition

ENTRY = dict(
    repo="test/example",
    revision="abc",
    category="code_sft",
    license="unknown",
    license_status="unknown",
    license_notes="Not declared",
    synthetic_status="mixed",
    synthetic_notes="Mixed sources",
    metadata_source="test",
)
FILE = dict(path="data/train.jsonl", source_config="default", source_split="train")


class UnifiedTests(unittest.TestCase):
    def test_context_and_source_preservation(self):
        row = dict(instruction="Change code", input="x=1", output="x=2", license="mit")
        normalized = normalize_v2(row, ENTRY, FILE, 0)
        self.assertEqual(normalized["prompt"], "Change code\n\nx=1")
        self.assertEqual(json.loads(normalized["source_record_json"]), row)
        self.assertIsNone(normalized["is_synthetic"])
        self.assertEqual(json.loads(normalized["source_license_json"]), "mit")
        pa.Table.from_pylist([normalized], schema=SCHEMA)

    def test_sharegpt_and_serialized_messages(self):
        row = {
            "conversations": json.dumps(
                [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]
            )
        }
        result = normalize_v2(row, ENTRY, FILE, 1)
        self.assertEqual(json.loads(result["messages_json"])[1]["role"], "assistant")
        self.assertEqual(result["prompt"], "q")
        self.assertEqual(result["response"], "a")

    def test_missing_prompt_is_not_invented(self):
        row = {"question": "-", "r1_generation": "<think>x</think>answer"}
        result = normalize_v2(row, ENTRY, FILE, 0)
        self.assertIsNone(result["prompt"])
        self.assertEqual(result["normalization_status"], "needs_review")
        self.assertEqual(result["response"], row["r1_generation"])

    def test_same_source_id_different_file_has_distinct_id(self):
        a = normalize_v2({"id": "duplicate"}, ENTRY, FILE, 0)
        b = normalize_v2({"id": "duplicate"}, ENTRY, dict(FILE, path="other.jsonl"), 0)
        self.assertNotEqual(a["id"], b["id"])

    def test_all_rows_preserved_resume_and_parquet(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            original = [
                dict(question="q", answer="a", is_passed=False),
                dict(question="q", answer="b", is_passed=True),
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in original))
            root = Path(directory) / "output"
            with patch("unified.hf_hub_download", return_value=str(path)) as download:
                result = process_file(root, ENTRY, FILE)
                self.assertEqual(result["rows"], 2)
                self.assertEqual(process_file(root, ENTRY, FILE)["rows"], 2)
                self.assertEqual(download.call_count, 1)
            shard = file_partition(root, ENTRY, FILE) / result["files"][0]["name"]
            data = pq.ParquetFile(shard).read().to_pylist()
            self.assertEqual(
                [json.loads(r["source_record_json"]) for r in data], original
            )
            self.assertTrue(pq.ParquetFile(shard).schema_arrow.equals(SCHEMA))


if __name__ == "__main__":
    unittest.main()
