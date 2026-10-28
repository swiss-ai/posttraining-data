import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq
from unify_local import rename, unify
from unified import SCHEMA


class UnificationTests(unittest.TestCase):
    def test_column_mapping_preserves_original_and_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.jsonl"
            original = {
                "task_text": "Write f",
                "completion_text": "def f(): pass",
                "extra": {"x": 1},
            }
            source.write_text(json.dumps(original) + "\n")
            self.assertEqual(
                unify(
                    [source],
                    root / "output",
                    "example/source",
                    "abc",
                    {"task_text": "prompt", "completion_text": "response"},
                ),
                1,
            )
            path = next((root / "output").rglob("*.parquet"))
            table = pq.ParquetFile(path).read()
            self.assertTrue(table.schema.equals(SCHEMA))
            row = table.to_pylist()[0]
            self.assertEqual(row["prompt"], original["task_text"])
            self.assertEqual(row["response"], original["completion_text"])
            self.assertEqual(json.loads(row["source_record_json"]), original)
            with self.assertRaises(FileExistsError):
                unify([source], root / "output", "example/source", "abc", {})

    def test_conflicting_mapping_fails(self):
        with self.assertRaises(ValueError):
            rename({"task": "a", "prompt": "b"}, {"task": "prompt"})

    def test_no_source_overwrite_or_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.jsonl"
            path.write_text("{}\n")
            with self.assertRaises(ValueError):
                unify([path], Path(temp), "../bad", "abc", {})
            self.assertEqual(path.read_text(), "{}\n")
