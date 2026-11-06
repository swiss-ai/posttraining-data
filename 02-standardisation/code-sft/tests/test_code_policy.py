import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from code_policy import decision, configure
from unified import process_file
from test_unified import ENTRY, FILE


class CodePolicyTests(unittest.TestCase):
    def test_math_cannot_be_overridden(self):
        self.assertEqual(
            decision(
                {"domain": "math", "source": "codeforces"},
                {"code_policy": "labels"},
                FILE,
            ),
            "excluded_noncode",
        )

    def test_code_and_unknown(self):
        for domain in ["code", "prime_rl_code", "code_output_prediction"]:
            self.assertEqual(
                decision({"task_type": domain}, {"code_policy": "labels"}, FILE), "keep"
            )
        self.assertEqual(
            decision(
                {"prompt": "write Python to solve this math problem"},
                {"code_policy": "labels"},
                FILE,
            ),
            "excluded_unknown",
        )
        self.assertEqual(
            decision({"task": "swesmith-123_copy0"}, {"code_policy": "labels"}, FILE),
            "keep",
        )

    def test_categories(self):
        for category in [
            "code_sql",
            "code_tasks",
            "mixed_sft",
            "code_editing",
            "code_sft",
            "agentic",
            "code_agentic",
        ]:
            self.assertIn(
                configure(dict(repo="test/example", category=category, files=[]))[
                    "category"
                ],
                ["code_sft", "code_agentic"],
            )

    def test_filtered_resume_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(r)
                    for r in [
                        dict(domain="math"),
                        dict(domain="code", is_passed=False),
                        dict(domain="science"),
                        dict(foo="unknown"),
                    ]
                )
            )
            entry = dict(ENTRY, code_policy="labels", code_policy_version="code-only-1")
            with patch("unified.hf_hub_download", return_value=str(source)) as download:
                result = process_file(Path(tmp) / "out", entry, FILE)
                self.assertEqual(
                    (
                        result["rows"],
                        result["source_rows"],
                        result["excluded_noncode"],
                        result["excluded_unknown"],
                    ),
                    (1, 4, 2, 1),
                )
                self.assertEqual(process_file(Path(tmp) / "out", entry, FILE), result)
                self.assertEqual(download.call_count, 1)


if __name__ == "__main__":
    unittest.main()
