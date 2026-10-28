import copy
import json
import unittest
from openai_reasoning import tag_errors, extract_leading, result_reasoning_errors
from openai_chat import convert
from openai_validation import check_result
from openai_reference_answers import REPO, REVISION
from test_openai_sample_pilot import sample


class ReasoningIntegrityTests(unittest.TestCase):
    def test_plain_answers_without_reasoning_pass(self):
        result = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "Plain answer"},
                ]
            )
        )
        self.assertEqual(result["status"], "passed_sample")
        self.assertFalse(result["reasoning_by_message"])

    def test_unclosed_orphan_nested_and_wrong_order_fail_strict_export(self):
        for text in [
            "<think>unfinished",
            "reason</think>answer",
            "<think>A<think>B</think>",
            "</think>A<think>B",
        ]:
            with self.subTest(text=text):
                result = convert(
                    sample(
                        [
                            {"role": "user", "content": "Q"},
                            {"role": "assistant", "content": text},
                        ]
                    )
                )
                self.assertEqual(result["status"], "source_incomplete_reasoning")
                self.assertFalse(result["format_valid"])
                self.assertTrue(result["reasoning_integrity"])
                self.assertEqual(result["chat"]["messages"][-1]["content"], text)
                self.assertTrue(check_result(result))

    def test_multiple_complete_prefixes_and_case_variants_extract(self):
        self.assertEqual(
            extract_leading(" <Think>one</Think>\n<think>two</think>Answer"),
            ("Answer", ["one", "two"]),
        )

    def test_markers_in_code_and_json_are_literals(self):
        literals = [
            '```python\nprint("<think>")\n```',
            "The literal `<think>` is a marker.",
            '{"output":"<think>"}',
            "<function=editor><parameter=text><think></parameter></function>",
        ]
        for text in literals:
            with self.subTest(text=text):
                self.assertEqual(tag_errors(text), [])
        text = '<think>Reason\n```python\nprint("</think>")\n```\nMore reasoning</think>Answer'
        answer, reasons = extract_leading(text)
        self.assertEqual(answer, "Answer")
        self.assertIn("More reasoning", reasons[0])
        self.assertFalse(tag_errors(reasons[0]))

    def test_dedicated_reasoning_field_can_unwrap_missing_end_with_real_answer(self):
        result = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "Real answer",
                        "reasoning_content": "<think>Known reasoning field",
                    },
                ]
            )
        )
        self.assertEqual(result["status"], "passed_sample")
        self.assertEqual(
            result["reasoning_by_message"][0]["values"], ["Known reasoning field"]
        )
        self.assertFalse(result_reasoning_errors(result))

    def test_gold_reference_matches_language_and_preserves_original_source(self):
        raw = {
            "task_type": "verifiable_code",
            "prompt": "Solve using the programming language python:",
            "llm_response": "<think>unfinished",
            "gold_standard_solution": "```python\nprint(1)\n```",
            "score": 0.0,
        }
        snapshot = sample(
            [{"role": "user", "content": raw["prompt"]}], repo=REPO, raw=raw
        )
        snapshot["stored_sample"]["source_revision"] = REVISION
        before = copy.deepcopy(snapshot)
        result = convert(snapshot)
        self.assertEqual(result["status"], "passed_sample")
        self.assertEqual(
            result["chat"]["messages"][-1]["content"], raw["gold_standard_solution"]
        )
        self.assertEqual(
            result["reference_replacements"][0]["source_field"],
            "gold_standard_solution",
        )
        self.assertFalse(result["reasoning_by_message"])
        self.assertEqual(snapshot, before)
        self.assertFalse(check_result(result))
        for revision, gold in [
            (REVISION, "```cpp\nint main(){}\n```"),
            ("different", raw["gold_standard_solution"]),
            (REVISION, ""),
        ]:
            changed = copy.deepcopy(snapshot)
            changed["stored_sample"]["source_revision"] = revision
            changed_raw = dict(raw, gold_standard_solution=gold)
            changed["stored_sample"]["source_record_json"] = json.dumps(changed_raw)
            self.assertEqual(convert(changed)["status"], "source_incomplete_reasoning")


if __name__ == "__main__":
    unittest.main()
