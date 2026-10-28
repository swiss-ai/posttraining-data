import copy
import unittest
from test_openai_sample_pilot import sample
from openai_chat import convert
from openai_training_payload import render_reasoning
from openai_validation import check_result


class TrainingPayloadTests(unittest.TestCase):
    def test_balanced_reasoning_and_answer_in_same_assistant_message(self):
        result = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "<think>Reason</think>Answer"},
                ]
            )
        )
        render_reasoning(result)
        self.assertEqual(
            result["chat"]["messages"][1]["content"],
            "<think>\nReason\n</think>\n\nAnswer",
        )
        self.assertFalse(check_result(result))
        before = copy.deepcopy(result)
        render_reasoning(result)
        self.assertEqual(result, before)

    def test_answer_only_is_unchanged(self):
        result = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "Answer"},
                ]
            )
        )
        before = copy.deepcopy(result["chat"])
        render_reasoning(result)
        self.assertEqual(result["chat"], before)

    def test_sidecar_mode_and_double_render_mode_guard(self):
        result = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "<think>Reason</think>Answer"},
                ]
            )
        )
        render_reasoning(result, "sidecar")
        self.assertEqual(result["chat"]["messages"][1]["content"], "Answer")
        with self.assertRaises(ValueError):
            render_reasoning(result, "inline")

    def test_multiple_segments_have_single_outer_pair(self):
        result = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "<think>One</think><think>Two</think>Answer",
                    },
                ]
            )
        )
        render_reasoning(result)
        self.assertEqual(
            result["chat"]["messages"][1]["content"],
            "<think>\nOne\n\nTwo\n</think>\n\nAnswer",
        )
        self.assertFalse(check_result(result))

    def test_code_example_cannot_hide_generated_closing_delimiter(self):
        reason = 'Example:\n```python\ns = """\n```python2\na = 1\n```\n```python3\nb = 2\n```\n"""\n```\nContinue investigating.'
        result = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "Answer",
                        "reasoning_content": reason,
                    },
                ]
            )
        )
        render_reasoning(result)
        self.assertFalse(check_result(result))
        self.assertEqual(
            result["chat"]["messages"][1]["content"],
            "<think>\n" + reason + "\n</think>\n\nAnswer",
        )
        result["chat"]["messages"][1]["content"] = result["chat"]["messages"][1][
            "content"
        ].replace("\n</think>\n\n", "\n", 1)
        self.assertTrue(check_result(result))

    def test_recorded_boundary_does_not_allow_bad_answer_tags(self):
        result = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "Answer",
                        "reasoning_content": "Reason",
                    },
                ]
            )
        )
        render_reasoning(result)
        result["chat"]["messages"][1]["content"] += "\n<think>Unfinished"
        self.assertTrue(check_result(result))

    def test_sidecar_tampering_is_detected(self):
        result = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "Answer",
                        "reasoning_content": "Reason",
                    },
                ]
            )
        )
        render_reasoning(result)
        result["reasoning_by_message"][0]["values"] = ["Different reasoning"]
        self.assertTrue(check_result(result))


if __name__ == "__main__":
    unittest.main()
