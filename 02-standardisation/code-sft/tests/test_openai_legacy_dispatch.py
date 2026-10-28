import copy
import json
import unittest
from test_openai_sample_pilot import sample, call, definition
from openai_chat import convert
from openai_validation import check_result
from openai_legacy_dispatch import REPO, REVISION, dispatch_native
from chat_review import unresolved


def source(messages, tools=None, revision=REVISION):
    result = sample(messages, tools, repo=REPO)
    result["stored_sample"]["source_revision"] = revision
    return result


class LegacyDispatchTests(unittest.TestCase):
    def test_native_payload_ids_results_and_reasoning_preserved(self):
        args = {"cmd": "printf ' a\\n'", "count": "3", "extra": {"x": [1, False]}}
        snap = source(
            [
                {"role": "user", "content": "Q"},
                {
                    "role": "assistant",
                    "content": "<think>reason</think>",
                    "tool_calls": [call("original-id", args)],
                },
                {
                    "role": "tool",
                    "tool_call_id": "original-id",
                    "content": " original output\n",
                },
                {"role": "assistant", "content": "Done"},
            ]
        )
        before = copy.deepcopy(snap)
        result = convert(snap)
        action = next(m for m in result["chat"]["messages"] if m.get("tool_calls"))
        function = action["tool_calls"][0]["function"]
        self.assertEqual(function["name"], "source_tool_call")
        self.assertEqual(
            json.loads(function["arguments"]), {"name": "bash", "arguments": args}
        )
        self.assertEqual(action["tool_calls"][0]["id"], "original-id")
        self.assertIn(
            {
                "role": "tool",
                "tool_call_id": "original-id",
                "content": " original output\n",
            },
            result["chat"]["messages"],
        )
        reason = result["reasoning_by_message"][0]
        self.assertEqual(
            result["chat"]["messages"][reason["message_index"]]["role"], "assistant"
        )
        self.assertEqual(snap, before)
        self.assertEqual(check_result(result) + unresolved(result), [])
        self.assertEqual(
            result["legacy_dispatch"]["native_tool_provenance"][0]["origin"],
            "observed_signature_only",
        )

    def test_complete_unlinked_batch_preserves_children_and_observations(self):
        snap = source(
            [
                {"role": "user", "content": "Q"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [call("a", {"x": 1}), call("b", {"x": 2})],
                },
                {"role": "tool", "content": "second?"},
                {"role": "tool", "content": "first?"},
                {"role": "assistant", "content": "Done"},
            ]
        )
        result = convert(snap)
        action = next(m for m in result["chat"]["messages"] if m.get("tool_calls"))
        children = json.loads(action["tool_calls"][0]["function"]["arguments"])["calls"]
        self.assertEqual([c["name"] for c in children], ["source_tool_call"] * 2)
        self.assertEqual(
            [c["arguments"] for c in children],
            [
                {"name": "bash", "arguments": {"x": 1}},
                {"name": "bash", "arguments": {"x": 2}},
            ],
        )
        self.assertEqual([c["source_call_id"] for c in children], ["a", "b"])
        observation = json.loads(
            next(
                m["content"] for m in result["chat"]["messages"] if m["role"] == "tool"
            )
        )
        self.assertEqual(observation["association"], "batch_only")
        self.assertEqual(
            [m["content"] for m in observation["source_observations"]],
            ["second?", "first?"],
        )
        self.assertEqual(check_result(result) + unresolved(result), [])

    def test_released_native_schema_is_still_enforced(self):
        tool = definition()
        tool["function"]["parameters"] = {"type": "object", "required": ["command"]}
        result = convert(
            source(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": None, "tool_calls": [call()]},
                ],
                [tool],
            )
        )
        self.assertNotIn("legacy_dispatch", result)
        self.assertTrue(any("command" in error for error in check_result(result)))

    def test_pinned_revision_guard(self):
        result = convert(
            source(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": None, "tool_calls": [call()]},
                ],
                revision="other",
            )
        )
        self.assertNotIn("legacy_dispatch", result)
        self.assertIn("original_tool_schema_unavailable", unresolved(result))

    def test_missing_observation_is_still_rejected(self):
        result = convert(
            source(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": None, "tool_calls": [call()]},
                    {"role": "assistant", "content": "Done"},
                ]
            )
        )
        self.assertEqual(result["status"], "needs_adapter")

    def test_runtime_forwards_exact_name_and_arguments_to_supplied_executor(self):
        recorded = []
        args = {"name": "native_alias", "arguments": {"code": "not executed by test"}}
        value = dispatch_native(
            args,
            lambda name, payload: recorded.append((name, payload)) or "observation",
        )
        self.assertEqual(value, "observation")
        self.assertEqual(recorded, [("native_alias", args["arguments"])])
        with self.assertRaises(ValueError):
            dispatch_native({"name": "tool", "arguments": []}, lambda *_: None)


if __name__ == "__main__":
    unittest.main()
