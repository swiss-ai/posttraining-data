"""Source-specific repairs of damaged tool schemas and of the missing opening think tag."""

import json
import unittest

from chat_review import unresolved
from openai_chat import convert
from openai_training_payload import render_reasoning
from openai_validation import check_result
from openai_schema_repairs import repair_tool_schema

NEMOTRON, ULTRA, XCODER = (
    "nvidia/Nemotron-SWE-v1",
    "openbmb/UltraData-SFT-Agent-2609",
    "IIGroup/X-Coder-SFT-376k",
)


def sample(messages, tools=None, repo="test/plain"):
    return dict(
        dataset=repo,
        stored_sample={"source_record_json": "{}"},
        decoded_messages=messages,
        decoded_tools=tools,
        source_shard="fixture",
        row_offset_in_shard=0,
    )


def accepted(result):
    """The checks `standardise` applies after conversion: no errors from either."""
    if result["status"] != "passed_sample":
        return False
    render_reasoning(result, "sidecar")
    return not (unresolved(result) + check_result(result))


def tool(name, parameters):
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


def conversation(name, arguments):
    return [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": json.dumps(
                [{"id": "a", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]
            ),
        },
        {"role": "tool", "content": "ok", "tool_call_ids": ["a"]},
        {"role": "assistant", "content": "Done"},
    ]


# One struct shared by every tool: what a tool does not define is null
SHARED = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "c", "enum": None},
        "thought": None,
    },
    "required": ["command"],
    "additionalProperties": False,
}


class ArrowNullTests(unittest.TestCase):
    def test_null_nodes_removed_and_recorded(self):
        fixed, repairs = repair_tool_schema(NEMOTRON, "bash", SHARED)
        self.assertEqual(
            fixed["properties"], {"command": {"type": "string", "description": "c"}}
        )
        self.assertEqual(repairs[0]["count"], 2)
        self.assertIsNone(SHARED["properties"]["thought"], "input must not be modified")

    def test_null_default_is_a_value_and_stays(self):
        schema = {"type": "object", "properties": {"x": {"type": "string", "default": None}}}
        self.assertEqual(repair_tool_schema(NEMOTRON, "t", schema)[0], schema)

    def test_other_repositories_are_untouched(self):
        self.assertEqual(repair_tool_schema("test/plain", "bash", SHARED), (SHARED, []))

    def test_conversion_passes_only_with_the_repair(self):
        messages = conversation("bash", {"command": "ls"})
        self.assertFalse(accepted(convert(sample(messages, [tool("bash", SHARED)]))))
        result = convert(sample(messages, [tool("bash", SHARED)], NEMOTRON))
        self.assertTrue(accepted(result))
        self.assertEqual(result["schema_repairs"][0]["kind"], "removed_null_schema_nodes")

    def test_call_using_a_removed_property_is_still_rejected(self):
        messages = conversation("bash", {"command": "ls", "thought": "hmm"})
        result = convert(sample(messages, [tool("bash", SHARED)], NEMOTRON))
        self.assertFalse(accepted(result))


class CircularTests(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "patch": {
                "type": "object",
                "properties": {"schedule": "[Circular]", "name": {"type": "string", "description": "[Circular]"}},
            },
        },
    }

    def test_circular_nodes_become_unconstrained_and_text_is_kept(self):
        fixed, repairs = repair_tool_schema(ULTRA, "cron", self.SCHEMA)
        patch = fixed["properties"]["patch"]["properties"]
        self.assertEqual(patch["schedule"], {})
        self.assertEqual(patch["name"]["description"], "[Circular]")
        self.assertEqual(repairs[0]["nodes"], ["cron/properties/patch/properties/schedule"])

    def test_conversion_passes_with_the_repair(self):
        messages = conversation("cron", {"action": "list"})
        self.assertFalse(accepted(convert(sample(messages, [tool("cron", self.SCHEMA)]))))
        result = convert(sample(messages, [tool("cron", self.SCHEMA)], ULTRA))
        self.assertTrue(accepted(result))
        self.assertEqual(result["schema_repairs"][0]["kind"], "replaced_circular_schema_nodes")


class XCoderTests(unittest.TestCase):
    def messages(self, response):
        return [{"role": "user", "content": "Q"}, {"role": "assistant", "content": response}]

    def test_lone_closing_tag_ends_the_reasoning(self):
        result = convert(sample(self.messages("Okay, think.</think>\n\nAnswer"), repo=XCODER))
        self.assertEqual(result["status"], "passed_sample")
        self.assertEqual(result["chat"]["messages"][1]["content"], "Answer")
        self.assertEqual(result["reasoning_by_message"][0]["values"], ["Okay, think."])

    def test_explicit_opening_tag_is_unchanged(self):
        result = convert(sample(self.messages("<think>\nR\n</think>\n\nA"), repo=XCODER))
        self.assertEqual(result["status"], "passed_sample")
        self.assertEqual(result["reasoning_by_message"][0]["values"], ["R"])

    def test_unclosed_tag_is_still_rejected(self):
        result = convert(sample(self.messages("<think>never closed"), repo=XCODER))
        self.assertNotEqual(result["status"], "passed_sample")


class RejectedShellActionTests(unittest.TestCase):
    """A turn with several bash blocks that the harness rejected was never executed."""

    TWO = "Fix it:\n\n```bash\nls\n```\n\nThen check:\n\n```bash\npwd\n```"

    def notice(self, found):
        return (
            f"Please always provide EXACTLY ONE action in triple backticks, found {found} actions.\n\n"
            "Please format your action in triple backticks."
        )

    def conversation(self, found):
        return [
            {"role": "user", "content": "Task"},
            {"role": "assistant", "content": self.TWO},
            {"role": "user", "content": self.notice(found)},
            {"role": "assistant", "content": "Understood."},
        ]

    def test_rejected_turn_is_kept_as_text_without_a_command(self):
        result = convert(sample(self.conversation(2), repo="SWE-bench/SWE-smith-trajectories"))
        self.assertEqual(result["status"], "passed_sample")
        messages = [m for m in result["chat"]["messages"] if m["role"] != "system"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual(messages[1]["content"], self.TWO)
        self.assertFalse(messages[1].get("tool_calls"))
        self.assertTrue(messages[2]["content"].startswith("Please always provide EXACTLY ONE"))

    def test_without_a_matching_harness_notice_the_order_stays_unknown(self):
        for found in (3, 1):
            result = convert(sample(self.conversation(found), repo="SWE-bench/SWE-smith-trajectories"))
            self.assertNotEqual(result["status"], "passed_sample", found)
        plain = self.conversation(2)
        plain[2]["content"] = "ok"
        self.assertNotEqual(convert(sample(plain, repo="SWE-bench/SWE-smith-trajectories"))["status"], "passed_sample")

    def test_single_block_still_becomes_a_command(self):
        messages = [
            {"role": "user", "content": "Task"},
            {"role": "assistant", "content": "Run:\n\n```bash\nls\n```"},
            {"role": "user", "content": "<returncode>0</returncode>\n<output>\na\n</output>"},
            {"role": "assistant", "content": "Done."},
        ]
        result = convert(sample(messages, repo="SWE-bench/SWE-smith-trajectories"))
        self.assertEqual(result["status"], "passed_sample")
        messages = [m for m in result["chat"]["messages"] if m["role"] != "system"]
        self.assertTrue(messages[1].get("tool_calls"))


if __name__ == "__main__":
    unittest.main()
