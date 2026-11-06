import copy
import json
import unittest
from openai_chat import find_terminal_payload, convert
from openai_legacy_dispatch import REPO, REVISION
from openai_swe_v3_protocol import HARNESS_ERROR, repair_terminal_messages
from test_openai_sample_pilot import sample
from openai_validation import check_result


class SweV3ProtocolTests(unittest.TestCase):
    def test_exact_harness_error_maps_to_tool_observation(self):
        messages = [
            {
                "role": "user",
                "content": "Format your response as JSON with keystrokes\nTask Description: Q",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"commands": [{"keystrokes": "pwd\n", "duration": 0.5}]}
                ),
            },
            {"role": "user", "content": HARNESS_ERROR},
            {
                "role": "assistant",
                "content": json.dumps({"commands": [], "task_complete": True}),
            },
        ]
        before = copy.deepcopy(messages)
        repaired, evidence = repair_terminal_messages(
            messages, REPO, REVISION, find_terminal_payload
        )
        self.assertEqual(messages, before)
        self.assertEqual(repaired[2], {"role": "tool", "content": HARNESS_ERROR})
        self.assertEqual(
            evidence[0]["execution_status"], "source_harness_reported_error"
        )
        snap = sample(messages, repo=REPO)
        snap["stored_sample"]["source_revision"] = REVISION
        result = convert(snap)
        self.assertEqual(result["status"], "passed_sample")
        self.assertTrue(
            any(
                p["kind"] == "explicit_terminal_harness_error_observation"
                for p in result["source_repairs"]
            )
        )
        self.assertEqual(check_result(result), [])
        self.assertTrue(
            any(
                m["role"] == "tool" and m["content"] == HARNESS_ERROR
                for m in result["chat"]["messages"]
            )
        )

    def test_other_errors_and_revisions_do_not_trigger_repair(self):
        messages = [
            {"role": "assistant", "content": '{"commands":[]}'},
            {"role": "user", "content": "ERROR: user question"},
        ]
        self.assertEqual(
            repair_terminal_messages(messages, REPO, REVISION, find_terminal_payload),
            (messages, []),
        )
        messages[1]["content"] = HARNESS_ERROR
        self.assertEqual(
            repair_terminal_messages(
                messages, REPO, "different", find_terminal_payload
            ),
            (messages, []),
        )
        messages[0]["content"] = "Ordinary discussion"
        self.assertEqual(
            repair_terminal_messages(messages, REPO, REVISION, find_terminal_payload),
            (messages, []),
        )

    def test_json_tail_recovery_requires_exact_complete_command_echoes(self):
        payload = {"commands": [{"keystrokes": "pwd\n", "duration": 0.5}]}
        text = json.dumps(payload) + "\n}"
        messages = [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "New Terminal Output:\nu@h# pwd\nresult\nu@h#"},
        ]
        repaired, evidence = repair_terminal_messages(
            messages, REPO, REVISION, find_terminal_payload
        )
        self.assertEqual(find_terminal_payload(repaired[0]["content"])[0], payload)
        self.assertTrue(repaired[0]["content"].startswith(text))
        self.assertFalse(evidence[0]["command_values_changed"])
        messages[1]["content"] = "New Terminal Output:\nu@h# ls\nresult"
        self.assertEqual(
            repair_terminal_messages(messages, REPO, REVISION, find_terminal_payload),
            (messages, []),
        )

    def test_intact_command_array_requires_complete_matching_echoes(self):
        from openai_swe_v3_protocol import recover_intact_command_array

        payload = {"commands": [{"keystrokes": "pwd\n", "duration": 0.5}]}
        malformed = json.dumps(payload)[:-1] + ', "analysis":"text"}]}'
        observation = "New Terminal Output:\nu@h# pwd\n/workspace\nu@h#"
        self.assertEqual(recover_intact_command_array(malformed, observation), payload)
        self.assertIsNone(
            recover_intact_command_array(
                malformed, observation.replace("# pwd", "# ls")
            )
        )
        self.assertIsNone(
            recover_intact_command_array(
                malformed + ' "task_complete":true', observation
            )
        )

    def test_completion_recovery_requires_source_confirmation(self):
        from openai_swe_v3_protocol import recover_confirmed_completion

        payload = {"commands": [], "task_complete": True, "analysis": "Done"}
        text = json.dumps(payload) + "\nProse summary after payload"
        observation = 'Current Terminal Screen:\nAre you sure you want to mark the task as complete? If so, include "task_complete": true in your JSON response again.'
        self.assertEqual(recover_confirmed_completion(text, observation), payload)
        self.assertIsNone(recover_confirmed_completion(text, "Ordinary user question"))
        self.assertIsNone(
            recover_confirmed_completion(
                json.dumps(dict(payload, commands=[{"keystrokes": "pwd"}])), observation
            )
        )

    def test_complete_json_with_only_extra_closers_preserves_multiline_commands(self):
        from openai_swe_v3_protocol import recover_complete_object_with_closing_suffix

        payload = {
            "commands": [
                {"keystrokes": "cat << EOF\nline one\nEOF\n", "duration": 1.0}
            ],
            "analysis": "source",
        }
        text = json.dumps(payload) + "]}"
        observation = "New Terminal Output:\nu@h# cat << EOF\n..."
        self.assertEqual(
            recover_complete_object_with_closing_suffix(text, observation), payload
        )
        self.assertIsNone(
            recover_complete_object_with_closing_suffix(
                json.dumps(payload) + "different action", observation
            )
        )
        self.assertIsNone(
            recover_complete_object_with_closing_suffix(text, "Ordinary user message")
        )
        duplicate = '{"commands":[],"commands":[{"keystrokes":"pwd"}]}]}'
        self.assertIsNone(
            recover_complete_object_with_closing_suffix(duplicate, observation)
        )


if __name__ == "__main__":
    unittest.main()
