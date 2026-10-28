import copy
import json
import unittest
from unittest.mock import patch
from openai_chat import convert, validate_chat


def sample(messages, tools=None, repo="test/plain", raw=None):
    return dict(
        dataset=repo,
        stored_sample={"source_record_json": json.dumps(raw or {})},
        decoded_messages=messages,
        decoded_tools=tools,
        source_shard="fixture",
        row_offset_in_shard=0,
    )


def call(cid="a", args=None):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps(args or {})},
    }


def definition():
    return {
        "type": "function",
        "function": {"name": "bash", "parameters": {"type": "object"}},
    }


class PilotTests(unittest.TestCase):
    def test_reasoning_deduplication(self):
        r = convert(
            sample(
                [
                    {"role": "HUMAN", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "<think>Reason</think>Answer",
                        "thought": "<think>Reason</think>Answer",
                    },
                ]
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        self.assertEqual(r["chat"]["messages"][1]["content"], "Answer")
        self.assertEqual(r["reasoning_by_message"][0]["values"], ["Reason"])

    def test_closing_only_reasoning_is_dataset_specific(self):
        ms = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "Reason</think>Answer"},
        ]
        r = convert(sample(ms, repo="microsoft/rStar-Coder"))
        self.assertEqual(r["chat"]["messages"][1]["content"], "Answer")
        self.assertEqual(
            convert(sample(ms))["chat"]["messages"][1]["content"], ms[1]["content"]
        )

    def test_link_tool_result(self):
        r = convert(
            sample(
                [
                    {"role": "user", "content": "Run"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": json.dumps([call()]),
                    },
                    {"role": "tool", "content": "result", "tool_call_ids": ["a"]},
                    {"role": "assistant", "content": "Done"},
                ],
                [definition()],
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        self.assertEqual(r["chat"]["messages"][2]["tool_call_id"], "a")

    def test_invalid_call_and_orphan_result(self):
        bad = call()
        bad["function"]["arguments"] = "bad"
        r = convert(
            sample(
                [
                    {"role": "user", "content": "Run"},
                    {"role": "assistant", "content": None, "tool_calls": [bad]},
                    {"role": "tool", "content": "result", "tool_call_id": "wrong"},
                ]
            )
        )
        self.assertEqual(r["status"], "needs_adapter")
        self.assertTrue(any("arguments" in x for x in r["issues"]))
        self.assertTrue(any("Unmatched" in x for x in r["issues"]))

    def test_distinct_reasoning_is_preserved_and_flagged(self):
        r = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "<think>A</think>answer",
                        "thought": "B",
                    },
                ]
            )
        )
        self.assertEqual(r["reasoning_by_message"][0]["values"], ["A", "B"])
        self.assertTrue(r["training_review_required"])

    def test_missing_source_target_is_not_fabricated(self):
        r = convert(sample([{"role": "user", "content": "Q"}]))
        self.assertEqual(r["status"], "source_missing_target")
        self.assertFalse(r["format_valid"])
        self.assertEqual(r["chat"]["messages"], [{"role": "user", "content": "Q"}])

    def test_terminal_batch_keystrokes_and_duration_exact(self):
        commands = [
            {"keystrokes": "cat <<'EOF'\n<&>\nEOF\n", "duration": 0.125},
            {"keystrokes": "C-c", "duration": 0.1},
        ]
        payload = {"analysis": "Reason", "plan": "Plan", "commands": commands}
        r = convert(
            sample(
                [
                    {
                        "role": "user",
                        "content": "Format your response as JSON with keystrokes\nTask Description: Task",
                    },
                    {"role": "assistant", "content": json.dumps(payload)},
                    {"role": "user", "content": "Current Terminal Screen:\nResult"},
                    {
                        "role": "assistant",
                        "content": json.dumps({"commands": [], "task_complete": True}),
                    },
                ]
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        action = next(m for m in r["chat"]["messages"] if m.get("tool_calls"))
        self.assertEqual(
            json.loads(action["tool_calls"][0]["function"]["arguments"])["commands"],
            commands,
        )
        self.assertEqual(
            next(m for m in r["chat"]["messages"] if m["role"] == "tool")["content"],
            "Current Terminal Screen:\nResult",
        )
        self.assertEqual(r["reasoning_by_message"][0]["values"], ["Reason"])

    def test_xml_code_whitespace_preserved(self):
        code = '\n  x = "<&>"\n'
        r = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "<function=editor><parameter=file_text>"
                        + code
                        + "</parameter></function>",
                    },
                    {"role": "user", "content": "EXECUTION RESULT of [editor]:\nOK"},
                    {"role": "assistant", "content": "Done"},
                ],
                repo="Lego-X/SWE-Lego-Real-Data",
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        action = next(m for m in r["chat"]["messages"] if m.get("tool_calls"))
        self.assertEqual(
            json.loads(action["tool_calls"][0]["function"]["arguments"])["file_text"],
            code,
        )

    def test_parallel_unknown_order_rejected_outside_known_release(self):
        r = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call("a"), call("b")],
                    },
                    {"role": "tool", "content": "unknown"},
                ]
            )
        )
        self.assertEqual(r["status"], "needs_adapter")
        self.assertTrue(any("Ambiguous" in x for x in r["issues"]))

    def test_parallel_results_batched_without_invented_pairing(self):
        r = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call("a"), call("b")],
                    },
                    {"role": "tool", "content": "first"},
                    {"role": "tool", "content": "second"},
                    {"role": "assistant", "content": "Done"},
                ],
                [definition()],
                repo="nvidia/Open-SWE-Traces",
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        batch = next(m for m in r["chat"]["messages"] if m.get("tool_calls"))[
            "tool_calls"
        ][0]
        self.assertEqual(batch["function"]["name"], "source_tool_batch")
        self.assertEqual(
            [
                c["source_call_id"]
                for c in json.loads(batch["function"]["arguments"])["calls"]
            ],
            ["a", "b"],
        )
        result = next(m for m in r["chat"]["messages"] if m["role"] == "tool")
        self.assertEqual(result["tool_call_id"], batch["id"])
        observations = json.loads(result["content"])
        self.assertEqual(observations["association"], "batch_only")
        self.assertEqual(
            [o["content"] for o in observations["source_observations"]],
            ["first", "second"],
        )
        self.assertFalse(any("source order" in w for w in r["warnings"]))

    def test_trailing_observation_preserved_without_fake_answer(self):
        r = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": None, "tool_calls": [call()]},
                    {"role": "tool", "content": "submitted", "tool_call_id": "a"},
                ],
                [definition()],
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        self.assertTrue(r["ends_with_tool_call"])
        self.assertEqual(r["trailing_observations"][0]["content"], "submitted")
        self.assertEqual(len(r["chat"]["messages"]), 2)

    def test_codefeedback_keeps_genuine_user_feedback(self):
        r = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "```python\nprint(1)\n```"},
                    {"role": "user", "content": "Execution result: 1"},
                    {"role": "assistant", "content": "Done"},
                    {"role": "user", "content": "Make it faster"},
                    {"role": "assistant", "content": "Try this"},
                ],
                repo="m-a-p/Code-Feedback",
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        self.assertEqual(
            next(m for m in r["chat"]["messages"] if m["content"] == "Make it faster")[
                "role"
            ],
            "user",
        )
        self.assertEqual(
            next(
                m
                for m in r["chat"]["messages"]
                if m["content"] == "Execution result: 1"
            )["role"],
            "tool",
        )

    def test_rejected_malformed_terminal_invocation(self):
        bad = '<think>Reason",\n "plan": "Test", "commands": [{"keystrokes": "pwd\\n", "duration": 0.1}], "task_complete": false}\n</think>'
        r = convert(
            sample(
                [
                    {
                        "role": "user",
                        "content": "Format your response as JSON with keystrokes\nTask Description: Q",
                    },
                    {"role": "assistant", "content": bad},
                    {
                        "role": "user",
                        "content": "Previous response had parsing errors:\nERROR: No valid JSON found in response",
                    },
                    {
                        "role": "assistant",
                        "content": '{"commands":[],"task_complete":true}',
                    },
                ],
                repo="nvidia/Nemotron-Terminal-Corpus",
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        self.assertTrue(any("rejected" in w for w in r["warnings"]))
        self.assertTrue(
            any(
                m["role"] == "tool" and "parsing errors" in m["content"]
                for m in r["chat"]["messages"]
            )
        )

    def test_converter_does_not_mutate_input(self):
        s = sample(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "Answer"},
            ]
        )
        before = copy.deepcopy(s)
        convert(s)
        self.assertEqual(s, before)

    def test_source_argument_spelling_aliases(self):
        tool = {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
        }
        action = {
            "id": "a",
            "type": "function",
            "function": {"name": "read", "arguments": '{"filePath":"/test.py"}'},
        }
        r = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": None, "tool_calls": [action]},
                ],
                [tool],
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        self.assertEqual(
            json.loads(
                r["chat"]["messages"][1]["tool_calls"][0]["function"]["arguments"]
            ),
            {"file_path": "/test.py"},
        )
        self.assertEqual(
            r["source_message_extras"][1]["fields"]["tool_calls"][0], action
        )

    def test_tool_schema_recovered_from_source_prompt(self):
        system = (
            "---- BEGIN FUNCTION #1: run ----\nDescription: Run.\nParameters:\n"
            "  (1) timeout (number, required): Seconds.\n---- END FUNCTION #1 ----"
        )
        r = convert(
            sample(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "<function=run><parameter=timeout>1.5</parameter></function>",
                    },
                ],
                repo="Lego-X/SWE-Lego-Real-Data",
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        self.assertEqual(r["tool_provenance"][0]["origin"], "source_prompt")
        self.assertEqual(
            json.loads(
                r["chat"]["messages"][-1]["tool_calls"][0]["function"]["arguments"]
            ),
            {"timeout": 1.5},
        )

    def test_serialized_source_tool_types(self):
        tool = {
            "type": "function",
            "function": {
                "name": "grep",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer"},
                        "ignore_case": {"type": "boolean"},
                        "pattern": {"type": "string"},
                    },
                },
            },
        }
        action = {
            "id": "a",
            "function": {
                "name": "grep",
                "arguments": '{"limit":"20","ignore_case":"True","pattern":"001"}',
            },
        }
        r = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": None, "tool_calls": [action]},
                ],
                [tool],
            )
        )
        self.assertEqual(
            json.loads(
                r["chat"]["messages"][-1]["tool_calls"][0]["function"]["arguments"]
            ),
            {"limit": 20, "ignore_case": True, "pattern": "001"},
        )

    def test_terminal_alternative_tool_blocks(self):
        content = (
            '<tool_call>{"name":"bash_command","arguments":{"keystrokes":"pwd\\n","duration":0.1}}</tool_call>'
            '<tool_call>{"name":"bash_command","arguments":{"keystrokes":"ls\\n","duration":0.5}}</tool_call>'
        )
        r = convert(
            sample(
                [
                    {
                        "role": "user",
                        "content": "Format your response as JSON with keystrokes\nTask Description: Q",
                    },
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "New Terminal Output:\nresult"},
                    {
                        "role": "assistant",
                        "content": '{"commands":[],"task_complete":true}',
                    },
                ]
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        call = next(m for m in r["chat"]["messages"] if m.get("tool_calls"))[
            "tool_calls"
        ][0]
        self.assertEqual(
            json.loads(call["function"]["arguments"])["commands"],
            [
                {"keystrokes": "pwd\n", "duration": 0.1},
                {"keystrokes": "ls\n", "duration": 0.5},
            ],
        )

    def test_terminal_completion_marker_keeps_summary(self):
        r = convert(
            sample(
                [
                    {
                        "role": "user",
                        "content": "Format your response as JSON with keystrokes\nTask Description: Q",
                    },
                    {
                        "role": "assistant",
                        "content": 'The fix is complete.\n<tool_call>{"name":"mark_task_complete","arguments":{}}</tool_call>',
                    },
                ]
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        self.assertIn("The fix is complete.", r["chat"]["messages"][-1]["content"])
        self.assertIn('"task_complete": true', r["chat"]["messages"][-1]["content"])

    def test_source_artifact_not_conversation(self):
        s = sample(None, raw=definition())
        s["stored_sample"]["source_file"] = "tools.json"
        r = convert(s)
        self.assertEqual(r["status"], "source_artifact")
        self.assertEqual(r["record_kind"], "tool_definition")

    def test_registry_must_match_source_revision(self):
        s = sample(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": None, "tool_calls": [call()]},
            ]
        )
        s["stored_sample"]["source_revision"] = "one"
        registry = {
            "test/plain": [
                {
                    "tool": definition(),
                    "source_revision": "two",
                    "source_file": "tools.json",
                    "record_id": "tool-row",
                    "source_shard": "fixture",
                }
            ]
        }
        with patch("openai_chat.source_tool_registry", return_value=registry):
            r = convert(s)
            self.assertEqual(
                r["tool_provenance"][0]["origin"], "observed_signature_only"
            )
            registry["test/plain"][0]["source_revision"] = "one"
            r = convert(s)
            self.assertEqual(r["tool_provenance"][0]["origin"], "source_artifact")

    def test_rejected_terminal_text_is_not_executed(self):
        r = convert(
            sample(
                [
                    {
                        "role": "user",
                        "content": "Format your response as JSON with keystrokes\nTask Description: Q",
                    },
                    {"role": "assistant", "content": "{bad json"},
                    {
                        "role": "user",
                        "content": "Previous response had parsing errors:\nERROR",
                    },
                    {
                        "role": "assistant",
                        "content": '{"commands":[],"task_complete":true}',
                    },
                ]
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        self.assertFalse(any(m.get("tool_calls") for m in r["chat"]["messages"]))
        self.assertTrue(any("rejected" in w for w in r["warnings"]))

    def test_trailing_user_request_not_given_invented_answer(self):
        r = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "Answer"},
                    {"role": "user", "content": "Another question"},
                ]
            )
        )
        self.assertEqual(r["status"], "passed_sample")
        self.assertEqual(r["chat"]["messages"][-1]["content"], "Answer")
        self.assertEqual(
            r["trailing_context"], [{"role": "user", "content": "Another question"}]
        )

    def test_validator_rejects_bad_fields_links_and_arguments(self):
        valid = {
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": None, "tool_calls": [call()]},
            ],
            "tools": [definition()],
        }
        self.assertEqual(validate_chat(valid), [])
        changed = copy.deepcopy(valid)
        changed["messages"][1]["reasoning_content"] = "private sidecar"
        self.assertTrue(validate_chat(changed))
        changed = copy.deepcopy(valid)
        changed["messages"][1]["tool_calls"][0]["function"]["arguments"] = "{} trailing"
        self.assertTrue(validate_chat(changed))
        changed = copy.deepcopy(valid)
        changed["messages"].append({"role": "assistant", "content": "Oops"})
        self.assertTrue(validate_chat(changed))
        changed = copy.deepcopy(valid)
        changed["messages"][1]["tool_calls"].append(call())
        self.assertTrue(validate_chat(changed))
        changed = copy.deepcopy(valid)
        changed["messages"][1]["role"] = "alien"
        self.assertTrue(validate_chat(changed))

    def test_published_execution_uses_last_cell_and_preserves_response(self):
        text = "```python\na = 1\n```\n```python\na = 1\nprint(a)\n```"
        snap = sample(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Execution result: 1"},
                {"role": "assistant", "content": "Done"},
            ],
            repo="m-a-p/Code-Feedback",
        )
        snap["stored_sample"]["source_revision"] = "known"
        registry = {
            "m-a-p/Code-Feedback": {
                "dataset_revision": "known",
                "selection_rule": "last_python_fence",
            }
        }
        with patch("openai_chat.execution_protocol_registry", return_value=registry):
            result = convert(snap)
        action = next(m for m in result["chat"]["messages"] if m.get("tool_calls"))
        self.assertEqual(result["status"], "passed_sample")
        self.assertEqual(
            json.loads(action["tool_calls"][0]["function"]["arguments"])["code"],
            "a = 1\nprint(a)",
        )
        self.assertEqual(action["content"], text)
        snap["stored_sample"]["source_revision"] = "different"
        with patch("openai_chat.execution_protocol_registry", return_value=registry):
            self.assertEqual(convert(snap)["status"], "needs_adapter")

    def test_producer_declaration_requires_revision_and_prompt(self):
        registry = {
            "test/plain": {
                "dataset_revision": "known",
                "system_prefix": "Official system",
                "tools": [definition()],
                "provenance": {"source_url": "fixture"},
            }
        }
        snap = sample(
            [
                {"role": "system", "content": "Official system text"},
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": None, "tool_calls": [call()]},
            ]
        )
        snap["stored_sample"]["source_revision"] = "known"
        with patch("openai_chat.producer_harness_registry", return_value=registry):
            self.assertEqual(
                convert(snap)["tool_provenance"][0]["origin"], "producer_harness"
            )
            snap["stored_sample"]["source_revision"] = "other"
            self.assertEqual(
                convert(snap)["tool_provenance"][0]["origin"], "observed_signature_only"
            )
            snap["stored_sample"]["source_revision"] = "known"
            snap["decoded_messages"][0]["content"] = "Unrelated system"
            self.assertEqual(
                convert(snap)["tool_provenance"][0]["origin"], "observed_signature_only"
            )

    def test_nested_terminal_requires_exact_unique_echoes(self):
        from openai_source_repairs import recover_observed_nested_terminal

        payload = {"commands": [{"keystrokes": "pwd\n", "duration": 0.1}]}
        malformed = json.dumps({"commands": [{"keystrokes": json.dumps(payload)}]})
        # The escaped nested object is not itself a JSON object in this representation.
        self.assertIsNone(
            recover_observed_nested_terminal(
                malformed, "New Terminal Output:\nu@h# pwd"
            )
        )
        text = "prefix " + json.dumps(payload)
        self.assertEqual(
            recover_observed_nested_terminal(text, "New Terminal Output:\nu@h# pwd"),
            payload,
        )
        self.assertIsNone(
            recover_observed_nested_terminal(text, "New Terminal Output:\nu@h# ls")
        )
        self.assertIsNone(
            recover_observed_nested_terminal(
                text, "New Terminal Output:\nu@h# pwd\nu@h# ls"
            )
        )

    def test_auxiliary_filter_preserves_unlisted_real_variants(self):
        from source_layout import is_training_source_file

        card = {"configs": [{"data_files": [{"path": "data/tool-*.parquet"}]}]}
        self.assertTrue(is_training_source_file("data/train-00001.parquet", card))
        self.assertFalse(is_training_source_file("tools.json", card))
        self.assertFalse(is_training_source_file("manifest.json", card))
        explicit = {"configs": [{"data_files": [{"path": "tools.json"}]}]}
        self.assertTrue(is_training_source_file("tools.json", explicit))

    def test_independent_validator_checks_children_of_batch(self):
        from openai_validation import check_result

        tool = definition()
        tool["function"]["parameters"] = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        result = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call("a", {"command": "pwd"}), call("b")],
                    },
                    {"role": "tool", "content": "one"},
                    {"role": "tool", "content": "two"},
                    {"role": "assistant", "content": "Done"},
                ],
                [tool],
                repo="nvidia/Nemotron-SFT-SWE-v3",
            )
        )
        self.assertEqual(result["status"], "passed_sample")
        self.assertTrue(
            any("'command' is a required property" in e for e in check_result(result))
        )

    def test_explicit_swenext_rejection_preserves_failed_text(self):
        text = "Explanation\n<function=>\n</function>"
        feedback = "NO TOOL CALL DETECTED.\nReply with EXACTLY ONE tool call."
        result = convert(
            sample(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": text},
                    {"role": "tool", "content": feedback},
                    {"role": "assistant", "content": "Done"},
                ],
                repo="TIGER-Lab/SWE-Next-SFT-Trajectories",
            )
        )
        self.assertEqual(result["status"], "passed_sample")
        self.assertIn(
            {"role": "assistant", "content": text}, result["chat"]["messages"]
        )
        self.assertIn({"role": "user", "content": feedback}, result["chat"]["messages"])
        self.assertEqual(
            result["source_repairs"][0]["execution"], "none_detected_by_source_harness"
        )
        self.assertFalse(result["chat"]["tools"])
        snap = sample(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": text},
                {
                    "role": "tool",
                    "content": "\nYou forgot to use a function call in your response. ",
                },
                {"role": "assistant", "content": "Done"},
            ],
            repo="TIGER-Lab/SWE-Next-SFT-Trajectories",
        )
        self.assertEqual(convert(snap)["status"], "passed_sample")

    def test_xml_parameter_keys_match_producer_and_duplicates_reject(self):
        from openai_chat import parse_xml_calls

        text = '<function=search><parameter=search_term>x</parameter><parameter=.json>False</parameter><parameter name="path"> ./ </parameter></function>'
        _, calls = parse_xml_calls(text, "fixture")
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args, {"search_term": "x", ".json": "False", "path": " ./ "})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_xml_calls(
                "<function=search><parameter=path>a</parameter><parameter=path>b</parameter></function>",
                "fixture",
            )


if __name__ == "__main__":
    unittest.main()
