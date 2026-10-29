from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ANNOTATIONS = Path(__file__).resolve().parents[1]
ROOT = ANNOTATIONS.parent
sys.path.insert(0, str(ANNOTATIONS))

from mixture_analysis.deterministic import RecordFormatError, annotate_record, extract_command_names
from mixture_analysis.judge import (
    HuggingFaceJudgeSettings,
    HuggingFaceSemanticJudge,
    JudgeSettings,
    SemanticJudge,
)
from mixture_analysis.pipeline import run
from mixture_analysis.report import Summary
from mixture_analysis.taxonomy import load_taxonomy


TAXONOMY_PATH = ANNOTATIONS / "taxonomies" / "swe_terminal_tool_use_v0.1.0.yaml"
COMPACT_TAXONOMY_PATH = ANNOTATIONS / "taxonomies" / "swe_terminal_tool_use_v0.2.0.yaml"


class TaxonomyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = load_taxonomy(TAXONOMY_PATH)

    def test_expected_dimensions_and_labels_are_registered(self):
        self.assertEqual(self.taxonomy.version, "0.1.0")
        self.assertIn("bug_fixing", self.taxonomy.labels["task_domains"])
        self.assertIn("shell", self.taxonomy.labels["tool_use.tool_types"])
        self.assertIn("python", self.taxonomy.labels["programming_languages"])

    def test_compact_taxonomy_has_only_the_five_requested_axes(self):
        compact = load_taxonomy(COMPACT_TAXONOMY_PATH)
        self.assertEqual(compact.version, "0.2.0")
        self.assertEqual(
            set(compact.labels),
            {
                "expert_domains",
                "programming_languages",
                "swe_categories",
                "terminal_categories",
                "tool_use_categories",
            },
        )
        for dimension in ("swe_categories", "terminal_categories", "tool_use_categories"):
            self.assertEqual(len(compact.labels[dimension]), 5)
        self.assertNotIn("programming_languages", compact.semantic_judge_dimensions)


class DeterministicClassifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = load_taxonomy(TAXONOMY_PATH)

    def annotate(self, row):
        return annotate_record(row, self.taxonomy)

    def labels(self, annotation, dimension):
        return annotation["classifications"][dimension]["labels"]

    def test_text_to_sql_does_not_become_terminal_use(self):
        annotation = self.annotate(
            {
                "id": "sql-1",
                "source_dataset": "example/sql",
                "source_revision": "abc",
                "messages_json": json.dumps(
                    [
                        {
                            "role": "user",
                            "content": "Question\n\nCREATE TABLE people (name VARCHAR)",
                        },
                        {"role": "assistant", "content": "SELECT name FROM people"},
                    ]
                ),
                "prompt": "Question\n\nCREATE TABLE people (name VARCHAR)",
                "response": "SELECT name FROM people",
            }
        )
        self.assertEqual(self.labels(annotation, "expert_domains"), ["none_of_the_above"])
        self.assertEqual(self.labels(annotation, "task_domains"), ["text_to_sql"])
        self.assertEqual(self.labels(annotation, "programming_languages"), ["sql"])
        self.assertEqual(self.labels(annotation, "terminal.command_names"), [])
        self.assertEqual(self.labels(annotation, "interaction_styles"), ["direct_response"])

    def test_structured_tools_and_shell_commands_are_separated(self):
        annotation = self.annotate(
            {
                "id": "tool-1",
                "source_dataset": "example/tools",
                "source_revision": "abc",
                "tools_json": json.dumps(
                    [
                        {"name": "bash", "description": "Run shell commands"},
                        {"name": "read_file", "description": "Read a file"},
                    ]
                ),
                "messages_json": json.dumps(
                    [
                        {"role": "user", "content": "Check the repository tests."},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "bash",
                                        "arguments": json.dumps({"command": "git status && pytest -q"}),
                                    },
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "c1", "content": "2 passed"},
                        {"role": "assistant", "content": "The tests pass."},
                    ]
                ),
                "metadata_json": json.dumps({"repo": "org/project", "file_path": "tests/test_x.py"}),
            }
        )
        self.assertEqual(self.labels(annotation, "expert_domains"), ["swe", "terminal", "tool_use"])
        self.assertEqual(self.labels(annotation, "tool_use.available_tool_names"), ["bash", "read_file"])
        self.assertEqual(self.labels(annotation, "tool_use.invoked_tool_names"), ["bash"])
        self.assertEqual(self.labels(annotation, "terminal.command_names"), ["git", "pytest"])
        self.assertEqual(
            self.labels(annotation, "terminal.activities"), ["test_execution", "version_control"]
        )
        self.assertIn("unused_available_tools", self.labels(annotation, "tool_use.trajectory_properties"))

    def test_explicit_swe_metadata_is_used_without_reading_intent_from_prose(self):
        annotation = self.annotate(
            {
                "id": "swe-1",
                "source_dataset": "example/swe",
                "source_revision": "abc",
                "messages_json": json.dumps(
                    [
                        {"role": "user", "content": "Please handle this issue."},
                        {"role": "assistant", "content": "Implemented."},
                    ]
                ),
                "metadata_json": json.dumps(
                    {
                        "task_type": "bug_fix",
                        "repo": "org/project",
                        "file_path": "src/parser.py",
                    }
                ),
            }
        )
        self.assertEqual(self.labels(annotation, "expert_domains"), ["swe"])
        self.assertEqual(self.labels(annotation, "task_domains"), ["bug_fixing"])
        self.assertEqual(self.labels(annotation, "programming_languages"), ["python"])
        self.assertEqual(self.labels(annotation, "swe.artifact_types"), ["source_code"])

    def test_malformed_tool_function_and_numeric_metadata_are_normalized(self):
        annotation = self.annotate(
            {
                "id": "edge-1",
                "source_dataset": "example/tools",
                "source_row_index": "17",
                "source_tokens": "42",
                "tools": [{"name": "custom", "function": "not-a-mapping"}],
                "messages": [{"role": "user", "content": "Use the custom tool."}],
            }
        )
        self.assertEqual(annotation["source_row_index"], 17)
        self.assertEqual(annotation["source_tokens"], 42)
        self.assertEqual(self.labels(annotation, "tool_use.tool_types"), ["other_tool"])

    def test_shell_parser_respects_quotes_and_heredoc_bodies(self):
        command = """python3 -c "import os; print(os.getcwd())" && cat > x.py <<'PY'\nimport fake\nprint('not a command')\nPY\npytest -q"""
        self.assertEqual(extract_command_names(command), ["cat", "pytest", "python3"])

    def test_native_apertus2_items_and_system_tools_are_supported(self):
        annotation = self.annotate(
            {
                "id": "native-1",
                "source": "example/native",
                "language": "python",
                "task": "bug_fix",
                "system": {
                    "thinking": "medium",
                    "tools": [{"name": "bash", "description": "Run shell commands"}],
                },
                "items": [
                    {"type": "user", "payload": "Fix the repository issue."},
                    {"type": "think", "payload": "I should inspect the tests."},
                    {
                        "type": "call",
                        "name": "bash",
                        "counter": 0,
                        "payload": json.dumps({"command": "git status && pytest -q"}),
                    },
                    {"type": "wait"},
                    {"type": "result", "name": "bash", "counter": 0, "payload": "2 passed"},
                    {"type": "reply", "payload": "Fixed."},
                ],
            }
        )
        self.assertEqual(annotation["source_dataset"], "example/native")
        self.assertEqual(self.labels(annotation, "expert_domains"), ["swe", "terminal", "tool_use"])
        self.assertEqual(self.labels(annotation, "task_domains"), ["bug_fixing", "terminal_task_execution"])
        self.assertEqual(self.labels(annotation, "terminal.command_names"), ["git", "pytest"])
        self.assertEqual(self.labels(annotation, "tool_use.invoked_tool_names"), ["bash"])
        self.assertIn("reasoning_trace", self.labels(annotation, "interaction_styles"))

    def test_non_record_manifest_is_rejected(self):
        with self.assertRaisesRegex(RecordFormatError, "no supported content"):
            self.annotate({"dataset-a": {"status": "available"}})

    def test_absence_of_expert_evidence_remains_unknown(self):
        annotation = self.annotate(
            {"id": "unknown-1", "prompt": "Please help with this.", "response": "Certainly."}
        )
        expert = annotation["classifications"]["expert_domains"]
        self.assertEqual(expert["status"], "unknown")
        self.assertEqual(expert["labels"], [])


class PipelineTest(unittest.TestCase):
    def test_jsonl_pipeline_writes_sidecars_summary_and_completion_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.jsonl"
            rows = [
                {
                    "id": "one",
                    "source_dataset": "example/sql",
                    "source_revision": "r1",
                    "prompt": "CREATE TABLE x (a INT)",
                    "response": "SELECT a FROM x",
                    "messages_json": json.dumps(
                        [
                            {"role": "user", "content": "CREATE TABLE x (a INT)"},
                            {"role": "assistant", "content": "SELECT a FROM x"},
                        ]
                    ),
                },
                {
                    "id": "two",
                    "source_dataset": "example/python",
                    "source_revision": "r1",
                    "metadata_json": json.dumps({"language": "python"}),
                    "messages_json": json.dumps(
                        [
                            {"role": "user", "content": "Explain this."},
                            {"role": "assistant", "content": "It returns one."},
                        ]
                    ),
                },
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "output"
            taxonomy = load_taxonomy(TAXONOMY_PATH)
            result = run(
                [source], output, taxonomy, output_format="jsonl", rows_per_shard=1
            )
            self.assertEqual(result["rows"], 2)
            self.assertEqual(len(list(output.glob("part-*.jsonl"))), 2)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["records"], 2)
            self.assertEqual(summary["source_datasets"], {"example/python": 1, "example/sql": 1})
            self.assertTrue((output / "run.json").exists())
            run_identity = json.loads((output / "run.json").read_text())
            self.assertIn("created_at_utc", run_identity)
            self.assertIn("code_commit", run_identity)
            self.assertRegex(run_identity["classifier_code_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(run_identity["runtime_metadata"], {})
            self.assertTrue((output / "_SUCCESS.json").exists())
            self.assertEqual(run([source], output, taxonomy, output_format="jsonl", rows_per_shard=1), result)

    def test_runtime_metadata_is_recorded_and_part_of_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.jsonl"
            source.write_text(
                json.dumps({"id": "one", "prompt": "Question", "response": "Answer"})
                + "\n",
                encoding="utf-8",
            )
            output = root / "output"
            taxonomy = load_taxonomy(TAXONOMY_PATH)
            metadata = {
                "cluster": "clariden",
                "uenv_image_id": "c05f143d5fbf0927",
            }
            run(
                [source],
                output,
                taxonomy,
                output_format="jsonl",
                runtime_metadata=metadata,
            )
            identity = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(identity["runtime_metadata"], metadata)
            with self.assertRaisesRegex(ValueError, "different inputs or options"):
                run(
                    [source],
                    output,
                    taxonomy,
                    output_format="jsonl",
                    runtime_metadata={**metadata, "uenv_image_id": "different"},
                )

    def test_limit_is_part_of_immutable_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.jsonl"
            source.write_text(
                json.dumps({"id": "one", "prompt": "Question", "response": "Answer"}) + "\n",
                encoding="utf-8",
            )
            output = root / "output"
            taxonomy = load_taxonomy(TAXONOMY_PATH)
            run([source], output, taxonomy, output_format="jsonl", limit=1)
            with self.assertRaisesRegex(ValueError, "different inputs or options"):
                run([source], output, taxonomy, output_format="jsonl")

    def test_input_shards_are_disjoint_and_cover_the_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.jsonl"
            rows = [
                {"id": f"row-{index}", "prompt": "Question", "response": "Answer"}
                for index in range(6)
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            taxonomy = load_taxonomy(TAXONOMY_PATH)
            observed = []
            for shard_index in range(2):
                output = root / f"output-{shard_index}"
                result = run(
                    [source],
                    output,
                    taxonomy,
                    output_format="jsonl",
                    num_input_shards=2,
                    input_shard_index=shard_index,
                )
                self.assertEqual(result["rows"], 3)
                records = [
                    json.loads(line)
                    for part in output.glob("part-*.jsonl")
                    for line in part.read_text(encoding="utf-8").splitlines()
                ]
                ids = {record["record_id"] for record in records}
                observed.append(ids)
                identity = json.loads((output / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(
                    identity["input_sharding"],
                    {
                        "method": "global_row_index_modulo",
                        "num_shards": 2,
                        "shard_index": shard_index,
                    },
                )
            self.assertFalse(observed[0] & observed[1])
            self.assertEqual(observed[0] | observed[1], {row["id"] for row in rows})

    def test_optional_semantic_judge_is_applied_and_versioned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "source_dataset": "example/semantic",
                        "prompt": "Fix the defect in this existing project.",
                        "response": "I updated the implementation.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            taxonomy = load_taxonomy(TAXONOMY_PATH)

            def request(_system, _user):
                return (
                    json.dumps(
                        {
                            "expert_domains": {
                                "status": "labeled",
                                "labels": ["swe"],
                                "confidence": "high",
                                "evidence": ["The request modifies an existing project."],
                            },
                            "task_domains": {
                                "status": "labeled",
                                "labels": ["bug_fixing"],
                                "confidence": "high",
                                "evidence": ["The task asks to fix a defect."],
                            },
                        }
                    ),
                    {"prompt_tokens": 100, "completion_tokens": 30},
                )

            judge = SemanticJudge(
                taxonomy,
                JudgeSettings(model="example/judge", retries=0, batch_size=1),
                request_function=request,
            )
            output = root / "output"
            run([source], output, taxonomy, output_format="jsonl", judge=judge)
            row = json.loads(next(output.glob("part-*.jsonl")).read_text())
            classifications = json.loads(row["classifications_json"])
            self.assertEqual(classifications["expert_domains"]["labels"], ["swe"])
            self.assertEqual(classifications["task_domains"]["method"], "llm_classifier")
            self.assertEqual(row["judge_prompt_tokens"], 100)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["judge_usage"]["requests"], 1)
            self.assertEqual(summary["judge_usage"]["prompt_tokens"], 100)
            identity = json.loads((output / "run.json").read_text())
            self.assertEqual(identity["judge"]["model"], "example/judge")
            self.assertNotIn("api_key", json.dumps(identity).lower())


class SemanticJudgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = load_taxonomy(TAXONOMY_PATH)

    def test_invalid_response_becomes_reportable_error(self):
        judge = SemanticJudge(
            self.taxonomy,
            JudgeSettings(model="example/judge", retries=0),
            request_function=lambda _system, _user: ("not json", {}),
        )
        record = {"id": "one", "prompt": "Help with this.", "response": "Done."}
        annotation = annotate_record(record, self.taxonomy)
        judged = judge.annotate(record, annotation)
        self.assertEqual(judged["classifications"]["expert_domains"]["status"], "error")
        self.assertEqual(judged["classifications"]["task_domains"]["status"], "error")
        self.assertEqual(judged["judge_error"], "JudgeResponseError")

    def test_resolved_dimensions_do_not_call_judge(self):
        calls = []

        def request(_system, _user):
            calls.append(1)
            raise AssertionError("judge should not be called")

        judge = SemanticJudge(
            self.taxonomy,
            JudgeSettings(model="example/judge", retries=0),
            request_function=request,
        )
        record = {
            "id": "sql",
            "prompt": "CREATE TABLE x (a INT)",
            "response": "SELECT a FROM x",
            "messages": [
                {"role": "user", "content": "CREATE TABLE x (a INT)"},
                {"role": "assistant", "content": "SELECT a FROM x"},
            ],
        }
        annotation = annotate_record(record, self.taxonomy)
        self.assertIs(judge.annotate(record, annotation), annotation)
        self.assertEqual(calls, [])


class HuggingFaceSemanticJudgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = load_taxonomy(TAXONOMY_PATH)

    def test_local_backend_batches_only_unresolved_records(self):
        batch_sizes = []

        def generate(prompts):
            batch_sizes.append(len(prompts))
            response = json.dumps(
                {
                    "expert_domains": {
                        "status": "labeled",
                        "labels": ["swe"],
                        "confidence": "medium",
                        "evidence": ["Existing project modification."],
                    },
                    "task_domains": {
                        "status": "labeled",
                        "labels": ["bug_fixing"],
                        "confidence": "medium",
                        "evidence": ["A defect is corrected."],
                    },
                }
            )
            return [
                (response, {"prompt_tokens": 200 + index, "completion_tokens": 40})
                for index, _prompt in enumerate(prompts)
            ]

        judge = HuggingFaceSemanticJudge(
            self.taxonomy,
            HuggingFaceJudgeSettings(
                model="example/local-judge",
                revision="deadbeef",
                batch_size=2,
            ),
            batch_generation_function=generate,
        )
        records = [
            {"id": "one", "prompt": "Fix this project defect.", "response": "Fixed."},
            {"id": "two", "prompt": "Repair the existing package.", "response": "Done."},
            {
                "id": "sql",
                "prompt": "CREATE TABLE x (a INT)",
                "response": "SELECT a FROM x",
                "messages": [
                    {"role": "user", "content": "CREATE TABLE x (a INT)"},
                    {"role": "assistant", "content": "SELECT a FROM x"},
                ],
            },
        ]
        annotations = [annotate_record(record, self.taxonomy) for record in records]
        judged = judge.annotate_batch(records, annotations)
        self.assertEqual(batch_sizes, [2])
        self.assertEqual(judged[0]["classifications"]["expert_domains"]["labels"], ["swe"])
        self.assertEqual(judged[0]["judge_provider"], "huggingface_local")
        self.assertEqual(judged[0]["judge_model_revision"], "deadbeef")
        self.assertNotIn("judge_provider", judged[2])
        identity = judge.run_identity()
        self.assertEqual(identity["provider"], "huggingface_local")
        self.assertEqual(identity["resolved_revision"], "deadbeef")
        self.assertFalse(identity["enable_thinking"])
        self.assertNotIn("base_url", identity)

    def test_local_backend_runs_through_sharded_pipeline(self):
        response = json.dumps(
            {
                "expert_domains": {
                    "status": "labeled",
                    "labels": ["none_of_the_above"],
                    "confidence": "medium",
                    "evidence": ["This is standalone coding."],
                },
                "task_domains": {
                    "status": "labeled",
                    "labels": ["code_generation"],
                    "confidence": "medium",
                    "evidence": ["A standalone function is requested."],
                },
            }
        )
        judge = HuggingFaceSemanticJudge(
            self.taxonomy,
            HuggingFaceJudgeSettings(
                model="example/local-judge",
                revision="deadbeef",
                batch_size=1,
            ),
            batch_generation_function=lambda prompts: [
                (response, {"prompt_tokens": 150, "completion_tokens": 35})
                for _prompt in prompts
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "source_dataset": "example/local",
                        "prompt": "Write a function that returns one.",
                        "response": "def one(): return 1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "output"
            run([source], output, self.taxonomy, output_format="jsonl", judge=judge)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["judge_usage"]["requests"], 1)
            self.assertEqual(
                summary["judge_usage"]["providers"],
                {"huggingface_local": 1},
            )
            identity = json.loads((output / "run.json").read_text())
            self.assertEqual(identity["judge"]["resolved_revision"], "deadbeef")

    def test_compact_judge_enforces_expert_category_applicability(self):
        taxonomy = load_taxonomy(COMPACT_TAXONOMY_PATH)
        response = json.dumps(
            {
                "expert_domains": {
                    "status": "labeled",
                    "labels": ["none_of_the_above"],
                    "confidence": "high",
                    "evidence": ["Standalone function generation."],
                },
                "swe_categories": {
                    "status": "not_applicable",
                    "labels": [],
                    "confidence": "high",
                    "evidence": [],
                },
                "terminal_categories": {
                    "status": "not_applicable",
                    "labels": [],
                    "confidence": "high",
                    "evidence": [],
                },
                "tool_use_categories": {
                    "status": "not_applicable",
                    "labels": [],
                    "confidence": "high",
                    "evidence": [],
                },
            }
        )
        judge = HuggingFaceSemanticJudge(
            taxonomy,
            HuggingFaceJudgeSettings(
                model="example/local-judge",
                revision="deadbeef",
                batch_size=1,
            ),
            batch_generation_function=lambda prompts: [
                (response, {"prompt_tokens": 100, "completion_tokens": 30})
                for _prompt in prompts
            ],
        )
        record = {
            "id": "standalone",
            "prompt": "Write a Python function that returns one.",
            "response": "def one(): return 1",
        }
        annotation = annotate_record(record, taxonomy)
        judged = judge.annotate(record, annotation)
        self.assertEqual(
            judged["classifications"]["expert_domains"]["labels"],
            ["none_of_the_above"],
        )
        self.assertEqual(
            judged["classifications"]["swe_categories"]["status"],
            "not_applicable",
        )
        self.assertEqual(
            judged["classifications"]["programming_languages"]["method"],
            "deterministic_rule",
        )


class SummaryTest(unittest.TestCase):
    def test_partial_token_coverage_does_not_publish_misleading_shares(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        first = annotate_record(
            {
                "id": "one",
                "source_dataset": "a",
                "source_tokens": 10,
                "metadata_json": json.dumps({"language": "python"}),
                "prompt": "Question",
                "response": "Answer",
            },
            taxonomy,
        )
        second = annotate_record(
            {
                "id": "two",
                "source_dataset": "b",
                "metadata_json": json.dumps({"language": "python"}),
                "prompt": "Question",
                "response": "Answer",
            },
            taxonomy,
        )
        summary = Summary()
        summary.add(first)
        summary.add(second)
        report = summary.as_dict()
        self.assertIsNone(report["source_tokens"])
        self.assertIsNone(report["dimensions"]["programming_languages"]["python"]["source_tokens"])
        self.assertIsNone(report["source_dataset_statistics"]["a"]["source_token_share"])


if __name__ == "__main__":
    unittest.main()
