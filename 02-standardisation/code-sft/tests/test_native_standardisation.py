import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

from apertus_common import Conversation, checkers_for, render_str
from datasets import Dataset, DatasetDict
import pyarrow.parquet as pq
import pytest

from standardise_native import checked_row, standardise

ENTRY = dict(
    repo="test/plain",
    revision="abc",
    category="code_sft",
    license="unknown",
    license_status="unknown",
    license_notes="",
    synthetic_status="unknown",
    synthetic_notes="",
    metadata_source="fixture",
)
SPEC = dict(path="fixture", source_config="local", source_split="train")
TURN = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "Answer"}]


def native(row, **kwargs):
    return checked_row(row, ENTRY, SPEC, 0, {}, "think-first", **kwargs)


def read_output(root):
    return [
        r
        for path in root.rglob("part-*.parquet")
        for r in pq.ParquetFile(path).read().to_pylist()
    ]


def test_native_reasoning_and_wire_roundtrip():
    source = {
        "conversations": [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "<think>Reason</think>Answer"},
        ]
    }
    row = native(source)
    conversation = Conversation.from_json(row["conversation_json"])
    value = json.loads(row["conversation_json"])
    assert [x["type"] for x in value["items"]] == ["user", "think", "reply", "wait"]
    assert value["items"][1]["payload"] == "Reason"
    assert value["items"][2]["payload"] == "Answer"
    assert render_str(conversation) == row["native_text"]
    assert "<|out|>think<|hdr|>Reason<|/out|>" in row["native_text"]
    assert conversation.check(checkers=checkers_for("full", training=True)).ok
    assert json.loads(row["source_record_json"]) == source


def test_top_level_system_instruction_is_preserved_and_conflicts_fail():
    row = native(
        {"question": "Q", "answer": "A", "system_prompt": {"content": "Be precise"}}
    )
    assert json.loads(row["conversation_json"])["system"]["behavior"] == "Be precise"
    with pytest.raises(ValueError, match="Conflicting"):
        native(
            {
                "messages": [{"role": "system", "content": "One"}, *TURN],
                "system_prompt": "Two",
            }
        )


def test_checked_tool_schema_and_native_call_result():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "parameters": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                },
            },
        }
    ]
    calls = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "add", "arguments": '{"n":1}'},
        }
    ]
    source = {
        "messages": [
            TURN[0],
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": "c1", "content": "2"},
            TURN[1],
        ],
        "tools": tools,
    }
    row = native(source)
    value = json.loads(row["conversation_json"])
    types = [x["type"] for x in value["items"]]
    assert "call" in types and "result" in types and types[-1] == "wait"
    calls[0]["function"]["arguments"] = '{"n":"wrong"}'
    with pytest.raises(ValueError, match="integer"):
        native(source)


def test_unclosed_reasoning_and_unpaired_internal_call_rejected():
    with pytest.raises(ValueError, match="reasoning"):
        native(
            {
                "messages": [
                    TURN[0],
                    {"role": "assistant", "content": "<think>unfinished"},
                ]
            }
        )
    with pytest.raises(ValueError):
        native(
            {
                "messages": [
                    TURN[0],
                    {"role": "tool", "tool_call_id": "missing", "content": "result"},
                    TURN[1],
                ]
            }
        )


def test_mid_dialogue_instructions_rejected_instead_of_moved():
    with pytest.raises(ValueError, match="Mid-dialogue"):
        native(
            {
                "messages": [
                    TURN[0],
                    TURN[1],
                    {"role": "system", "content": "New policy"},
                    *TURN,
                ]
            }
        )


def test_encoding_keeps_reference_tokens_and_loss_weights():
    encoding = Mock()
    encoding.encode_conversation.return_value = SimpleNamespace(
        token_ids=[1, 2, 3], loss_weights=[0.0, 0.0, 1.0]
    )
    row = native({"messages": TURN}, encoding=encoding)
    assert row["tokens"] == [1, 2, 3] and row["loss_weights"] == [0.0, 0.0, 1.0]
    encoding.encode_conversation.assert_called_once()
    assert encoding.encode_conversation.call_args.kwargs == {
        "return_loss_weights": True
    }


def test_existing_downloader_output_and_quarantine_resume(tmp_path):
    source = tmp_path / "downloaded"
    Dataset.from_list(
        [
            {"messages": TURN},
            {
                "messages": [
                    TURN[0],
                    {"role": "assistant", "content": "<think>unfinished"},
                ]
            },
        ]
    ).save_to_disk(str(source))
    metadata = dict(
        dataset_name="test/plain",
        commit_hash="source-commit",
        split="train",
        subset="default",
    )
    (source / "dataset_metadata.json").write_text(json.dumps(metadata))
    output = tmp_path / "native"
    result = standardise([source], output)
    assert result["counts"] == {"read": 2, "written": 1, "quarantined": 1}
    row = read_output(output)[0]
    assert row["source_revision"] == "source-commit"
    assert row["source_split"] == "train"
    rejected = next(output.rglob("rejected.jsonl"))
    assert (
        json.loads(rejected.read_text())["source_record"]["messages"][-1]["content"]
        == "<think>unfinished"
    )
    assert standardise([source], output) == result
    shard = next(output.rglob("part-*.parquet"))
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        standardise([source], output)


def test_datasetdict_requires_explicit_split_and_only_reads_selected(tmp_path):
    source = tmp_path / "dict"
    DatasetDict(
        train=Dataset.from_list([{"messages": TURN}]),
        test=Dataset.from_list([{"messages": TURN}]),
    ).save_to_disk(str(source))
    with pytest.raises(ValueError, match="multiple splits"):
        standardise([source], tmp_path / "ambiguous", repo="test/plain")
    result = standardise(
        [source], tmp_path / "selected", repo="test/plain", split="train"
    )
    assert result["counts"]["read"] == 1
    assert read_output(tmp_path / "selected")[0]["source_split"] == "train"


def test_resume_refuses_changed_source_mapping_and_unsafe_outputs(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"question": "Q", "answer": "A"}) + "\n")
    output = tmp_path / "native"
    standardise([source], output, repo="test/plain")
    with pytest.raises(ValueError, match="changed"):
        standardise([source], output, repo="test/plain", mapping={"question": "prompt"})
    with pytest.raises(ValueError, match="separate"):
        standardise([source], tmp_path, repo="test/plain")
    with pytest.raises(ValueError, match="Duplicate"):
        standardise([source, source], tmp_path / "other", repo="test/plain")
