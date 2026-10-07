import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from apertus_common import Claim, Conversation, Reply, SystemPrompt, User, Wait, load_encoding
from datasets import Dataset, DatasetDict, load_from_disk

from apertus2_processing._input import discover, iter_rows, make_plan
from apertus2_processing._output import iter_token_rows
from apertus2_processing._run import load_run, verified_manifests
from apertus2_processing._util import read_json
from apertus2_processing.cli import main


def args(operation, source, output, *extra):
    return [operation, str(source), "--output", str(output), *map(str, extra)]


def test_check_all_rows_and_issue_counts(tmp_path, conversation):
    invalid = Conversation(items=[Reply(payload="missing system")])
    warning = Conversation(
        system=SystemPrompt.build(),
        items=[User(payload="Hi"), Reply(payload="x"), Claim(payload="x"), Wait()],
    )
    path = tmp_path / "input.jsonl"
    path.write_text(
        conversation.to_json() + "\n{bad\n" + invalid.to_json() + "\n" + warning.to_json() + "\n"
    )
    out = tmp_path / "check"
    assert main(args("check", path, out, "--num-shards", 3, "--batch-rows", 1)) == 1
    summary = read_json(out / "summary.json")
    assert summary["complete"] and not summary["policy_pass"]
    assert summary["rows"] == {
        "processed": 4,
        "accepted": 2,
        "rejected": 2,
        "rows_with_error": 2,
        "rows_with_warning": 1,
    }
    issues = [
        json.loads(line)
        for shard in sorted((out / "shards").iterdir())
        for line in (shard / "issues.jsonl").read_text().splitlines()
    ]
    assert {issue["row"] for issue in issues} == {1, 2, 3}
    assert all(v["occurrences"] >= v["affected_records"] > 0 for v in summary["issues"])
    assert not (out / "dataset").exists()


@pytest.mark.parametrize("kind", ["hf", "parquet"])
def test_tokenize_preserves_exact_ids_and_split(tmp_path, artifact, conversation, kind):
    source = tmp_path / "input-hf"
    DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "conversation_json": [conversation.to_json()] * 3,
                    "conversation_id": ["a", "b", "c"],
                }
            ),
            "validation": Dataset.from_dict(
                {"conversation_json": [conversation.to_json()], "conversation_id": ["d"]}
            ),
        }
    ).save_to_disk(source)
    out = tmp_path / "tokens"
    assert (
        main(
            args(
                "tokenize",
                source,
                out,
                "--tokenizer",
                artifact,
                "--format",
                kind,
                "--check",
                "full",
                "--num-shards",
                3,
                "--batch-bytes",
                1,
            )
        )
        == 0
    )
    expected = load_encoding(artifact).encode_conversation(conversation)
    config = load_run(out)
    rows = list(iter_token_rows(out, verified_manifests(out, config)))
    assert [v["record_id"] for v in rows] == ["a", "b", "c", "d"]
    assert all(v["token_ids"] == expected for v in rows)
    assert {v["split"] for v in rows} == {"train", "validation"}
    if kind == "hf":
        data = load_from_disk(out / "dataset")
        assert len(data["train"]) == 3 and len(data["validation"]) == 1
    else:
        splits = read_json(out / "dataset" / "splits.json")
        assert set(splits) == {"train", "validation"}
        assert (
            sum(
                pq.read_table(out / "dataset" / p).num_rows
                for files in splits.values()
                for p in files
            )
            == 4
        )


def test_resume_semantic_identity_and_checksums(tmp_path, corpus, artifact):
    out = tmp_path / "tokens"
    command = args("tokenize", corpus, out, "--tokenizer", artifact, "--num-shards", 3)
    assert main(command) == 0
    assert main([*command, "--resume", "--workers", "2", "--batch-rows", "2"]) == 0
    assert main(command) == 2
    assert main([*command, "--resume", "--check", "full"]) == 2
    token = next((out / "shards").rglob("*.arrow"))
    with token.open("ab") as target:
        target.write(b"corruption")
    assert main([*command, "--resume"]) == 2
    assert main(["merge", str(out)]) == 2


def test_external_shards_and_exact_coverage(tmp_path, corpus):
    out = tmp_path / "check"
    command = args("check", corpus, out, "--num-shards", 3)
    assert main([*command, "--shard-index", "2"]) == 0
    assert main(["merge", str(out)]) == 2
    assert main([*command, "--shard-index", "0"]) == 0
    assert main([*command, "--shard-index", "1"]) == 0
    assert main(["merge", str(out)]) == 0
    assert read_json(out / "summary.json")["rows"]["processed"] == 7


def test_jsonl_offsets_cover_unicode_crlf_and_missing_newline(tmp_path, conversation):
    source = tmp_path / "input"
    source.mkdir()
    payload = conversation.to_json().encode()
    (source / "a.jsonl").write_bytes(payload + b"\r\n" + payload)
    (source / "b.jsonl").write_bytes(payload + b"\n")
    sources = discover(source)
    plan = make_plan(sources, 7)
    records = [row for shard in plan for row in iter_rows(sources, shard, None, None)]
    assert [(r[0]["source"], r[0]["row"]) for r in records] == [
        (str(source / "a.jsonl"), 0),
        (str(source / "a.jsonl"), 1),
        (str(source / "b.jsonl"), 0),
    ]
    assert all(Conversation.from_json(r[1]) == conversation for r in records)


@pytest.mark.parametrize("policy,status", [("fail", 1), ("skip", 0)])
def test_tokenization_error_policy(tmp_path, conversation, artifact, policy, status):
    source = tmp_path / "input.jsonl"
    source.write_text("broken\n" + conversation.to_json() + "\n")
    out = tmp_path / "tokens"
    assert (
        main(args("tokenize", source, out, "--tokenizer", artifact, "--on-error", policy)) == status
    )
    summary = read_json(out / "summary.json")
    assert summary["rows"]["rejected"] == 1 and summary["rows"]["accepted"] == 1
    assert (out / "dataset").exists() == (policy == "skip")


def test_optional_check_and_weights(tmp_path, artifact):
    conversation = Conversation(
        system=SystemPrompt.build(), items=[Reply(payload="x", loss_weight=0.5), Wait()]
    )
    source = tmp_path / "input.jsonl"
    source.write_text(conversation.to_json() + "\n")
    assert main(args("check", source, tmp_path / "strict")) == 1
    assert main(args("check", source, tmp_path / "allowed", "--allow-loss-weights")) == 0
    out = tmp_path / "tokens"
    assert (
        main(
            args(
                "tokenize",
                source,
                out,
                "--tokenizer",
                artifact,
                "--check",
                "full",
                "--emit-loss-weights",
            )
        )
        == 0
    )
    row = load_from_disk(out / "dataset")["train"][0]
    expected = load_encoding(artifact).encode_conversation(conversation, return_loss_weights=True)
    assert row["token_ids"] == expected.token_ids
    assert row["loss_weights"] == expected.loss_weights
    assert len(row["loss_weights"]) == row["token_count"]


def test_context_missing_and_supplied(tmp_path):
    from apertus_common import Call, ToolSpec

    system = SystemPrompt.build(
        tools=[ToolSpec(name="echo", description="Echo", json_schema={"type": "object"})]
    )
    prefix = Conversation(system=system, items=[User(payload="Hi")])
    fragment = Conversation(items=[Call.build(name="echo", counter=0, arguments={}), Wait()])
    source = tmp_path / "wrapped.jsonl"
    source.write_text(
        json.dumps(
            {
                "conversation_json": fragment.to_json(),
                "check_context": {"prefix": json.loads(prefix.to_json())},
            }
        )
        + "\n"
    )
    assert (
        main(
            args(
                "check", source, tmp_path / "without", "--conversation-column", "conversation_json"
            )
        )
        == 1
    )
    assert (
        main(
            args(
                "check",
                source,
                tmp_path / "with",
                "--conversation-column",
                "conversation_json",
                "--context-column",
                "check_context",
            )
        )
        == 0
    )


def test_spawn_subprocess_matches_serial(tmp_path, corpus, artifact):
    out = tmp_path / "spawn"
    command = [
        sys.executable,
        "-m",
        "apertus2_processing.cli",
        *args(
            "tokenize",
            corpus,
            out,
            "--tokenizer",
            artifact,
            "--workers",
            2,
            "--num-shards",
            4,
            "--batch-rows",
            2,
        ),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert read_json(out / "summary.json")["rows"]["processed"] == 7


def test_empty_corpus_and_empty_splits(tmp_path, artifact):
    source = tmp_path / "empty.jsonl"
    source.write_text("")
    out = tmp_path / "tokens"
    assert main(args("tokenize", source, out, "--tokenizer", artifact, "--num-shards", 3)) == 0
    assert len(load_from_disk(out / "dataset")["train"]) == 0


def test_changed_source_or_artifact_invalidates_resume(tmp_path, corpus, artifact):
    out = tmp_path / "tokens"
    command = args("tokenize", corpus, out, "--tokenizer", artifact)
    assert main(command) == 0
    original = corpus.read_bytes()
    corpus.write_bytes(original + b"\n")
    assert main([*command, "--resume"]) == 2
    corpus.write_bytes(original)
    (artifact / "new.json").write_text("{}")
    assert main([*command, "--resume"]) == 2


@pytest.mark.skipif(
    not os.environ.get("APERTUS_ENCODING"), reason="set APERTUS_ENCODING for real artifact"
)
def test_real_artifact(tmp_path, conversation):
    artifact = Path(os.environ["APERTUS_ENCODING"])
    source = tmp_path / "input.jsonl"
    source.write_text(conversation.to_json() + "\n")
    out = tmp_path / "tokens"
    assert (
        main(
            args(
                "tokenize",
                source,
                out,
                "--tokenizer",
                artifact,
                "--check",
                "full",
                "--emit-loss-weights",
            )
        )
        == 0
    )
    row = load_from_disk(out / "dataset")["train"][0]
    expected = load_encoding(artifact).encode_conversation(conversation, return_loss_weights=True)
    assert row["token_ids"] == expected.token_ids and row["loss_weights"] == expected.loss_weights


def test_conflicting_context_is_data_failure(tmp_path, conversation):
    source = tmp_path / "wrapped.jsonl"
    conflicting = Conversation(system=SystemPrompt.build(behavior="different"))
    source.write_text(
        json.dumps(
            {
                "conversation_json": conversation.to_json(),
                "ctx": {"prefix": json.loads(conflicting.to_json())},
            }
        )
        + "\n"
    )
    out = tmp_path / "checked"
    assert (
        main(
            args(
                "check",
                source,
                out,
                "--conversation-column",
                "conversation_json",
                "--context-column",
                "ctx",
            )
        )
        == 1
    )
    assert read_json(out / "summary.json")["issues"][0]["rule"] == "context/invalid"


def test_unexpected_encoding_exception_aborts(tmp_path, corpus, artifact, monkeypatch):
    from apertus_common import ApertusEncoding

    def broken(*unused_args, **unused_kwargs):
        raise ValueError("implementation defect")

    monkeypatch.setattr(ApertusEncoding, "encode_conversations", broken)
    out = tmp_path / "tokens"
    assert main(args("tokenize", corpus, out, "--tokenizer", artifact, "--on-error", "skip")) == 2
    assert not list((out / "shards").glob("*/manifest.json"))
    assert not (out / "dataset").exists()


def test_invalid_surrogate_is_reported(tmp_path, monkeypatch):
    import apertus2_processing._runner as runner

    source = tmp_path / "input.jsonl"
    source.write_text("{}\n")

    def rows(*unused_args):
        yield (
            {"source": str(source), "split": "train", "row": 0, "record_id": None},
            '{"payload":"\ud800"}',
            None,
            None,
        )

    monkeypatch.setattr(runner, "iter_rows", rows)
    out = tmp_path / "checked"
    assert main(args("check", source, out)) == 1
    assert read_json(out / "summary.json")["rows"]["rows_with_error"] == 1


def test_explicit_thread_configuration(tmp_path, corpus, monkeypatch):
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    out = tmp_path / "checked"
    assert main(args("check", corpus, out)) == 0
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ["RAYON_NUM_THREADS"] == "1"
    assert main(args("check", corpus, out, "--resume", "--tokenizer-threads", "3")) == 0
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
    assert os.environ["RAYON_NUM_THREADS"] == "3"


def test_context_column_cannot_be_silently_ignored(tmp_path, corpus, conversation):
    assert main(args("check", corpus, tmp_path / "bare", "--context-column", "ctx")) == 2
    wrapped = tmp_path / "wrapped.jsonl"
    wrapped.write_text(json.dumps({"conversation_json": conversation.to_json()}) + "\n")
    out = tmp_path / "wrapped-check"
    assert (
        main(
            args(
                "check",
                wrapped,
                out,
                "--conversation-column",
                "conversation_json",
                "--context-column",
                "ctx",
            )
        )
        == 1
    )
    assert read_json(out / "summary.json")["rows"]["rejected"] == 1


def test_context_bytes_count_toward_batch_limit(tmp_path, conversation, monkeypatch):
    import apertus2_processing._runner as runner

    source = tmp_path / "wrapped.jsonl"
    context = {
        "prefix": json.loads(
            Conversation(system=conversation.system, items=[User(payload="a" * 4000)]).to_json()
        )
    }
    source.write_text(
        (json.dumps({"conversation_json": conversation.to_json(), "ctx": context}) + "\n") * 3
    )
    batches = []
    original = runner.process_batch

    def inspect(batch, *remaining):
        batches.append(len(batch))
        return original(batch, *remaining)

    monkeypatch.setattr(runner, "process_batch", inspect)
    assert (
        main(
            args(
                "check",
                source,
                tmp_path / "report",
                "--conversation-column",
                "conversation_json",
                "--context-column",
                "ctx",
                "--batch-bytes",
                "4500",
            )
        )
        == 0
    )
    assert batches == [1, 1, 1]


def test_prepared_shards_avoid_rehashing_and_detect_change(tmp_path, corpus, monkeypatch):
    import apertus2_processing._guards as guards

    out = tmp_path / "prepared"
    assert main(args("check", corpus, out, "--num-shards", "2", "--prepare-only")) == 0
    checksum = guards.checksum

    def disallow_source_hash(path):
        assert Path(path) != corpus, "prepared shard must not hash the corpus"
        return checksum(path)

    monkeypatch.setattr(guards, "checksum", disallow_source_hash)
    assert main(["run-shard", str(out), "--shard-index", "0"]) == 0
    with corpus.open("a") as target:
        target.write("\n")
    assert main(["run-shard", str(out), "--shard-index", "1"]) == 2


def test_prepared_merge_rehashes_even_if_stat_unchanged(tmp_path, corpus):
    out = tmp_path / "prepared"
    assert main(args("check", corpus, out, "--prepare-only")) == 0
    assert main(["run-shard", str(out), "--shard-index", "0"]) == 0
    stat = corpus.stat()
    content = corpus.read_bytes()
    changed = content.replace(b"Hello", b"Hallo")
    assert len(content) == len(changed) and content != changed
    corpus.write_bytes(changed)
    os.utime(corpus, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert main(["merge", str(out)]) == 2


def test_mutated_shard_summary_cannot_forge_pass(tmp_path):
    source = tmp_path / "broken.jsonl"
    source.write_text("invalid\n")
    out = tmp_path / "check"
    assert main(args("check", source, out)) == 1
    path = out / "shards" / "000000" / "manifest.json"
    manifest = read_json(path)
    manifest["summary"]["rows"].update(accepted=1, rejected=0)
    path.write_text(json.dumps(manifest))
    assert main(["merge", str(out)]) == 2


def test_invalid_record_id_is_row_failure(tmp_path, conversation):
    source = tmp_path / "wrapped.jsonl"
    source.write_text(
        json.dumps({"conversation_json": conversation.to_json(), "record_id": "\ud800"}) + "\n"
    )
    out = tmp_path / "check"
    assert main(args("check", source, out, "--conversation-column", "conversation_json")) == 1
    assert read_json(out / "summary.json")["rows"]["rejected"] == 1


def test_nonfinite_context_is_row_failure(tmp_path, conversation):
    source = tmp_path / "wrapped.jsonl"
    source.write_text(
        json.dumps({"conversation_json": conversation.to_json(), "ctx": {"history": float("nan")}})
        + "\n"
    )
    out = tmp_path / "check"
    assert (
        main(
            args(
                "check",
                source,
                out,
                "--conversation-column",
                "conversation_json",
                "--context-column",
                "ctx",
            )
        )
        == 1
    )
    assert read_json(out / "summary.json")["issues"][0]["rule"] == "context/invalid"


def test_merge_rejects_missing_token_file_declaration(tmp_path, corpus, artifact):
    out = tmp_path / "tokens"
    assert main(args("tokenize", corpus, out, "--tokenizer", artifact)) == 0
    path = out / "shards" / "000000" / "manifest.json"
    manifest = read_json(path)
    manifest["token_files"] = []
    path.write_text(json.dumps(manifest))
    assert main(["merge", str(out)]) == 2


@pytest.mark.parametrize("mutation", ["duplicate", "split", "index"])
def test_merge_rejects_corrupt_token_descriptors(tmp_path, artifact, conversation, mutation):
    source = tmp_path / "input-hf"
    DatasetDict(
        {
            name: Dataset.from_dict({"conversation_json": [conversation.to_json()]})
            for name in ("train", "validation")
        }
    ).save_to_disk(source)
    out = tmp_path / "tokens"
    assert main(args("tokenize", source, out, "--tokenizer", artifact)) == 0
    path = out / "shards" / "000000" / "manifest.json"
    manifest = read_json(path)
    if mutation == "duplicate":
        manifest["token_files"][1] = dict(manifest["token_files"][0])
    elif mutation == "split":
        manifest["token_files"][0]["split"] = "wrong"
    else:
        manifest["index"] = 1
    path.write_text(json.dumps(manifest))
    assert main(["merge", str(out)]) == 2


@pytest.mark.parametrize(
    "context",
    [
        {"history": "partial", "checkpoint": {"calls": {}}},
        {"history": "partial", "checkpoint": {"document_ids": "abc"}},
        {"host_closures": {}},
    ],
)
def test_context_collection_types_are_not_coerced(tmp_path, conversation, context):
    source = tmp_path / "wrapped.jsonl"
    source.write_text(
        json.dumps({"conversation_json": conversation.to_json(), "ctx": context}) + "\n"
    )
    out = tmp_path / "check"
    assert (
        main(
            args(
                "check",
                source,
                out,
                "--conversation-column",
                "conversation_json",
                "--context-column",
                "ctx",
            )
        )
        == 1
    )
    assert read_json(out / "summary.json")["issues"][0]["rule"] == "context/invalid"
