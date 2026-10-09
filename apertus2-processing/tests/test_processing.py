import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from apertus_common import Conversation, checkers_for, load_encoding
from datasets import Dataset, DatasetDict, load_from_disk

from apertus2_processing import _indexed, _inputs, _run
from apertus2_processing.cli import main
from apertus2_processing.report import issue_rows

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
EXAMPLE_OPTIONS = {"native.json": [], "weighted.json": ["--loss-weights"]}
APERTUS_DATA = Path(sys.executable).parent / "apertus-data"


def write_parquet(path, rows, row_group_size=None):
    table = pa.table({"conversation_json": pa.array(rows, pa.string())})
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def example(name):
    """An example conversation as the compact JSON a dataset would store."""
    return json.dumps(json.loads((EXAMPLES / name).read_text()))


def run_all(mode, source, run, *options, workers=1):
    assert main(["prepare", mode, str(source), str(run), *map(str, options)]) == 0
    assert main(["encode", str(run), "--workers", str(workers)]) == 0
    return main(["merge", str(run)])


def documents(prefix, dtype):
    lengths = _indexed.read_lengths(prefix, dtype)
    data = np.fromfile(f"{prefix}.bin", dtype=dtype)
    return (
        [part.tolist() for part in np.split(data, np.cumsum(lengths)[:-1])] if len(lengths) else []
    )


def encoded(artifact, records):
    """Direct encodings as (token ids, float32 loss weights) per record."""
    encoding = load_encoding(artifact)
    results = [
        encoding.encode_conversation(Conversation.from_json(r), return_loss_weights=True)
        for r in records
    ]
    return [r.token_ids for r in results], [np.float32(r.loss_weights).tolist() for r in results]


def write_input(kind, path, records):
    """Store records in one supported layout; return prepare options it needs."""
    if kind == "parquet":
        for directory in (path / "nested", path / ".hidden"):
            directory.mkdir(parents=True)
        write_parquet(path / "a.parquet", records[:5], row_group_size=2)
        write_parquet(path / "nested" / "b.parquet", records[5:], row_group_size=2)
        write_parquet(path / ".hidden" / "skipped.parquet", ["not json"])
        (path / "README.md").write_text("not Parquet, skipped")
        return ["--loss-weights"]  # also merge loss weights across shards
    DatasetDict(
        {
            "train": Dataset.from_dict({"native": ["unused"]}),
            "validation": Dataset.from_dict({"native": records}),
        }
    ).save_to_disk(str(path))
    return ["--split", "validation", "--column", "native"]


@pytest.mark.parametrize("kind", ["parquet", "hf"])
def test_tokenized_shards_merge_into_input_order(tmp_path, artifact, records, kind):
    options = write_input(kind, tmp_path / "input", records)
    run = tmp_path / "run"
    status = run_all(
        "tokenize", tmp_path / "input", run, "--tokenizer", artifact, "--shard-rows", 3, *options
    )
    assert status == 0
    shards = json.loads((run / "run.json").read_text())["shards"]
    assert len(shards) == len(list((run / "shards").iterdir())) == 3
    tokens, weights = encoded(artifact, records)
    assert documents(run / "tokens", np.int32) == tokens
    if kind == "parquet":  # whole row groups of 2, 2, 1 | 2 rows, never split
        assert [shard["row_groups"] for shard in shards] == [[0, 1], [2], [0]]
        assert documents(run / "loss_weights", np.float32) == weights
    else:
        assert not (run / "loss_weights.bin").exists()
    assert json.loads((run / "summary.json").read_text())["documents"] == 7


def test_check_reports_exactly_which_samples_have_which_issues(tmp_path, records):
    unfinished = json.loads(records[0])
    unfinished["items"] = unfinished["items"][:-1]  # ends with output: training warning
    source = tmp_path / "input"
    source.mkdir()
    files = {
        "a.parquet": [records[0], "not json", records[1]],
        "b.parquet": [json.dumps(unfinished), '{"schema_version": 2}', "[" * 100_000],
    }
    for name, rows in files.items():
        write_parquet(source / name, rows, row_group_size=1)
    assert run_all("check", source, tmp_path / "run", "--shard-rows", 2) == 1

    lines = (tmp_path / "run" / "issues.jsonl").read_text().splitlines()
    issues = [json.loads(line) for line in lines]

    def rules(name, row):
        found = [i for i in issues if (Path(i["source"]).name, i["row"]) == (name, row)]
        return [(i["rule"], i["severity"]) for i in found]

    assert {(Path(i["source"]).name, i["row"]) for i in issues} == {
        ("a.parquet", 1),
        ("b.parquet", 0),
        ("b.parquet", 1),
        ("b.parquet", 2),
    }
    assert rules("a.parquet", 1) == [("structure/invalid", "error")]
    assert rules("b.parquet", 0) == [("training/final-wait", "warning")]
    assert rules("b.parquet", 2) == [("structure/invalid", "error")]  # too deep to parse
    expected = Conversation.from_json(files["b.parquet"][1]).check(
        checkers=checkers_for("full", training=True)
    )
    assert rules("b.parquet", 1) == [(v.rule, str(v.severity)) for v in expected.violations] + [
        (v.rule, "unevaluated") for v in expected.unevaluated
    ]

    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert (summary["rows"], summary["failed_rows"], summary["rows_with_issues"]) == (6, 3, 4)
    assert sum(issue["count"] for issue in summary["issues"]) == len(issues)

    a, b = str(source / "a.parquet"), str(source / "b.parquet")
    assert issue_rows(tmp_path / "run") == {a: [1], b: [0, 1, 2]}
    assert issue_rows(tmp_path / "run", failed_only=True) == {a: [1], b: [1, 2]}


def test_failing_samples_can_be_selected_with_datasets(tmp_path, records):
    """The README recipe: check a saved HF dataset, then select or drop failing rows."""
    rows = [*records[:3], '{"schema_version": 2}', *records[3:]]
    Dataset.from_dict({"conversation_json": rows}).save_to_disk(str(tmp_path / "native"))
    assert run_all("check", tmp_path / "native", tmp_path / "check", "--shard-rows", 2) == 1

    failed = issue_rows(tmp_path / "check", failed_only=True)
    assert list(failed.values()) == [[3]]
    for source, bad in failed.items():
        data = load_from_disk(source)
        assert data.select(bad)["conversation_json"] == ['{"schema_version": 2}']
        keep = [i for i in range(len(data)) if i not in set(bad)]
        assert data.select(keep)["conversation_json"] == records


@pytest.mark.parametrize("workers", [1, 2])
def test_deep_tool_arguments_are_reported_without_aborting(tmp_path, records, workers):
    record = json.loads(example("native.json"))
    call = next(item for item in record["items"] if item["type"] == "call")
    call["payload"] = '{"city":' + "[" * 100_000 + "0" + "]" * 100_000 + "}"
    source = write_parquet(tmp_path / "input.parquet", [records[0], json.dumps(record), records[1]])
    run = tmp_path / "check"
    assert run_all("check", source, run, workers=workers) == 1

    issues = [json.loads(line) for line in (run / "issues.jsonl").read_text().splitlines()]
    assert len(issues) == 1
    assert (issues[0]["source"], issues[0]["row"]) == (str(source), 1)
    assert (issues[0]["rule"], issues[0]["severity"]) == ("check/exception", "unevaluated")
    assert issues[0]["message"].startswith("RecursionError:")
    assert issue_rows(run, failed_only=True) == {str(source): [1]}
    summary = json.loads((run / "summary.json").read_text())
    assert (summary["rows"], summary["failed_rows"], summary["rows_with_issues"]) == (3, 1, 1)


def test_examples_pass_check_and_weighted_tokens_align(tmp_path, artifact):
    assert {p.name for p in EXAMPLES.glob("*.json")} == set(EXAMPLE_OPTIONS)
    for name, options in EXAMPLE_OPTIONS.items():
        source = write_parquet(tmp_path / f"{name}.parquet", [example(name)])
        assert run_all("check", source, tmp_path / f"check-{name}", *options) == 0
        assert (tmp_path / f"check-{name}" / "issues.jsonl").read_text() == ""
    # Annotated data fails checking unless loss weights are declared.
    source = tmp_path / "weighted.json.parquet"
    assert run_all("check", source, tmp_path / "check-undeclared") == 1

    run = tmp_path / "tokens"
    assert run_all("tokenize", source, run, "--tokenizer", artifact, "--loss-weights") == 0
    tokens, weights = encoded(artifact, [example("weighted.json")])
    assert documents(run / "tokens", np.int32) == tokens
    assert documents(run / "loss_weights", np.float32) == weights
    assert 0.5 in weights[0] and 0.0 in weights[0]


def test_indexed_files_match_megatron_builder_bytes(tmp_path):
    # Produced by Megatron-LM's IndexedDatasetBuilder (swiss-ai revision 04edde0) for
    # the int32 documents [0, 17] and [2**30]. A float32 index differs only in the
    # dtype code at byte 17: 7 instead of 4.
    int32 = (
        "4d4d4944494458000001000000000000000402000000000000000300000000000000"
        "020000000100000000000000000000000800000000000000000000000000000001"
        "000000000000000200000000000000"
    )
    golden = {np.int32: int32, np.float32: int32[:34] + "07" + int32[36:]}
    for dtype, docs in ((np.int32, [[0, 17], [2**30]]), (np.float32, [[0.0, 0.125], [2.5]])):
        with _indexed.Writer(tmp_path / "data", dtype) as writer:
            for doc in docs:
                writer.add(doc)
        assert (tmp_path / "data.idx").read_bytes().hex() == golden[dtype]
        assert (tmp_path / "data.bin").read_bytes() == np.asarray(
            [v for doc in docs for v in doc], dtype
        ).tobytes()
        _indexed.merge([tmp_path / "data", tmp_path / "data"], tmp_path / "merged", dtype)
        assert documents(tmp_path / "merged", dtype) == docs + docs


def test_resume_processes_only_incomplete_shards(tmp_path, artifact, records, monkeypatch):
    source = write_parquet(tmp_path / "input.parquet", records, row_group_size=2)
    run = tmp_path / "run"
    prepare = ["prepare", "tokenize", str(source), str(run), "--tokenizer", str(artifact)]
    prepare += ["--shard-rows", "2"]
    processed = []
    original = _run.Tokenize.__call__

    def crash_on_third_shard(self, source, shard, directory):
        processed.append(shard["start"])
        if shard["start"] == 4 and processed.count(4) == 1:
            raise RuntimeError("node failure")
        return original(self, source, shard, directory)

    monkeypatch.setattr(_run.Tokenize, "__call__", crash_on_third_shard)
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    assert main(prepare) == 0
    assert main(["encode", str(run), "--workers", "1"]) == 2
    assert sorted(os.listdir(run / "shards")) == ["000000", "000001"]
    assert (run / "claims" / "7.0" / "000002").is_dir()  # claimed by the failed step
    assert main(["merge", str(run)]) == 2

    # A later srun step of the same job does not inherit the failed step's claims.
    monkeypatch.setenv("SLURM_STEP_ID", "1")
    assert main(["encode", str(run), "--workers", "1"]) == 0
    assert Counter(processed) == {0: 1, 2: 1, 4: 2, 6: 1}

    # A resubmitted job runs prepare, which removes claims and partial shards of ended jobs.
    (run / "shards" / ".tmp-000002-dead").mkdir()
    assert main(prepare) == 0
    assert not (run / "claims").exists()
    assert not (run / "shards" / ".tmp-000002-dead").exists()
    assert main(["merge", str(run)]) == 0
    assert documents(run / "tokens", np.int32) == encoded(artifact, records)[0]


def test_rejected_inputs_options_and_run_directories(tmp_path, records, capsys):
    jsonl = tmp_path / "input.jsonl"
    jsonl.write_text(records[0] + "\n")
    assert main(["prepare", "check", str(jsonl), str(tmp_path / "from-jsonl")]) == 2
    assert "no .parquet files or saved HF dataset" in capsys.readouterr().err

    source = write_parquet(tmp_path / "input.parquet", records)
    run = tmp_path / "run"
    assert main(["prepare", "check", str(source), str(run)]) == 0
    assert main(["prepare", "check", str(source), str(run), "--loss-weights"]) == 2
    assert "other options" in capsys.readouterr().err
    write_parquet(source, [*records, records[0]])
    assert main(["encode", str(run), "--workers", "1"]) == 2
    assert main(["merge", str(run)]) == 2
    assert "changed since prepare" in capsys.readouterr().err
    # Deleting run.json does not turn a used run directory into a fresh one.
    used = tmp_path / "used"
    assert run_all("check", source, used) == 0
    (used / "run.json").unlink()
    assert main(["prepare", "check", str(source), str(used), "--loss-weights"]) == 2
    assert "not empty" in capsys.readouterr().err
    assert main(["merge", str(tmp_path / "unprepared")]) == 2
    assert "not prepared" in capsys.readouterr().err


@pytest.mark.parametrize("size,offset", [(1024, 10), (200_000, 0), (200_000, 199_999)])
def test_input_fingerprint_detects_same_size_same_mtime_rewrites(tmp_path, size, offset):
    path = tmp_path / "data"
    path.write_bytes(b"a" * size)
    before = _inputs.snapshot(path)
    with path.open("r+b") as stream:
        stream.seek(offset)
        stream.write(b"b")
    os.utime(path, ns=(path.stat().st_atime_ns, before[0]["mtime_ns"]))
    after = _inputs.snapshot(path)
    assert after[0]["size"] == before[0]["size"]
    assert after[0]["mtime_ns"] == before[0]["mtime_ns"]
    assert after[0]["sample_sha256"] != before[0]["sample_sha256"]


@pytest.mark.parametrize("field", ["version", "commit"])
def test_encode_and_merge_reject_changed_library(tmp_path, records, monkeypatch, capsys, field):
    source = write_parquet(tmp_path / "input.parquet", records)
    run = tmp_path / "run"
    assert main(["prepare", "check", str(source), str(run)]) == 0
    identity = _run.library()
    monkeypatch.setattr(_run, "library", lambda: identity | {field: "changed"})
    assert main(["encode", str(run), "--workers", "1"]) == 2
    assert "apertus-common version or revision changed since prepare" in capsys.readouterr().err
    assert not (run / "shards").exists()

    monkeypatch.setattr(_run, "library", lambda: identity)
    assert main(["encode", str(run), "--workers", "1"]) == 0
    monkeypatch.setattr(_run, "library", lambda: identity | {field: "changed"})
    assert main(["merge", str(run)]) == 2
    assert "apertus-common version or revision changed since prepare" in capsys.readouterr().err
    assert not (run / "summary.json").exists()
    assert not (run / "issues.jsonl").exists()

    monkeypatch.setattr(_run, "library", lambda: identity)
    assert main(["merge", str(run)]) == 0


def test_tokenize_stops_at_unparseable_record(tmp_path, artifact, records, capsys):
    source = write_parquet(tmp_path / "input.parquet", [records[0], "not json"])
    run = tmp_path / "run"
    assert main(["prepare", "tokenize", str(source), str(run), "--tokenizer", str(artifact)]) == 0
    assert main(["encode", str(run), "--workers", "1"]) == 2
    assert "row 1 is not a native conversation; run check first" in capsys.readouterr().err
    assert main(["encode", str(run), "--workers", "2"]) == 2  # failing worker process


def test_empty_input(tmp_path, artifact):
    source = write_parquet(tmp_path / "input.parquet", [])
    assert run_all("check", source, tmp_path / "check") == 0
    assert run_all("tokenize", source, tmp_path / "tokens", "--tokenizer", artifact) == 0
    assert documents(tmp_path / "tokens" / "tokens", np.int32) == []


@pytest.mark.skipif(not APERTUS_DATA.exists(), reason="install the apertus-data entry point")
def test_slurm_script_on_two_simulated_nodes(tmp_path, artifact, records):
    """Two srun tasks with three workers each share the shards; resubmitting resumes."""
    source = write_parquet(tmp_path / "input.parquet", records * 3, row_group_size=2)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "srun").write_text(
        "#!/bin/bash\n"
        "tasks=1\n"
        "while [[ $1 == --* ]]; do\n"
        "  [[ $1 == --ntasks=* ]] && tasks=${1#*=}\n"
        "  shift\n"
        "done\n"
        'if [[ $tasks == 1 ]]; then exec "$@"; fi\n'
        'SLURM_NTASKS=2 SLURM_PROCID=0 "$@" & first=$!\n'
        'SLURM_NTASKS=2 SLURM_PROCID=1 "$@"; second=$?\n'
        "wait $first && exit $second\n"
    )
    (bin_dir / "srun").chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SLURM_CPUS_ON_NODE": "3",
        "SLURM_JOB_ID": "",
        "SLURM_JOB_NUM_NODES": "2",
        "UV_PROJECT_ENVIRONMENT": str(APERTUS_DATA.parent.parent),
        "APERTUS_ENVIRONMENT": "none",
        "WORKERS": "3",
    }
    command = ["bash", "slurm/run.sh", "tokenize", str(source), str(tmp_path / "run")]
    command += ["--tokenizer", str(artifact), "--shard-rows", "2"]

    def submit(**extra):
        return subprocess.run(
            command, cwd=ROOT, env=env | extra, capture_output=True, text=True, check=False
        )

    assert submit().returncode == 2  # not inside a Slurm job
    first = submit(SLURM_JOB_ID="42")
    assert first.returncode == 0, first.stderr
    done = [line for line in first.stderr.splitlines() if " done: " in line]
    assert sorted(done) == sorted(
        f"shard {i}/11 done: {2 if i < 11 else 1} rows" for i in range(1, 12)
    )
    assert documents(tmp_path / "run" / "tokens", np.int32) == encoded(artifact, records * 3)[0]

    again = submit(SLURM_JOB_ID="43")
    assert again.returncode == 0, again.stderr
    assert " done: " not in again.stderr


def test_mapper_waits_after_every_assistant_answer():
    spec = importlib.util.spec_from_file_location("no_robots", ROOT / "mappers" / "no_robots.py")
    mapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapper)
    multi_turn = [
        {"role": "system", "content": "Answer in rhymes."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there"},
        {"role": "user", "content": "Fine"},
        {"role": "assistant", "content": "Glad to hear"},
    ]
    single_turn = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    checkers = checkers_for("full", training=True)
    for messages, items, behavior in (
        (multi_turn, ["user", "reply", "wait", "user", "reply", "wait"], "Answer in rhymes."),
        (single_turn, ["user", "reply", "wait"], None),
    ):
        raw = mapper.to_native(messages)
        assert all("direction" not in item for item in json.loads(raw)["items"])
        conversation = Conversation.from_json(raw)
        assert conversation.system is not None and conversation.system.thinking == "medium"
        assert conversation.system.behavior == behavior
        assert [item.type for item in conversation.items] == items
        report = conversation.check(checkers=checkers)
        assert report.ok and report.complete and not report.violations
    # A user line mislabelled as a second assistant answer is skipped, not merged.
    mislabelled = [*single_turn, {"role": "assistant", "content": "Thanks!"}]
    assert not mapper.is_turn_based(mislabelled)
    with pytest.raises(ValueError, match="alternate"):
        mapper.to_native(mislabelled)


@pytest.mark.skipif(not os.environ.get("APERTUS_ENCODING"), reason="set APERTUS_ENCODING")
def test_real_tokenizer(tmp_path):
    artifact = os.environ["APERTUS_ENCODING"]
    source = write_parquet(tmp_path / "native.parquet", [example("native.json")])
    assert run_all("tokenize", source, tmp_path / "run", "--tokenizer", artifact) == 0
    assert (
        documents(tmp_path / "run" / "tokens", np.int32)
        == encoded(artifact, [example("native.json")])[0]
    )


@pytest.mark.parametrize("legacy", ["direction", "v1"])
def test_old_native_shape_is_reported_and_tokenization_stops(tmp_path, artifact, records, legacy):
    record = json.loads(records[0])
    if legacy == "direction":
        record["items"][0]["direction"] = "in"
    else:
        record["schema_version"] = 1
    source = write_parquet(tmp_path / "old.parquet", [json.dumps(record)])
    assert run_all("check", source, tmp_path / "check") == 1
    issues = [
        json.loads(line) for line in (tmp_path / "check/issues.jsonl").read_text().splitlines()
    ]
    assert [(issue["row"], issue["rule"], issue["severity"]) for issue in issues] == [
        (0, "structure/invalid", "error")
    ]
    assert (
        main(
            [
                "prepare",
                "tokenize",
                str(source),
                str(tmp_path / "tokens"),
                "--tokenizer",
                str(artifact),
            ]
        )
        == 0
    )
    assert main(["encode", str(tmp_path / "tokens"), "--workers", "1"]) == 2
    assert not (tmp_path / "tokens/tokens.bin").exists()
    assert not list((tmp_path / "tokens/shards").iterdir())


def test_unknown_output_is_parsed_but_reported_as_nonconformant(tmp_path):
    from apertus_common import SystemPrompt, UnknownOutput, Wait

    record = Conversation(
        system=SystemPrompt.build(),
        items=[
            UnknownOutput(output_type="user", payload="generated, not user input"),
            Wait(),
        ],
    ).to_json()
    source = write_parquet(tmp_path / "unknown.parquet", [record])
    assert run_all("check", source, tmp_path / "check") == 1
    issues = [
        json.loads(line) for line in (tmp_path / "check/issues.jsonl").read_text().splitlines()
    ]
    assert any(
        issue["rule"] == "profile/known-output" and "'user'" in issue["message"] for issue in issues
    )
    assert not any(issue["rule"] == "structure/invalid" for issue in issues)
