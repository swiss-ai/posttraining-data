import json
import os
import re
import shutil
import subprocess

import pytest
from apertus_common import (
    Call,
    Claim,
    Conversation,
    Document,
    Host,
    Reply,
    Result,
    SystemPrompt,
    Think,
    ToolSpec,
    User,
    Wait,
    load_encoding,
)
from conftest import encoded, run_all, write_parquet
from datasets import Dataset

from apertus2_processing import _html, _inspect
from apertus2_processing.cli import main


def test_choose_uses_indices_or_a_seeded_sample():
    assert _inspect._choose(7, [6, 0], 3, None) == [6, 0]
    sample = _inspect._choose(7, (), 3, 1)
    assert len(sample) == 3 and sample == sorted(sample)
    assert _inspect._choose(7, (), 3, 1) == sample
    assert _inspect._choose(2, (), 5, 1) == [0, 1]
    with pytest.raises(ValueError, match="out of range"):
        _inspect._choose(7, [7], 3, None)
    with pytest.raises(ValueError, match="negative"):
        _inspect._choose(7, (), -1, None)


def printed(capsys):
    """Captured stdout without the leading `=== run` line."""
    return capsys.readouterr().out.partition("\n")[2]


def blocks(capsys):
    return [b for b in printed(capsys).split("=== document ") if b]


def test_inspect_decodes_documents_and_lists_tokens_with_weights(
    tmp_path, artifact, records, capsys
):
    source = tmp_path / "native"
    Dataset.from_dict({"conversation_json": records}).save_to_disk(str(source))
    run = tmp_path / "tokens"
    assert run_all("tokenize", source, run, "--tokenizer", artifact, "--loss-weights") == 0
    capsys.readouterr()

    assert main(["inspect", str(run), "--index", "2", "--index", "6", "--source"]) == 0
    shown = blocks(capsys)
    assert [b.split(" ")[0] for b in shown] == ["2", "6"]
    assert "question 2 🙂" in shown[0] and "answer answer answer" in shown[0]
    assert "<|wait|>" in shown[0] and "with loss (weights 0, 1, trainer defaults)" in shown[0]
    assert main(["inspect", str(run), "--index", "2", "--weights"]) == 0
    assert "with loss (weights 0, 1, from loss_weights)" in blocks(capsys)[0]
    assert f"--- source {source} row 2\n{records[2]}" in shown[0]
    assert "question 6 🙂" in shown[1]

    ids, weights = encoded(artifact, records)
    assert main(["inspect", str(run), "--index", "3", "--tokens", "--weights"]) == 0
    lines = printed(capsys).splitlines()
    assert lines[0].startswith(f"=== document 3 (of 7): {len(ids[3])} tokens, ")
    rows = [line.split(None, 3) for line in lines[1:]]
    assert [int(r[0]) for r in rows] == list(range(len(ids[3])))
    assert [int(r[1]) for r in rows] == ids[3]
    assert [r[2] for r in rows] == [f"{w:g}" for w in weights[3]]

    assert main(["inspect", str(run), "--sample", "2", "--seed", "3"]) == 0
    assert len(blocks(capsys)) == 2

    # Weights are validated only when requested; a consistent but foreign pair is rejected.
    short = tmp_path / "short"
    Dataset.from_dict({"conversation_json": records[:2]}).save_to_disk(str(short))
    assert (
        run_all(
            "tokenize", short, tmp_path / "short-run", "--tokenizer", artifact, "--loss-weights"
        )
        == 0
    )
    for name in ("loss_weights.bin", "loss_weights.idx"):
        (run / name).write_bytes((tmp_path / "short-run" / name).read_bytes())
    capsys.readouterr()
    assert main(["inspect", str(run), "--index", "0"]) == 0
    capsys.readouterr()
    assert main(["inspect", str(run), "--index", "0", "--weights"]) == 2
    assert "disagrees with the tokens index" in capsys.readouterr().err


def test_inspect_maps_documents_to_rows_across_parquet_files(tmp_path, artifact, records, capsys):
    source = tmp_path / "input"
    (source / "nested").mkdir(parents=True)
    write_parquet(source / "a.parquet", records[:4], row_group_size=3)
    write_parquet(source / "nested" / "b.parquet", records[4:], row_group_size=2)
    run = tmp_path / "tokens"
    assert run_all("tokenize", source, run, "--tokenizer", artifact, "--shard-rows", 2) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "inspect",
                str(run),
                *(["--index", "0", "--index", "3", "--index", "4", "--index", "6"]),
                "--source",
            ]
        )
        == 0
    )
    shown = blocks(capsys)
    expected = [
        ("a.parquet", 0),
        ("a.parquet", 3),
        ("nested/b.parquet", 0),
        ("nested/b.parquet", 2),
    ]
    for block, index, (file, row) in zip(shown, [0, 3, 4, 6], expected, strict=True):
        assert block.startswith(f"{index} (of 7): ") and "trainer defaults" in block
        assert f"--- source {source / file} row {row}\n{records[index]}" in block
        assert f"question {index} 🙂" in block

    # Without a written mask the weight column holds the trainer defaults, which equal the
    # encoder output for data without annotations.
    ids, weights = encoded(artifact, records)
    assert main(["inspect", str(run), "--index", "1", "--tokens"]) == 0
    rows = [line.split(None, 3) for line in printed(capsys).splitlines()[1:]]
    assert [float(r[2]) for r in rows] == weights[1] and [int(r[1]) for r in rows] == ids[1]
    assert main(["inspect", str(run), "--index", "1", "--weights"]) == 2
    assert "without --loss-weights" in capsys.readouterr().err

    # Rows are only shown while the inputs are exactly those that were tokenized.
    (source / "a.parquet").write_bytes((source / "a.parquet").read_bytes()[:-1] + b"x")
    assert main(["inspect", str(run), "--index", "0", "--source"]) == 2
    assert "changed since tokenization" in capsys.readouterr().err
    assert main(["inspect", str(run), "--index", "0"]) == 0  # tokens alone remain readable
    capsys.readouterr()
    os.rename(source, tmp_path / "moved")
    assert main(["inspect", str(run), "--index", "0", "--source"]) == 2
    assert "no longer exists" in capsys.readouterr().err


def test_inspect_prints_the_run_overview_first(tmp_path, artifact, records, capsys):
    source = tmp_path / "native"
    Dataset.from_dict({"conversation_json": records}).save_to_disk(str(source))
    run = tmp_path / "tokens"
    assert run_all("tokenize", source, run, "--tokenizer", artifact) == 0
    summary = json.loads((run / "summary.json").read_text())
    capsys.readouterr()
    assert main(["inspect", str(run), "--sample", "0"]) == 0
    line = capsys.readouterr().out
    length = summary["length"]
    assert line == (
        f"=== run {run}: 7 documents, {summary['tokens']:,} tokens; length mean "
        f"{length['mean']:,.0f}, p50 {length['p50']}, p90 {length['p90']}, p99 {length['p99']}, "
        f"max {length['max']}; >4k 0 (0.0%), >8k 0 (0.0%), >16k 0 (0.0%), >32k 0 (0.0%)\n"
    )
    # Validation precedes the run line: a failing inspect leaves stdout empty.
    assert main(["inspect", str(run), "--index", "99"]) == 2
    out, err = capsys.readouterr()
    assert out == "" and "out of range" in err


def test_inspect_rejects_other_directories(tmp_path, records, capsys):
    source = tmp_path / "native"
    Dataset.from_dict({"conversation_json": records}).save_to_disk(str(source))
    assert main(["inspect", str(tmp_path / "nowhere")]) == 2
    assert "not a merged run" in capsys.readouterr().err
    check = tmp_path / "check"
    assert run_all("check", source, check) == 0
    assert main(["inspect", str(check)]) == 2
    assert "is a check run" in capsys.readouterr().err


def payload_of(page):
    match = re.search(r'<script id="data"[^>]*>(.*?)</script>', page, re.DOTALL)
    return json.loads(match.group(1))


def test_default_weights_match_the_library(artifact):
    encoding = load_encoding(artifact)
    tool = ToolSpec(name="add", description="Add.", schema={"type": "object", "properties": {}})
    conversation = Conversation(
        system=SystemPrompt.build(tools=[tool], behavior="Be brief."),
        items=[
            Document(payload="notes", from_="user", id="a1"),
            User(payload="q [a1]"),
            Think(payload="plan"),
            Call(name="add", counter=0, payload="{}"),
            Wait(),
            Result(name="add", counter=0, payload="1"),
            Host(payload="notice"),
            Think(payload="done"),
            Claim(payload="1"),
            Reply(payload="one"),
            Wait(),
        ],
    )
    result = encoding.encode_conversation(conversation, return_loss_weights=True)
    assert _inspect._default_weights(result.token_ids, encoding.controls) == result.loss_weights
    assert 0.0 in result.loss_weights and 1.0 in result.loss_weights


def test_inspect_output_writes_text_files_and_html_report(tmp_path, artifact, records, capsys):
    hostile = '<!--<script>alert(1)</script> "quoted" & done'
    annotated = Conversation(
        system=SystemPrompt.build(),
        items=[User(payload=hostile), Reply(payload="ok", loss_weight=0.5), Wait()],
    ).to_json()
    source = tmp_path / "native"
    Dataset.from_dict({"conversation_json": [*records, annotated]}).save_to_disk(str(source))
    run = tmp_path / "tokens"
    assert run_all("tokenize", source, run, "--tokenizer", artifact, "--loss-weights") == 0
    capsys.readouterr()

    out = tmp_path / "report"
    (out / "document-000001.txt").parent.mkdir()
    (out / "document-000001.txt").write_text("stale")
    args = ["inspect", str(run), "--index", "3", "--index", "7", "--tokens", "--source"]
    assert main([*args, "--output", str(out)]) == 0
    assert "wrote 2 documents and report.html to" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == [
        "document-000003.txt",
        "document-000007.txt",
        "report.html",
    ]
    assert main(args) == 0
    texts = [(out / f"document-00000{i}.txt").read_text() for i in (3, 7)]
    assert "".join(texts) == printed(capsys)

    page = (out / "report.html").read_text()
    assert page.count("<script") == 2 and "<!--" not in page.split("</header>")[1]
    payload = payload_of(page)
    assert payload["stats"]["run"] == str(run)
    assert [d["index"] for d in payload["documents"]] == [3, 7]
    assert payload["stats"]["documents"] == 8 and payload["stats"]["length"]["max"] >= 1
    assert "p50 " in page.split("</header>")[0]
    ids, weights = encoded(artifact, records)
    doc, last = payload["documents"]
    assert doc["ids"] == ids[3] and doc["weights"] == weights[3] and not doc["weights_written"]
    assert doc["meta"].endswith("trainer defaults)") and doc["source"]["row"] == 3
    assert hostile in last["text"] and last["source"]["json"] == annotated
    assert set(doc["controls"]) >= {encoding_controls for encoding_controls in doc["ids"][:2]}

    # Written weights differ from the defaults on annotated data, and the report says so.
    assert main([*args, "--weights", "--output", str(out)]) == 0
    doc, last = payload_of((out / "report.html").read_text())["documents"]
    assert last["weights_written"] and 0.5 in last["weights"] and 0.5 not in doc["weights"]
    assert (
        last["meta"].endswith("from loss_weights)")
        and (out / "document-000007.txt").read_text().count(" 0.5 ") > 0
    )

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available to syntax-check the report script")
    script = tmp_path / "report.js"
    script.write_text(_html.SCRIPT)
    subprocess.run([node, "--check", str(script)], check=True)
