import os

import pytest
from conftest import encoded, run_all, write_parquet
from datasets import Dataset

from apertus2_processing import _inspect
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


def blocks(capsys):
    return [b for b in capsys.readouterr().out.split("=== document ") if b]


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
    assert "<|wait|>" in shown[0] and "with loss" not in shown[0]
    assert main(["inspect", str(run), "--index", "2", "--weights"]) == 0
    assert "with loss (weights 0, 1)" in blocks(capsys)[0]
    assert f"--- source {source} row 2\n{records[2]}" in shown[0]
    assert "question 6 🙂" in shown[1]

    ids, weights = encoded(artifact, records)
    assert main(["inspect", str(run), "--index", "3", "--tokens", "--weights"]) == 0
    lines = capsys.readouterr().out.splitlines()
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
        assert block.startswith(f"{index} (of 7): ") and "with loss" not in block
        assert f"--- source {source / file} row {row}\n{records[index]}" in block
        assert f"question {index} 🙂" in block

    assert main(["inspect", str(run), "--index", "1", "--tokens"]) == 0
    assert all(len(line.split(None, 2)) == 3 for line in capsys.readouterr().out.splitlines()[1:])
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


def test_inspect_rejects_other_directories(tmp_path, records, capsys):
    source = tmp_path / "native"
    Dataset.from_dict({"conversation_json": records}).save_to_disk(str(source))
    assert main(["inspect", str(tmp_path / "nowhere")]) == 2
    assert "not a merged run" in capsys.readouterr().err
    check = tmp_path / "check"
    assert run_all("check", source, check) == 0
    assert main(["inspect", str(check)]) == 2
    assert "is a check run" in capsys.readouterr().err
