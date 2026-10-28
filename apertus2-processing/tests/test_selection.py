import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from conftest import documents, encoded, main, run_all, write_parquet
from datasets import Dataset

from apertus2_processing import _indexed, _selection

SPLIT = ("train", "validation")
GROUP = ["a", "a", "b", "b", "c", "c", "d"]
HALF = ["--validation", "0.5"]


def tokenized(tmp_path, artifact, records, **columns):
    """Return the input, the tokenize run, and the expected tokens, weights and lengths."""
    source = tmp_path / "input"
    source.mkdir()
    write_parquet(source / "a.parquet", records, row_group_size=3, **columns)
    run = tmp_path / "tokens"
    assert run_all("tokenize", source, run, "--tokenizer", artifact, "--loss-weights") == 0
    tokens, weights = encoded(artifact, records)
    return source, run, tokens, weights, [len(t) for t in tokens]


def assert_selected(directory, kept, tokens, weights):
    """`directory` holds exactly the documents `kept`, in order, with their origins."""
    assert documents(directory / "tokens", np.int32) == [tokens[i] for i in kept]
    assert documents(directory / "loss_weights", np.float32) == [weights[i] for i in kept]
    assert np.load(directory / "documents.npy").tolist() == kept


def sides(output):
    """The original document indices written to train and to validation."""
    return [np.load(output / part / "documents.npy").tolist() for part in SPLIT]


def assert_grouped(output):
    """Both sides hold documents, and no `GROUP` value is on both."""
    train, validation = ({GROUP[i] for i in part} for part in sides(output))
    assert train and validation and not train & validation


def test_buckets_split_every_document_by_length(tmp_path, artifact, records, capsys):
    source, run, tokens, weights, lengths = tokenized(tmp_path, artifact, records)
    short, middle = sorted(set(lengths))[:2]  # the longest documents fall above both bounds
    output = tmp_path / "buckets"
    capsys.readouterr()
    assert main(["bucket", str(run), str(output), "--bounds", str(short), str(middle)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == json.loads((output / "summary.json").read_text())

    names = [f"1-{short}", f"{short + 1}-{middle}", f"{middle + 1}+"]
    expected = {name: [] for name in names}
    for index, length in enumerate(lengths):
        expected[names[(length > short) + (length > middle)]].append(index)
    assert [entry["name"] for entry in summary["buckets"]] == names
    for entry in summary["buckets"]:
        kept, bucket = expected[entry["name"]], output / entry["name"]
        assert_selected(bucket, kept, tokens, weights)
        assert entry["documents"] == len(kept)
        assert entry["tokens"] == sum(lengths[i] for i in kept)
        result = json.loads((bucket / "summary.json").read_text())
        assert result["documents"] == len(kept) and result["mode"] == "tokenize"
    assert summary["buckets"][-1]["max_tokens"] is None
    assert summary["documents"] == len(records) and summary["tokens"] == sum(lengths)

    # A bucket reads like a run: inspect maps its documents to the source rows.
    index = expected[names[1]][0]
    assert main(["inspect", str(output / names[1]), "--index", "0", "--source"]) == 0
    shown = capsys.readouterr().out
    assert f"--- source {source / 'a.parquet'} row {index}\n{records[index]}" in shown


def test_cap_keeps_documents_up_to_max_tokens(tmp_path, artifact, records):
    _, run, tokens, weights, lengths = tokenized(tmp_path, artifact, records)
    limit = sorted(set(lengths))[1]
    kept = [i for i, length in enumerate(lengths) if length <= limit]
    output = tmp_path / "capped"
    assert main(["cap", str(run), str(output), "--max-tokens", str(limit)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert_selected(output, kept, tokens, weights)
    assert summary["documents"] == len(kept) and summary["mode"] == "tokenize"
    dropped = [length for length in lengths if length > limit]
    assert dropped and summary["dropped"] == {"documents": len(dropped), "tokens": sum(dropped)}
    assert summary["length"]["max"] == limit


def test_selections_narrow_to_the_parent_and_keep_original_positions(
    tmp_path, artifact, records, capsys
):
    _, run, _, _, lengths = tokenized(tmp_path, artifact, records)
    short, middle = sorted(set(lengths))[:2]
    capped = tmp_path / "capped"
    assert main(["cap", str(run), str(capped), "--max-tokens", str(middle)]) == 0
    bounds = [str(short), str(middle + 100)]
    assert main(["bucket", str(capped), str(tmp_path / "again"), "--bounds", *bounds]) == 0
    summary = json.loads((tmp_path / "again" / "summary.json").read_text())
    # The second range stops at the cap; the open range above the last bound is left out.
    assert [b["name"] for b in summary["buckets"]] == [f"1-{short}", f"{short + 1}-{middle}"]
    nested = tmp_path / "again" / f"{short + 1}-{middle}"
    kept = [i for i, length in enumerate(lengths) if short < length <= middle]
    assert np.load(nested / "documents.npy").tolist() == kept
    assert json.loads((nested / "run.json").read_text())["selection"]["run"] == str(run.resolve())
    capsys.readouterr()
    assert main(["inspect", str(nested), "--index", "0", "--source"]) == 0
    assert f"row {kept[0]}\n{records[kept[0]]}" in capsys.readouterr().out

    # Capping a bucket below its own range is refused.
    assert main(["cap", str(nested), str(tmp_path / "none"), "--max-tokens", str(short)]) == 2
    assert f"below the run's minimum of {short + 1}" in capsys.readouterr().err


def test_selections_reject_bad_limits_and_outputs(tmp_path, artifact, records, capsys):
    run = tokenized(tmp_path, artifact, records)[1]
    for bounds in (["0"], ["8", "8"], ["9", "8"]):
        assert main(["bucket", str(run), str(tmp_path / "out"), "--bounds", *bounds]) == 2
        assert "strictly increasing" in capsys.readouterr().err
    assert main(["cap", str(run), str(tmp_path / "out"), "--max-tokens", "0"]) == 2
    assert "must be positive" in capsys.readouterr().err
    assert main(["cap", str(run), str(run / "inside"), "--max-tokens", "8"]) == 2
    assert "outside the run" in capsys.readouterr().err
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "file").write_text("")
    for output in (tmp_path / "full", tmp_path / "full" / "file"):
        assert main(["bucket", str(run), str(output), "--bounds", "8"]) == 2
        assert "is not an empty directory" in capsys.readouterr().err


def test_failed_selections_leave_no_partial_output(tmp_path, artifact, records, monkeypatch):
    run = tokenized(tmp_path, artifact, records)[1]

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(_indexed, "select", fail)
    assert main(["bucket", str(run), str(tmp_path / "failed"), "--bounds", "8"]) == 2
    assert not (tmp_path / "failed").exists() and not list(tmp_path.glob(".tmp-*"))
    monkeypatch.undo()

    # A killed job skips the cleanup; the next selection into the output removes its staging.
    # Another output's staging, even one with this output's name as prefix, stays.
    for name in (".tmp-failed-0123abcd", ".tmp-failed-v2-0123abcd"):
        (tmp_path / name).mkdir()
    assert main(["bucket", str(run), str(tmp_path / "failed"), "--bounds", "8"]) == 0
    assert [p.name for p in tmp_path.glob(".tmp-*")] == [".tmp-failed-v2-0123abcd"]


def test_selections_are_marked_and_refused_as_prepared_runs(tmp_path, artifact, records, capsys):
    source, run = tokenized(tmp_path, artifact, records)[:2]
    capped = tmp_path / "out"
    assert main(["cap", str(run), str(capped), "--max-tokens", "4096"]) == 0
    prepare = ["prepare", "tokenize", str(source), str(capped), "--tokenizer", str(artifact)]
    for command in (prepare + ["--loss-weights"], ["encode", str(capped)], ["merge", str(capped)]):
        assert main(command) == 2
        assert "is a selection" in capsys.readouterr().err

    # documents.npy must exist and describe the selection's own documents.
    np.save(capped / "documents.npy", np.arange(1))
    assert main(["inspect", str(capped), "--index", "0", "--source"]) == 2
    assert "disagrees with the tokens index" in capsys.readouterr().err
    (capped / "documents.npy").unlink()
    assert main(["inspect", str(capped), "--index", "0", "--source"]) == 2
    assert "is a selection without documents.npy" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command, options",
    [
        ("cap", ["--max-tokens", "4096"]),
        ("bucket", ["--bounds", "4096"]),
        ("split", HALF),
        ("sample", ["--fraction", "0.5"]),
    ],
)
def test_selections_lock_active_writers_and_recover_after_exit(
    tmp_path, artifact, records, capsys, command, options
):
    run = tokenized(tmp_path, artifact, records)[1]
    output = tmp_path / "out"
    script = """
import sys
from apertus2_processing._selection import _Parent
with _Parent(sys.argv[1], sys.argv[2]).stage() as directory:
    print(directory, flush=True)
    sys.stdin.readline()
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(run), str(output)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as writer:
        try:
            active = Path(writer.stdout.readline().strip())
            assert active.parent == tmp_path and active.is_dir()
            args = [command, str(run), str(output), *options]
            assert main(args) == 2
            assert "another selection is writing" in capsys.readouterr().err
            assert active.is_dir() and not output.exists()
        finally:
            writer.kill()
            writer.wait(timeout=10)
    # A killed writer releases the lock but leaves staging to be cleaned by the retry.
    assert active.is_dir()
    assert main(args) == 0
    assert not active.exists() and (output / "summary.json").exists()


def test_split_writes_a_hashed_share_to_validation(tmp_path, artifact, records):
    _, run, tokens, weights, lengths = tokenized(tmp_path, artifact, records)
    output = tmp_path / "split"
    assert main(["split", str(run), str(output), *HALF]) == 0
    train, validation = sides(output)
    assert sorted(train + validation) == list(range(len(records))) and validation
    assert_selected(output / "validation", validation, tokens, weights)
    summary = json.loads((output / "summary.json").read_text())
    assert summary == {
        "mode": "split",
        "run": str(run.resolve()),
        "fraction": 0.5,
        "seed": 0,
        "group_by": None,
        "train": {"documents": len(train), "tokens": sum(lengths[i] for i in train)},
        "validation": {"documents": len(validation), "tokens": sum(lengths[i] for i in validation)},
    }


def test_split_agrees_across_selections(tmp_path, artifact, records):
    _, run, _, _, lengths = tokenized(tmp_path, artifact, records)
    output = tmp_path / "split"
    assert main(["split", str(run), str(output), *HALF]) == 0
    validation = sides(output)[1]

    # A cap splits like the full run.
    limit = sorted(set(lengths))[-2]
    capped = tmp_path / "capped"
    assert main(["cap", str(run), str(capped), "--max-tokens", str(limit)]) == 0
    assert main(["split", str(capped), str(tmp_path / "s"), *HALF]) == 0
    assert sides(tmp_path / "s")[1] == [i for i in validation if lengths[i] <= limit]

    # Splitting a split records both.
    again = tmp_path / "again"
    assert main(["split", str(output / "train"), str(again), *HALF, "--seed", "1"]) == 0
    splits = json.loads((again / "validation" / "run.json").read_text())["selection"]["splits"]
    assert [(s["seed"], s["part"]) for s in splits] == [(0, "train"), (1, "validation")]


def test_split_groups_parquet_rows_of_a_selection(tmp_path, artifact, records):
    _, run, _, _, lengths = tokenized(tmp_path, artifact, records, group=GROUP)
    limit = sorted(set(lengths))[-2]
    capped = tmp_path / "capped"
    assert main(["cap", str(run), str(capped), "--max-tokens", str(limit)]) == 0
    grouped = tmp_path / "grouped"
    assert main(["split", str(capped), str(grouped), *HALF, "--group-by", "group"]) == 0
    assert_grouped(grouped)
    selection = json.loads((grouped / "train" / "run.json").read_text())["selection"]
    assert selection["max_tokens"] == limit and selection["splits"][0]["part"] == "train"


def test_split_groups_equal_numeric_ids_across_parquet_schemas(tmp_path, artifact, records):
    source = tmp_path / "input"
    source.mkdir()
    groups = [[0, 1, 2, 3], [-0.0, 1.0, 2.0, 3.0]]
    for index, values in enumerate(groups):
        write_parquet(source / f"{index}.parquet", records[:4], group=values)
    run = tmp_path / "tokens"
    assert run_all("tokenize", source, run, "--tokenizer", artifact) == 0
    output = tmp_path / "split"
    assert main(["split", str(run), str(output), *HALF, "--group-by", "group"]) == 0
    train, validation = sides(output)
    assert train and validation and sorted(train + validation) == list(range(8))
    assert not {i % 4 for i in train} & {i % 4 for i in validation}


@pytest.mark.parametrize(
    "left, right",
    [(0, -0.0), (2, 2.0), (1, True), (0.5, Decimal("0.5")), (2**60, Decimal(2**60))],
)
def test_group_hash_preserves_numeric_equality(left, right):
    assert _selection._hashed(left, 0) == _selection._hashed(right, 0)


def test_group_hash_keeps_large_distinct_integers():
    assert _selection._hashed(2**60, 0) != _selection._hashed(2**60 + 1, 0)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), Decimal("NaN")])
def test_group_hash_rejects_nonfinite_numbers(value):
    with pytest.raises(ValueError, match="numeric group values must be finite"):
        _selection._hashed(value, 0)


def test_split_rejects_bad_fractions_groups_and_outputs(tmp_path, artifact, records, capsys):
    source, run = tokenized(tmp_path, artifact, records, missing=[None, *"abcdef"])[:2]
    command = ["split", str(run), str(tmp_path / "out"), "--validation"]
    for args, error in [
        (["0"], "between 0 and 1"),
        (["1"], "between 0 and 1"),
        (["0.001"], "leaves train or validation empty"),
        (["0.5", "--group-by", "missing"], "missing values"),
        (["0.5", "--group-by", "absent"], "column 'absent' not in"),
    ]:
        assert main([*command, *args]) == 2
        assert error in capsys.readouterr().err
    assert main(["split", str(run), str(source / "out"), "--validation", "0.5"]) == 2
    assert "outside the run and its input" in capsys.readouterr().err
    write_parquet(source / "a.parquet", records, missing=list("abcdefg"))
    assert main([*command, "0.5", "--group-by", "missing"]) == 2
    assert "changed" in capsys.readouterr().err


def test_split_groups_rows_of_a_saved_hf_dataset(tmp_path, artifact, records):
    source = tmp_path / "native"
    Dataset.from_dict({"conversation_json": records, "group": GROUP}).save_to_disk(str(source))
    run = tmp_path / "tokens"
    assert run_all("tokenize", source, run, "--tokenizer", artifact) == 0
    output = tmp_path / "split"
    assert main(["split", str(run), str(output), *HALF, "--group-by", "group"]) == 0
    assert_grouped(output)


def sample_hashes(count, seed=0):
    """Per-index hashes that `sample` sorts by (pins its `sample:{seed}` salt)."""
    return _selection._hashes(range(count), f"sample:{seed}").tolist()


def test_sample_takes_a_seeded_prefix_by_documents_or_tokens(tmp_path, artifact, records):
    _, run, tokens, weights, lengths = tokenized(tmp_path, artifact, records)
    hashes = sample_hashes(len(records))
    order = sorted(range(len(records)), key=hashes.__getitem__)
    budget = sum(lengths[i] for i in order[:4]) - 1  # the fourth document does not fit
    for name, options, kept in [
        ("first-two", ["--documents", "2"], order[:2]),
        ("budget", ["--tokens", str(budget)], order[:3]),
        ("all", ["--documents", str(len(records) + 1)], order),
    ]:
        output = tmp_path / name
        assert main(["sample", str(run), str(output), *options]) == 0
        assert_selected(output, sorted(kept), tokens, weights)
        summary = json.loads((output / "summary.json").read_text())
        dropped = [length for i, length in enumerate(lengths) if i not in kept]
        assert summary["tokens"] == sum(lengths[i] for i in kept)
        assert summary["dropped"] == {"documents": len(dropped), "tokens": sum(dropped)}


def test_sample_fraction_agrees_across_selections(tmp_path, artifact, records):
    _, run, tokens, weights, lengths = tokenized(tmp_path, artifact, records)
    hashes = sample_hashes(len(records))
    fraction = sorted(hashes)[5]
    kept = [i for i, value in enumerate(hashes) if value < fraction]
    output = tmp_path / "sample"
    assert main(["sample", str(run), str(output), "--fraction", str(fraction)]) == 0
    assert_selected(output, kept, tokens, weights)
    selection = json.loads((output / "run.json").read_text())["selection"]
    within = {"min_tokens": 1, "max_tokens": None}
    expected = {"fraction": fraction, "documents": None, "tokens": None, "seed": 0}
    assert selection["samples"] == [expected | {"within": within}]

    # A fraction of 1 keeps every document.
    assert main(["sample", str(run), str(tmp_path / "whole"), "--fraction", "1"]) == 0
    assert np.load(tmp_path / "whole" / "documents.npy").tolist() == list(range(len(records)))

    # A cap samples like the full run.
    limit = sorted(set(lengths))[-2]
    capped = tmp_path / "capped"
    assert main(["cap", str(run), str(capped), "--max-tokens", str(limit)]) == 0
    capped_sample = tmp_path / "capped-sample"
    assert main(["sample", str(capped), str(capped_sample), "--fraction", str(fraction)]) == 0
    in_cap = [i for i in kept if lengths[i] <= limit]
    assert np.load(capped_sample / "documents.npy").tolist() == in_cap

    # A sample splits like the full run.
    assert main(["split", str(run), str(tmp_path / "full"), *HALF]) == 0
    assert main(["split", str(output), str(tmp_path / "part"), *HALF]) == 0
    validation = sides(tmp_path / "full")[1]
    assert sides(tmp_path / "part")[1] == [i for i in validation if i in kept]


def test_sample_records_the_selection_it_was_drawn_from(tmp_path, artifact, records):
    run = tokenized(tmp_path, artifact, records)[1]
    first, capped, second = (tmp_path / name for name in ("first", "capped", "second"))
    assert main(["sample", str(run), str(first), "--documents", "5"]) == 0
    limit = 4096
    assert main(["cap", str(first), str(capped), "--max-tokens", str(limit)]) == 0
    assert main(["sample", str(capped), str(second), "--documents", "1", "--seed", "1"]) == 0
    selection = json.loads((second / "run.json").read_text())["selection"]
    unset = {"fraction": None, "tokens": None}
    assert selection["max_tokens"] == limit
    assert selection["samples"] == [
        {**unset, "documents": 5, "seed": 0, "within": {"min_tokens": 1, "max_tokens": None}},
        {**unset, "documents": 1, "seed": 1, "within": {"min_tokens": 1, "max_tokens": limit}},
    ]


def test_sample_rejects_bad_limits(tmp_path, artifact, records, capsys):
    run = tokenized(tmp_path, artifact, records)[1]
    command = ["sample", str(run), str(tmp_path / "out")]
    for args, error in [
        (["--fraction", "0"], "must be in (0, 1]"),
        (["--fraction", "1.5"], "must be in (0, 1]"),
        (["--documents", "0"], "must be positive"),
        (["--tokens", "0"], "must be positive"),
        (["--tokens", "1"], "the sample is empty"),
    ]:
        assert main([*command, *args]) == 2
        assert error in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([*command, "--fraction", "0.5", "--documents", "1"])
    with pytest.raises(ValueError, match="exactly one"):
        _selection.sample(run, tmp_path / "out", fraction=0.5, documents=1)
    assert not (tmp_path / "out").exists()
