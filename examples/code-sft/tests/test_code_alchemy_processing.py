from argparse import Namespace
import json

import pyarrow as pa
import pyarrow.parquet as pq

from process_code_alchemy import run, REVISION
from standardise_native import SCHEMA, checked_row
from standardize import SCHEMA as BENCH_SCHEMA, normalize


def test_all_subsets_native_pretraining_eval_holdout_and_existing_duplicates(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "_SUCCESS.json").write_text(
        json.dumps(dict(revision=REVISION, validated=True))
    )
    (source / "_MATERIALIZED.json").write_text("{}")
    pair = dict(text="User:\nQ\n\nAssistant:\nA", style="code_contests", blob_id="same")
    rows = {
        "code-qa": [
            pair,
            pair,
            dict(text="// complete code\nf(){}", style="multipl-e"),
        ],
        "code-dev": [
            dict(
                text_with_placeholders="User:\nExplain {{{REPLACE_WITH_BLOB_ID_SOURCE}}}\n\nAssistant:\nDev",
                source_code="x=1",
                source_code_status="hydrated",
            )
        ],
        "code-dialogue": [
            dict(text_with_placeholders="User:\nDialogue\n\nAssistant:\nResponse")
        ],
        "code-trace": [
            dict(text="User:\nTrace\n\nAssistant:\nOutput", blob_id="heldout")
        ],
        "code-enhance": [dict(text="def enhanced(): pass")],
    }
    for subset, data in rows.items():
        directory = source / subset / "train"
        directory.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(data), directory / "part-000.parquet")
    for subset in ["dev-eval", "trace-eval"]:
        directory = source / subset / "test"
        directory.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                [dict(prompt="Held out " + subset, blob_id="heldout")]
            ),
            directory / "part-000.parquet",
        )
    benchmarks = tmp_path / "benchmarks"
    directory = benchmarks / "org/eval"
    directory.mkdir(parents=True)
    (directory / "source.json").write_text("{}")
    (directory / "_SUCCESS.json").write_text('{"rows":1,"revision":"abc"}')
    benchmark = normalize(
        dict(prompt="External evaluation"),
        repo="org/eval",
        revision="abc",
        config="test",
        split="test",
        index=0,
        category="benchmarks",
    )
    pq.write_table(
        pa.Table.from_pylist([benchmark], schema=BENCH_SCHEMA),
        directory / "part-000.parquet",
    )
    lock = tmp_path / "benchmark-lock.json"
    lock.write_text('{"datasets":[{"repo":"org/eval","revision":"abc"}]}')
    reference = tmp_path / "reference"
    (reference / "datasets").mkdir(parents=True)
    entry = dict(
        repo="open-alchemy/code-alchemy",
        revision=REVISION,
        category="code_sft",
        license="unknown",
        license_status="unknown",
        license_notes="",
        synthetic_notes="",
        metadata_source="fixture",
    )
    row = checked_row(
        pair,
        entry,
        dict(path="reference", source_config="code-qa", source_split="train"),
        0,
        {},
        "think-first",
    )
    pq.write_table(
        pa.Table.from_pylist([row], schema=SCHEMA),
        reference / "datasets/native.parquet",
    )
    (reference / "_SUCCESS.json").write_text('{"rows":1}')
    args = Namespace(
        input=source,
        output=tmp_path / "output",
        benchmarks=benchmarks,
        benchmark_lock=lock,
        reference=reference,
        workers=2,
        threads=2,
        memory_limit="256MB",
        limit_files=None,
    )
    result = run(args)
    assert result["ready"]
    assert result["standardisation"] == dict(
        read=7, native=5, pretraining=2, quarantined=0
    )
    assert result["decisions"] == dict(keep=4, duplicate=2, benchmark=1)
    assert result["existing_reference_rows"] == 1
    assert (
        sum(
            pq.ParquetFile(p).metadata.num_rows
            for p in (args.output / "decontaminated/clean/datasets").rglob("*.parquet")
        )
        == 2
    )
    assert (
        sum(
            pq.ParquetFile(p).metadata.num_rows
            for p in (args.output / "decontaminated/clean/pretraining").rglob(
                "*.parquet"
            )
        )
        == 2
    )
    assert run(args) == result
