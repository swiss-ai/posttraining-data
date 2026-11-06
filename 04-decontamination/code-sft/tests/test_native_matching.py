import json
import pyarrow as pa
import pyarrow.parquet as pq
import dedup


def test_native_think_payload_is_matched_without_json_escaping(monkeypatch):
    prompt = "A sufficiently distinctive\nreasoning-only benchmark reference"
    monkeypatch.setattr(dedup, "EXACT", {dedup.match_text(prompt): "witness"})
    monkeypatch.setattr(dedup, "MATCHER", None)
    row = dict(
        conversation_json=json.dumps(dict(items=[dict(type="think", payload=prompt)]))
    )
    assert dedup.contamination(row) == ("witness", "exact_text")


def test_benchmark_witness_on_duplicate_blocks_the_whole_exact_group(tmp_path):
    (tmp_path / "_work").mkdir()
    records = [
        dict(
            file_index=0,
            row_index=i,
            id=str(i),
            source_dataset="source",
            sample_hash="same",
            benchmark_match=None if i == 0 else "heldout-blob",
            match_kind=None,
            benchmark_similarity=None,
            fuzzy_bands=None,
            fuzzy_size=0,
            fuzzy_guard=None,
        )
        for i in range(2)
    ]
    path = tmp_path / "fingerprints.parquet"
    pq.write_table(pa.Table.from_pylist(records, schema=dedup.FP_SCHEMA), path)
    _, totals = dedup.classify([str(path)], tmp_path, 2, "128MB")
    assert totals == dict(benchmark=1, duplicate=1)
