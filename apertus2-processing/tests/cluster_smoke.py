"""Real Slurm integration assertions, invoked by slurm/test.sh on compute nodes."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from apertus_common import Conversation, Reply, SystemPrompt, User, Wait, load_encoding
from datasets import Dataset, DatasetDict

from apertus2_processing import _indexed
from apertus2_processing.report import issue_rows

ROOT = Path(__file__).resolve().parents[1]


def records():
    examples = [
        json.loads((ROOT / "examples" / name).read_text())
        for name in ("native.json", "weighted.json")
    ]
    return [
        json.dumps(examples[i % 2])
        if i % 3 == 0
        else Conversation(
            system=SystemPrompt.build(),
            items=[
                User(payload=f"Question {i}: Grüezi, 世界 🙂"),
                Reply(payload=(f"Answer {i}. " * (i % 7 + 1))),
                Wait(),
            ],
        ).to_json()
        for i in range(1024)
    ]


def shard_state(run):
    return {
        str(path.relative_to(run)): {
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted((run / "tokens" / "shards").glob("*/*"))
        if path.is_file()
    }


def prepare(run):
    rows = records()
    (run / "parquet" / "nested").mkdir(parents=True)
    for name, batch in (("a.parquet", rows[:512]), ("nested/b.parquet", rows[512:])):
        pq.write_table(
            pa.table({"conversation_json": batch}), run / "parquet" / name, row_group_size=16
        )
    DatasetDict({"validation": Dataset.from_dict({"native": rows})}).save_to_disk(run / "hf")
    unfinished = json.loads(rows[1])
    unfinished["items"].pop()
    DatasetDict(
        {"validation": Dataset.from_dict({"native": [rows[1], "not json", json.dumps(unfinished)]})}
    ).save_to_disk(run / "bad")


def interrupt(run):
    """Simulate a lost shard and abandoned temporary output; keep completed shards."""
    state = shard_state(run)
    (run / "before-resume.json").write_text(json.dumps(state))
    shutil.rmtree(run / "tokens" / "shards" / "000003")
    (run / "tokens" / "shards" / ".tmp-000003-interrupted").mkdir()


def resumed(run):
    before = json.loads((run / "before-resume.json").read_text())
    after = shard_state(run)
    assert before.keys() == after.keys()
    for path, original in before.items():
        assert after[path]["sha256"] == original["sha256"], path
        if "/000003/" not in path:
            assert after[path] == original, f"completed shard rewritten: {path}"
    assert not (run / "tokens" / "shards" / ".tmp-000003-interrupted").exists()
    done = [line for line in (run / "resume.log").read_text().splitlines() if " done: " in line]
    assert len(done) == 1 and "shard 4/32 done: 32 rows" in done[0], done
    (run / "before-noop.json").write_text(json.dumps(after))


def verify(run):
    assert shard_state(run) == json.loads((run / "before-noop.json").read_text())
    assert " done: " not in (run / "noop.log").read_text()
    nodes = {line.split()[-1] for line in (run / "nodes.txt").read_text().splitlines()}
    assert len(nodes) == 2 and all(node.startswith("nid") for node in nodes), nodes
    summary = json.loads((run / "check" / "summary.json").read_text())
    assert summary["rows"] == 1024 and summary["failed_rows"] == 0, summary
    bad = json.loads((run / "bad-check" / "summary.json").read_text())
    assert (bad["rows"], bad["failed_rows"], bad["rows_with_issues"]) == (3, 1, 2), bad
    assert list(issue_rows(run / "bad-check", failed_only=True).values()) == [[1]]
    assert list(issue_rows(run / "bad-check").values()) == [[1, 2]]
    encoding = load_encoding(os.environ["APERTUS_ENCODING"])
    expected = [
        encoding.encode_conversation(Conversation.from_json(row), return_loss_weights=True)
        for row in records()
    ]
    # Compare every document, independently of sharding or batch encoding.
    for directory in ("tokens", "hf-tokens"):
        outputs = [("tokens", np.int32, [r.token_ids for r in expected])]
        if directory == "tokens":
            outputs.append(("loss_weights", np.float32, [r.loss_weights for r in expected]))
        for name, dtype, docs in outputs:
            prefix = run / directory / name
            lengths = _indexed.read_lengths(prefix, dtype)
            np.testing.assert_array_equal(lengths, [len(doc) for doc in docs])
            np.testing.assert_array_equal(
                np.fromfile(f"{prefix}.bin", dtype),
                np.concatenate([np.asarray(doc, dtype) for doc in docs]),
            )
        result = json.loads((run / directory / "summary.json").read_text())
        assert result["documents"] == len(expected)
        assert result["tokens"] == sum(len(r.token_ids) for r in expected)
    for extension in ("bin", "idx"):
        assert (run / "tokens" / f"tokens.{extension}").read_bytes() == (
            run / "hf-tokens" / f"tokens.{extension}"
        ).read_bytes()
    (run / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "nodes": sorted(nodes),
                "documents": len(expected),
                "tokens": result["tokens"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    action, directory = sys.argv[1:]
    {"prepare": prepare, "interrupt": interrupt, "resumed": resumed, "verify": verify}[action](
        Path(directory)
    )
