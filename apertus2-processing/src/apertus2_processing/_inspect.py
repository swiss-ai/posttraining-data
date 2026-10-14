"""Read documents back out of a merged tokenize run: decoded text, tokens, weights, source."""

import dataclasses
import itertools
import random
from pathlib import Path

import numpy as np

from . import _html, _indexed, _inputs
from ._run import CONFIG, DTYPES, SUMMARY, read_json

REPORT = "report.html"


def _choose(count, indices, sample, seed):
    """The given document indices as given, else a sorted random sample."""
    if indices:
        bad = [i for i in indices if not 0 <= i < count]
        if bad:
            raise ValueError(f"document index out of range (0..{count - 1}): {bad}")
        return list(indices)
    if sample < 0:
        raise ValueError("--sample must not be negative")
    return sorted(random.Random(seed).sample(range(count), min(sample, count)))


def _source_json(config, index):
    """Return (path, row, raw JSON) of a document; documents follow shard order."""
    for shard in config["shards"]:
        if index < shard["end"] - shard["start"]:
            break
        index -= shard["end"] - shard["start"]
    source = config["sources"][shard["source"]]
    raw = next(itertools.islice(_inputs.rows(source, shard, config["column"]), index, None))
    return source["path"], shard["start"] + index, raw


def _verify_sources(config):
    """Refuse to show source rows unless the input files are exactly those tokenized."""
    missing = [s["path"] for s in config["sources"] if not Path(s["path"]).exists()]
    if missing:
        raise ValueError(f"source no longer exists: {', '.join(missing)}")
    root = config["input"]
    # The snapshot lists the tokenizer too, which may be nested under the input.
    recorded = {f["path"]: f for f in config["files"]}
    recorded = [
        f for f in recorded.values() if f["path"] == root or f["path"].startswith(root + "/")
    ]
    if _inputs.snapshot(root) != recorded:
        raise ValueError("source files changed since tokenization; rows no longer match")


def _default_weights(ids, controls):
    """The trainer's defaults: loss on output items (control tokens included) and waits.

    Mirrors the library's loss-weight defaults (README, "Loss weights") from the token
    stream alone; `test_default_weights_match_the_library` pins the two together.
    """
    weights, inside = [], False
    for token in ids:
        inside = inside or token == controls.output_start
        weights.append(1.0 if inside or token == controls.wait else 0.0)
        inside = inside and token != controls.output_end
    return weights


def documents(run, *, indices=(), sample=3, seed=None, tokenizer=None, source=False, weights=False):
    """Yield one record per selected document of a merged tokenize run.

    A record holds `index`, `total`, `ids`, `pieces` (each token decoded alone), `text`
    (the document decoded), `weights` (one float per token: the written `loss_weights`
    when `weights` is requested, else the trainer defaults), `weights_written`, `controls`
    (the control and BOS token IDs), `meta` (the header summary), and `source`
    (`{"path", "row", "json"}`, or None unless requested). `source` and `weights` are
    verified first: the inputs must be unchanged since tokenization, and the loss-weight
    index must agree with the tokens.
    """
    from apertus_common import load_encoding

    run = Path(run).resolve()
    if not ((run / CONFIG).exists() and (run / SUMMARY).exists()):
        raise ValueError(f"{run} is not a merged run")
    config = read_json(run / CONFIG)
    if config["mode"] != "tokenize":
        raise ValueError(f"{run} is a {config['mode']} run; inspect reads tokenize runs")
    if source:
        _verify_sources(config)
    try:
        encoding = load_encoding(tokenizer or config["tokenizer"])
    except Exception as error:
        raise ValueError(f"cannot load tokenizer: {error}") from error
    lengths = _indexed.read_lengths(run / "tokens", DTYPES["tokens"])
    if weights:
        if not config["loss_weights"]:
            raise ValueError("the run was tokenized without --loss-weights")
        weight_lengths = _indexed.read_lengths(run / "loss_weights", DTYPES["loss_weights"])
        if not np.array_equal(weight_lengths, lengths):
            raise ValueError("loss_weights index disagrees with the tokens index")
    controls = sorted({*dataclasses.astuple(encoding.controls), encoding.bos_token_id})
    for index in _choose(len(lengths), indices, sample, seed):
        ids = _indexed.read_document(run / "tokens", DTYPES["tokens"], lengths, index).tolist()
        values = _default_weights(ids, encoding.controls)
        if weights:
            values = _indexed.read_document(
                run / "loss_weights", DTYPES["loss_weights"], lengths, index
            )
            values = [round(float(w), 7) for w in values]  # float32 noise off the display
        record = {
            "index": index,
            "total": len(lengths),
            "ids": ids,
            "pieces": [encoding.decode_text([token]) for token in ids],
            "text": encoding.decode_text(ids),
            "weights": values,
            "weights_written": weights,
            "controls": controls,
            "source": None,
        }
        if source:
            path, row, raw = _source_json(config, index)
            record["source"] = {"path": path, "row": row, "json": raw}
        record["meta"] = _meta(record)
        yield record


def _meta(record):
    """Header summary: token count and which loss weights apply."""
    values = np.asarray(record["weights"])
    distinct = ", ".join(f"{w:g}" for w in np.unique(values))
    origin = "from loss_weights" if record["weights_written"] else "trainer defaults"
    return (
        f"{len(record['ids'])} tokens, {np.count_nonzero(values)} with loss "
        f"(weights {distinct}, {origin})"
    )


def render(record, *, tokens=False):
    """Format a record as the printable block: header, text or token table, source."""
    lines = [f"=== document {record['index']} (of {record['total']}): {record['meta']}"]
    if tokens:
        columns = zip(record["ids"], record["weights"], record["pieces"], strict=True)
        for position, (token, weight, piece) in enumerate(columns):
            lines.append(f"{position:6d} {token:7d} {weight:>4g} {piece!r}")
    else:
        lines.append(record["text"])
    if record["source"]:
        lines += [
            f"--- source {record['source']['path']} row {record['source']['row']}",
            record["source"]["json"],
        ]
    return "\n".join(lines)


def write_report(records, directory, *, run, tokens=False):
    """Write one text file per document plus a standalone HTML report.

    Returns (document paths, report path). Earlier `document-*.txt` files in the
    directory are removed so the report and the files always describe the same set.
    """
    records = list(records)  # validated and rendered twice: text files and the HTML page
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("document-*.txt"):
        stale.unlink()
    paths = []
    for record in records:
        path = directory / f"document-{record['index']:06d}.txt"
        path.write_text(render(record, tokens=tokens) + "\n", encoding="utf-8")
        paths.append(path)
    report = directory / REPORT
    report.write_text(_html.report(records, run=str(run)), encoding="utf-8")
    return paths, report
