"""Read documents back out of a merged tokenize run: decoded text, tokens, weights, source."""

import itertools
import random
from pathlib import Path

import numpy as np

from . import _indexed, _inputs
from ._run import CONFIG, DTYPES, SUMMARY, read_json


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


def _weights_summary(weights):
    distinct = ", ".join(f"{w:g}" for w in np.unique(weights))
    return f", {np.count_nonzero(weights)} with loss (weights {distinct})"


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


def render(
    run,
    *,
    indices=(),
    sample=3,
    seed=None,
    tokenizer=None,
    source=False,
    tokens=False,
    weights=False,
):
    """Yield one printable block per selected document of a merged tokenize run.

    `source` and `weights` are verified before use: the inputs must be unchanged since
    tokenization, and the loss-weight index must agree with the token index.
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
    for index in _choose(len(lengths), indices, sample, seed):
        ids = _indexed.read_document(run / "tokens", DTYPES["tokens"], lengths, index).tolist()
        head = f"=== document {index} (of {len(lengths)}): {len(ids)} tokens"
        columns = [range(len(ids)), ids]
        if weights:
            values = _indexed.read_document(
                run / "loss_weights", DTYPES["loss_weights"], lengths, index
            )
            head += _weights_summary(values)
            columns.append(f"{w:g}" for w in values)
        lines = [head]
        if tokens:
            for position, token, *weight in zip(*columns, strict=True):
                cells = [f"{position:6d}", f"{token:7d}", *(f"{w:>4}" for w in weight)]
                lines.append(" ".join(cells) + f" {encoding.decode_text([token])!r}")
        else:
            lines.append(encoding.decode_text(ids))
        if source:
            path, row, raw = _source_json(config, index)
            lines += [f"--- source {path} row {row}", raw]
        yield "\n".join(lines)
