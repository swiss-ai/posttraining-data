"""Selections of a merged tokenize run: a token cap, length buckets, or a validation split."""

import fcntl
import hashlib
import math
from contextlib import contextmanager
from decimal import Decimal
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np

from . import _indexed, _inputs
from ._run import (
    CONFIG,
    DOCUMENTS,
    DTYPES,
    SUMMARY,
    merged_run,
    outputs,
    remove_stale_staging,
    run_lengths,
    source_documents,
    staged,
    tokenize_summary,
    totals,
    verify_inputs,
    write_json,
)


class _Parent:
    """A merged tokenize run (or a selection of one) to select documents from."""

    def __init__(self, run, output):
        self.run, self.config = merged_run(run)
        self.output = Path(output).resolve()
        if any(self.output.is_relative_to(p) for p in (self.run, self.config["input"])):
            raise ValueError("the output directory must be outside the run and its input")
        self.lengths = run_lengths(self.run, weights=self.config["loss_weights"])
        self.origin = source_documents(self.run, self.config, len(self.lengths))
        # "run" always names the original tokenize run.
        default = {"run": str(self.run), "min_tokens": 1, "max_tokens": None}
        self.selection = self.config.get("selection", default)

    @contextmanager
    def stage(self):
        """Lock the destination before cleaning up and publishing a selection."""
        self.output.parent.mkdir(parents=True, exist_ok=True)
        # Keep the lock file: unlinking it could let writers lock different inodes.
        with (self.output.parent / f".lock-{self.output.name}").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"another selection is writing {self.output}") from error
            if self.output.exists() and (not self.output.is_dir() or any(self.output.iterdir())):
                raise ValueError(f"{self.output} is not an empty directory")
            remove_stale_staging(self.output)
            with staged(self.output) as directory:
                yield directory

    def limits(self, low, high):
        """`low..high` (high None: unbounded) narrowed to the run's own range."""
        own = self.selection["max_tokens"]
        if high is None or (own is not None and own < high):
            high = own
        return {"min_tokens": max(low, self.selection["min_tokens"]), "max_tokens": high}

    def keep(self, limits):
        high = limits["max_tokens"] or np.inf
        return (self.lengths >= limits["min_tokens"]) & (self.lengths <= high)

    def write(self, directory, keep, selection, **extra):
        """Write the kept documents to `directory`, updating the recorded `selection`."""
        selection = self.selection | selection
        directory.mkdir(exist_ok=True)
        for name in outputs(self.config):
            _indexed.select(self.run / name, directory / name, DTYPES[name], self.lengths, keep)
        np.save(directory / DOCUMENTS, self.origin[keep])
        write_json(directory / CONFIG, self.config | {"selection": selection})
        summary = (
            tokenize_summary(self.config, self.lengths[keep]) | {"selection": selection} | extra
        )
        write_json(directory / SUMMARY, summary)
        return summary


def _empty(limits):
    return limits["max_tokens"] is not None and limits["min_tokens"] > limits["max_tokens"]


def cap(run, output, max_tokens):
    """Write the documents of at most `max_tokens` tokens to `output`, in run order."""
    if max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    parent = _Parent(run, output)
    limits = parent.limits(1, max_tokens)
    if _empty(limits):
        raise ValueError(f"--max-tokens is below the run's minimum of {limits['min_tokens']}")
    keep = parent.keep(limits)
    with parent.stage() as directory:
        return parent.write(directory, keep, limits, dropped=totals(parent.lengths[~keep]))


def bucket(run, output, bounds):
    """Write `output/<low>-<high>/` per range up to each bound and `output/<last+1>+/` above.

    A selection narrows the ranges to its own; ranges outside it are left out.
    """
    if not bounds or bounds[0] < 1 or any(a >= b for a, b in pairwise(bounds)):
        raise ValueError("--bounds must be positive and strictly increasing")
    parent = _Parent(run, output)
    ranges = [parent.limits(low + 1, high) for low, high in pairwise([0, *bounds, None])]
    buckets = []
    with parent.stage() as directory:
        for limits in ranges:
            if _empty(limits):
                continue
            low, high = limits["min_tokens"], limits["max_tokens"]
            name = f"{low}-{high}" if high is not None else f"{low}+"
            keep = parent.keep(limits)
            parent.write(directory / name, keep, limits)
            buckets.append({"name": name, **limits, **totals(parent.lengths[keep])})
        result = {
            "mode": "bucket",
            "run": str(parent.run),
            "bounds": bounds,
            **totals(parent.lengths),
            "buckets": buckets,
        }
        write_json(directory / SUMMARY, result)
    return result


def _hashed(value, seed):
    """A stable pseudo-random number in [0, 1) for `value`."""
    # Equal numeric group IDs must hash alike across Parquet schemas, without rounding.
    if isinstance(value, (bool, float, Decimal)):
        try:
            value = Fraction(value)
        except (ValueError, OverflowError) as error:
            raise ValueError("numeric group values must be finite") from error
        if value.denominator == 1:
            value = value.numerator
    digest = hashlib.blake2b(f"{seed}:{value!r}".encode(), digest_size=8).digest()
    return int.from_bytes(digest) / 2**64


def split(run, output, fraction, *, seed=0, group_by=None):
    """Write about a `fraction` of documents to `output/validation/`, the rest to `train/`.

    A seeded hash of each document's original index (or its source row's `group_by`
    value) picks the side, so every selection of one run splits the same way.
    """
    if not 0 < fraction < 1:
        raise ValueError("--validation must be between 0 and 1")
    parent = _Parent(run, output)
    keys = parent.origin.tolist()
    if group_by:
        verify_inputs(parent.config, tokenizer=False)
        values = _inputs.column_values(parent.config, group_by)
        keys = [values[i] for i in keys]
        if any(key is None or (isinstance(key, float) and math.isnan(key)) for key in keys):
            raise ValueError(f"column {group_by!r} has missing values")
    validation = np.array([_hashed(key, seed) < fraction for key in keys], bool)
    if validation.all() or not validation.any():
        raise ValueError("the split leaves train or validation empty; change --validation")
    settings = {"fraction": fraction, "seed": seed, "group_by": group_by}
    result = {"mode": "split", "run": str(parent.run), **settings}
    with parent.stage() as directory:
        for name, keep in (("train", ~validation), ("validation", validation)):
            splits = [*parent.selection.get("splits", []), settings | {"part": name}]
            parent.write(directory / name, keep, {"splits": splits})
            result[name] = totals(parent.lengths[keep])
        write_json(directory / SUMMARY, result)
    return result
