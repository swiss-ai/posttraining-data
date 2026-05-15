"""Shared normalization + hashing helpers for the per-dataset compute scripts.

Every per-dataset compute-<dataset>.py file imports from here and only has to
supply an iterator that yields (id, prompt_text, response_text). The schema of
the per-shard output parquet is identical across datasets so the same
aggregate-clusters.py can consume any of them.
"""
import re
from typing import Iterable, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

_IMG = re.compile(r"<image>\n?")
_AUDIO = re.compile(r"<audio>\n?")
_VIDEO = re.compile(r"<video>\n?")
_WS = re.compile(r"\s+")
_NUM = re.compile(r"-?\d+(?:[.,]\d+)*")
_BBOX = re.compile(r"\[\s*<N>(?:\s*,\s*<N>)+\s*\]")


def norm(text: str) -> str:
    """Lowercase + whitespace-collapse + strip media placeholders."""
    if not text:
        return ""
    t = _IMG.sub("", text)
    t = _AUDIO.sub("", t)
    t = _VIDEO.sub("", t)
    return _WS.sub(" ", t).strip().lower()


def skeleton(text: str) -> str:
    """Replace numeric runs and bbox-like coordinate lists with placeholders."""
    s = _NUM.sub("<N>", text)
    s = _BBOX.sub("<BBOX>", s)
    return s


def h64(text: str) -> int:
    return xxhash.xxh3_64_intdigest(text)


def first_human_assistant_fromvalue(convs) -> Tuple[str, str]:
    """ShareGPT-style: list of {from, value}."""
    human, assistant = "", ""
    for turn in convs:
        role = turn.get("from", "")
        if not human and role in ("human", "user"):
            human = turn.get("value", "")
        elif not assistant and role in ("gpt", "assistant", "bot"):
            assistant = turn.get("value", "")
        if human and assistant:
            break
    return human, assistant


def first_human_assistant_rolecontent(convs) -> Tuple[str, str]:
    """OpenAI-style: list of {role, content}."""
    human, assistant = "", ""
    for turn in convs:
        role = turn.get("role", "")
        if not human and role in ("user", "human"):
            human = turn.get("content", "")
        elif not assistant and role == "assistant":
            assistant = turn.get("content", "")
        if human and assistant:
            break
    return human, assistant


def write_shard(rows: Iterable[Tuple[str, str, str]], dst: str) -> int:
    """Materialize an iterator of (id, prompt_raw, response_raw) tuples to a
    per-shard hash parquet. Returns row count."""
    ids = []
    p_hash = []
    p_skel = []
    p_pref = []
    r_hash = []
    p_len = []
    r_len = []
    for rid, prompt_raw, response_raw in rows:
        prompt = norm(prompt_raw)
        response = norm(response_raw)
        toks = prompt.split()
        ids.append(str(rid))
        p_hash.append(h64(prompt))
        p_skel.append(h64(skeleton(prompt)))
        p_pref.append(h64(" ".join(toks[:8])))
        r_hash.append(h64(response))
        p_len.append(min(len(toks), 65535))
        r_len.append(min(len(response.split()), 65535))
    tbl = pa.table({
        "id": pa.array(ids, type=pa.string()),
        "prompt_hash": pa.array(p_hash, type=pa.uint64()),
        "prompt_skeleton_hash": pa.array(p_skel, type=pa.uint64()),
        "prompt_prefix8_hash": pa.array(p_pref, type=pa.uint64()),
        "response_hash": pa.array(r_hash, type=pa.uint64()),
        "prompt_token_count": pa.array(p_len, type=pa.uint16()),
        "response_token_count": pa.array(r_len, type=pa.uint16()),
    })
    pq.write_table(tbl, dst, compression="zstd")
    return tbl.num_rows
