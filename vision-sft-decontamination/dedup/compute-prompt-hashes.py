"""Per-shard prompt/response hashing for vision-SFT verbatim+template dedup.

Reads one parquet shard (Innovator-VL / ShareGPT-style schema: `id`,
`conversations[{from, value}]`) column-pruned (image bytes never touched).
Emits a tiny parquet with one row per training conversation, carrying the
three prompt-hash views described in vision-sft-decontam/dedup/README.md.

Usage:
    python compute-prompt-hashes.py <src.parquet> <dst.parquet>
"""

import re
import sys
from typing import Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

_IMG = re.compile(r"<image>\n?")
_WS = re.compile(r"\s+")
_NUM = re.compile(r"-?\d+(?:[.,]\d+)*")
_BBOX = re.compile(r"\[\s*<N>(?:\s*,\s*<N>)+\s*\]")


def _norm(text: str) -> str:
    text = _IMG.sub("", text)
    text = _WS.sub(" ", text).strip().lower()
    return text


def _skeleton(text: str) -> str:
    s = _NUM.sub("<N>", text)
    s = _BBOX.sub("<BBOX>", s)
    return s


def _h64(text: str) -> int:
    return xxhash.xxh3_64_intdigest(text)


def _first_human_assistant(convs) -> Tuple[str, str]:
    human, assistant = "", ""
    for turn in convs:
        role = turn["from"]
        if not human and role in ("human", "user"):
            human = turn["value"]
        elif not assistant and role in ("gpt", "assistant", "bot"):
            assistant = turn["value"]
        if human and assistant:
            break
    return human, assistant


def process(src: str, dst: str) -> None:
    tbl = pq.read_table(src, columns=["id", "conversations"])
    ids = tbl.column("id").to_pylist()
    convs_col = tbl.column("conversations").to_pylist()

    n = len(ids)
    out_id = []
    prompt_hash = [0] * n
    prompt_skel = [0] * n
    prompt_pref = [0] * n
    response_hash = [0] * n
    p_len = [0] * n
    r_len = [0] * n

    for i, (rid, convs) in enumerate(zip(ids, convs_col)):
        if not convs:
            continue
        prompt_raw, resp_raw = _first_human_assistant(convs)
        prompt = _norm(prompt_raw)
        resp = _norm(resp_raw)
        skel = _skeleton(prompt)
        tokens = prompt.split()
        pref8 = " ".join(tokens[:8])
        out_id.append(rid)
        prompt_hash[i] = _h64(prompt)
        prompt_skel[i] = _h64(skel)
        prompt_pref[i] = _h64(pref8)
        response_hash[i] = _h64(resp)
        p_len[i] = min(len(tokens), 65535)
        r_len[i] = min(len(resp.split()), 65535)

    out_tbl = pa.table({
        "id": pa.array(ids, type=pa.string()),
        "prompt_hash": pa.array(prompt_hash, type=pa.uint64()),
        "prompt_skeleton_hash": pa.array(prompt_skel, type=pa.uint64()),
        "prompt_prefix8_hash": pa.array(prompt_pref, type=pa.uint64()),
        "response_hash": pa.array(response_hash, type=pa.uint64()),
        "prompt_token_count": pa.array(p_len, type=pa.uint16()),
        "response_token_count": pa.array(r_len, type=pa.uint16()),
    })
    pq.write_table(out_tbl, dst, compression="zstd")
    print(f"{src} -> {dst} ({out_tbl.num_rows} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: compute-prompt-hashes.py <src.parquet> <dst.parquet>", file=sys.stderr)
        sys.exit(2)
    process(sys.argv[1], sys.argv[2])
