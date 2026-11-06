"""Rust MinHash candidate generation with exact shingle-Jaccard verification.

Training LSH bands are joined on disk by DuckDB. Only candidate rows are reread
for verification; original examples and all-pairs distance matrices stay off RAM.
"""

import hashlib
import json
import re
import struct
import concurrent.futures
import multiprocessing
from collections import deque
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rensa import RMinHash, RMinHashLSH

NUM_PERM = 128
BANDS = 16
SHINGLE_SIZE = 5
MIN_TOKENS = 30
TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
BENCH_INDEX = None
BENCH_SETS = {}
BENCH_REFS = {}
BENCH_THRESHOLD = 0.8


def shingles(text):
    tokens = TOKEN.findall(text)
    if len(tokens) < MIN_TOKENS:
        return set()
    return {
        "\x1f".join(tokens[i : i + SHINGLE_SIZE])
        for i in range(len(tokens) - SHINGLE_SIZE + 1)
    }


def sketch(values):
    result = RMinHash(num_perm=NUM_PERM, seed=42)
    result.update(sorted(values))
    return result


def bands(values):
    if not values:
        return []
    digest = sketch(values).digest()
    return [
        hashlib.blake2b(struct.pack("<8I", *digest[i : i + 8]), digest_size=8).digest()
        for i in range(0, NUM_PERM, 8)
    ]


def jaccard(left, right):
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)


def views(row):
    from dedup import compact, messages, string_leaves, parsed

    turns = messages(row)
    content, assistant = [], []
    roles = []
    message_metadata = []
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, dict):
                roles.append(turn.get("role"))
                message_metadata.append(
                    {k: v for k, v in turn.items() if k != "content"}
                )
                text = "\n".join(string_leaves(turn.get("content")))
                content.append(str(turn.get("role")) + "\n" + text)
                if turn.get("role") == "assistant":
                    assistant.append(text)
    else:
        content.extend(string_leaves(turns))
    for key in ("prompt", "response", "reasoning"):
        if row.get(key):
            content.append(row[key])
    if row.get("response"):
        assistant.append(row["response"])
    guard = hashlib.sha256(
        compact(
            {
                "roles": roles,
                "message_metadata": message_metadata,
                "tools": parsed(row.get("tools_json")),
                "tests": parsed(row.get("tests_json")),
            }
        ).encode()
    ).hexdigest()
    return "\n".join(content), "\n".join(assistant), guard


def signature(row):
    text, _, guard = views(row)
    values = shingles(text)
    return bands(values), len(values), guard


def build_benchmarks(exact, threshold):
    global BENCH_INDEX, BENCH_SETS, BENCH_REFS, BENCH_THRESHOLD
    BENCH_THRESHOLD = threshold
    BENCH_INDEX = RMinHashLSH(
        threshold=max(0.1, threshold - 0.1), num_perm=NUM_PERM, num_bands=32
    )
    BENCH_SETS, BENCH_REFS = {}, {}
    for text, reference in sorted(exact.items()):
        values = shingles(text)
        if not values:
            continue
        key = len(BENCH_SETS)
        BENCH_INDEX.insert(key, sketch(values))
        BENCH_SETS[key], BENCH_REFS[key] = values, reference
    return len(BENCH_SETS)


def benchmark_match(row):
    from dedup import messages, string_leaves

    texts = [row.get("prompt"), row.get("response"), row.get("reasoning")]
    turns = messages(row)
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, dict):
                texts.extend(string_leaves(turn.get("content")))
    seen = set()
    for text in texts:
        if not isinstance(text, str) or text in seen:
            continue
        seen.add(text)
        values = shingles(text)
        if not values:
            continue
        for key in sorted(BENCH_INDEX.query(sketch(values))):
            other = BENCH_SETS[key]
            if (
                min(len(values), len(other)) / max(len(values), len(other))
                < BENCH_THRESHOLD
            ):
                continue
            score = jaccard(values, other)
            if score >= BENCH_THRESHOLD:
                return BENCH_REFS[key], "fuzzy_prompt", score
    return None, None, None


def verify_pair(left, right, threshold):
    lt, la, lg = views(left)
    rt, ra, rg = views(right)
    if lg != rg:
        return None
    score = jaccard(shingles(lt), shingles(rt))
    if score < threshold:
        return None
    # A shared long prompt must not mask a substantially different answer.
    if la != ra:
        ls, rs = shingles(la), shingles(ra)
        if not ls or not rs or jaccard(ls, rs) < threshold:
            return None
    return score


def verify_batch(task):
    pairs, threshold = task
    matches = []
    for a, b, left, right in pairs:
        score = verify_pair(json.loads(left), json.loads(right), threshold)
        if score is not None:
            matches.append((a, b, score))
    return matches


def ordered_verification(reader, threshold, workers, count):
    def tasks():
        for batch in reader:
            yield (
                list(zip(*(batch.column(i).to_pylist() for i in range(4)))),
                threshold,
            )

    if workers == 1 or count < 1024:
        yield from map(verify_batch, tasks())
        return
    # Spawn avoids inheriting DuckDB's live native threads. Keep in-flight text bounded.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = deque()
        for task in tasks():
            pending.append(pool.submit(verify_batch, task))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def apply_training(db, files, output, threshold, max_candidates=50_000_000, workers=1):
    from dedup import sql_string

    db.execute("""CREATE OR REPLACE TABLE fuzzy_nodes AS
      SELECT row_number() OVER (ORDER BY file_index, row_index) AS node,
        file_index, row_index, fuzzy_bands, fuzzy_size, fuzzy_guard
      FROM decisions WHERE duplicate_rank=1 AND fuzzy_size > 0""")
    db.execute("CREATE OR REPLACE TABLE fuzzy_candidates (a BIGINT, b BIGINT)")
    total = 0
    for band in range(1, BANDS + 1):
        # Sort/hash joins spill according to the main DuckDB memory limit.
        db.execute(f"""CREATE OR REPLACE TEMP TABLE buckets AS
          SELECT node, fuzzy_size, fuzzy_guard, fuzzy_bands[{band}] AS band FROM fuzzy_nodes""")
        estimate = db.execute("""SELECT coalesce(sum(n*(n-1)/2), 0) FROM
          (SELECT count(*) AS n FROM buckets GROUP BY band, fuzzy_guard)""").fetchone()[
            0
        ]
        if total + estimate > max_candidates:
            raise ValueError(
                f"LSH candidate budget exceeded at band {band}: {total + estimate:,.0f}. "
                "Increase --max-fuzzy-candidates after reviewing the data; no candidates silently dropped."
            )
        db.execute(
            """INSERT INTO fuzzy_candidates SELECT a.node, b.node FROM buckets a JOIN buckets b
          ON a.band=b.band AND a.fuzzy_guard=b.fuzzy_guard AND a.node<b.node
          WHERE least(a.fuzzy_size,b.fuzzy_size)>=?*greatest(a.fuzzy_size,b.fuzzy_size)""",
            [threshold],
        )
        total += estimate
    db.execute(
        "CREATE OR REPLACE TABLE fuzzy_pairs AS SELECT DISTINCT a,b FROM fuzzy_candidates"
    )
    db.execute("DROP TABLE fuzzy_candidates")
    count = db.execute("SELECT count(*) FROM fuzzy_pairs").fetchone()[0]
    print(f"FUZZY {count} candidate pairs to verify", flush=True)
    # Candidate examples are staged on disk, not kept in a global Python object map.
    db.execute("""CREATE OR REPLACE TABLE fuzzy_needed AS SELECT n.node,n.file_index,n.row_index FROM fuzzy_nodes n
      JOIN (SELECT a AS node FROM fuzzy_pairs UNION SELECT b FROM fuzzy_pairs) p USING (node)""")
    db.execute(
        "COPY fuzzy_needed TO "
        + sql_string(output / "_work" / "fuzzy_needed")
        + " (FORMAT PARQUET, PARTITION_BY(file_index), OVERWRITE_OR_IGNORE)"
    )
    db.execute("CREATE OR REPLACE TABLE fuzzy_text (node BIGINT, payload VARCHAR)")
    for directory in sorted((output / "_work" / "fuzzy_needed").glob("file_index=*")):
        file_index = int(directory.name.split("=")[1])
        wanted = {}
        for mask in directory.glob("*.parquet"):
            for r in pq.ParquetFile(mask).read().to_pylist():
                wanted[r["row_index"]] = r["node"]
        offset = 0
        source = pq.ParquetFile(files[file_index])
        columns = [
            c
            for c in [
                "messages_json",
                "prompt",
                "response",
                "reasoning",
                "tools_json",
                "tests_json",
            ]
            if c in source.schema_arrow.names
        ]
        for batch in source.iter_batches(batch_size=256, columns=columns):
            selected = [i for i in range(batch.num_rows) if offset + i in wanted]
            if selected:
                rows = (
                    pa.Table.from_batches([batch])
                    .take(pa.array(selected, type=pa.int64()))
                    .to_pylist()
                )
                values = pa.table(
                    {
                        "node": [wanted[offset + i] for i in selected],
                        "payload": [json.dumps(r, ensure_ascii=False) for r in rows],
                    }
                )
                db.register("candidate_batch", values)
                db.execute("INSERT INTO fuzzy_text SELECT * FROM candidate_batch")
                db.unregister("candidate_batch")
            offset += batch.num_rows
    # Greedy deterministic representatives: every discarded row matches a surviving exact representative.
    # No transitive connected-component chaining across weakly similar endpoints.
    removed = set()
    witnesses = []
    db.execute(
        "CREATE OR REPLACE TABLE fuzzy_matches (a BIGINT,b BIGINT,similarity DOUBLE)"
    )
    cursor = db.cursor()
    reader = cursor.execute("""SELECT p.a,p.b,l.payload,r.payload FROM fuzzy_pairs p
      JOIN fuzzy_text l ON l.node=p.a JOIN fuzzy_text r ON r.node=p.b ORDER BY p.b,p.a""").fetch_record_batch(
        256
    )
    for matches in ordered_verification(reader, threshold, workers, count):
        for a, b, score in matches:
            if a in removed or b in removed:
                continue
            removed.add(b)
            witnesses.append((a, b, score))
        if witnesses:
            table = pa.table(
                {
                    "a": [w[0] for w in witnesses],
                    "b": [w[1] for w in witnesses],
                    "similarity": [w[2] for w in witnesses],
                }
            )
            db.register("witness_batch", table)
            db.execute("INSERT INTO fuzzy_matches SELECT * FROM witness_batch")
            db.unregister("witness_batch")
            witnesses.clear()
    cursor.close()
    db.execute("ALTER TABLE decisions ADD COLUMN fuzzy_similarity DOUBLE")
    db.execute("""UPDATE decisions d SET decision='fuzzy_duplicate',keeper_file_index=a.file_index,
      keeper_row_index=a.row_index,keeper_id=k.id,fuzzy_similarity=m.similarity
      FROM fuzzy_matches m JOIN fuzzy_nodes a ON a.node=m.a JOIN fuzzy_nodes b ON b.node=m.b
      JOIN decisions k ON k.file_index=a.file_index AND k.row_index=a.row_index
      WHERE d.file_index=b.file_index AND d.row_index=b.row_index""")
    return {"candidate_pairs": count, "removed": len(removed)}
