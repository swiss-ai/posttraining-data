"""Aggregate per-shard prompt-hash parquets into cluster tables.

Inputs: directory of per-shard parquets produced by compute-prompt-hashes.py.
Outputs (alongside): one cluster parquet per hash level + summary.json.

Usage:
    python aggregate-clusters.py <shards_dir> <output_dir>
"""

import json
import sys
from pathlib import Path

import polars as pl


HASH_LEVELS = [
    ("prompt_hash", "clusters_prompt.parquet"),
    ("prompt_skeleton_hash", "clusters_prompt_skeleton.parquet"),
    ("prompt_prefix8_hash", "clusters_prompt_prefix8.parquet"),
]


def main(shards_dir: str, out_dir: str) -> None:
    shards_path = Path(shards_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_parquet(str(shards_path / "*.parquet"))
    total_rows = lf.select(pl.len()).collect().item()
    print(f"loaded {total_rows} rows from {shards_path}/*.parquet")

    summary = {
        "shards_dir": str(shards_path),
        "total_rows": total_rows,
        "levels": {},
    }

    for col, fname in HASH_LEVELS:
        clusters = (
            lf.group_by(col)
            .agg([
                pl.len().alias("count"),
                pl.first("id").alias("representative_id"),
                pl.n_unique("response_hash").alias("unique_responses"),
                pl.median("prompt_token_count").alias("prompt_token_count_median"),
            ])
            .sort("count", descending=True)
            .collect(engine="streaming")
        )
        clusters.write_parquet(out_path / fname, compression="zstd")
        n_clusters = clusters.height
        top = clusters.head(15)
        print(f"\n=== {col}: {n_clusters} unique clusters ({total_rows / n_clusters:.1f}x dedup) ===")
        for row in top.iter_rows(named=True):
            print(
                f"  count={row['count']:8d}  uniq_resp={row['unique_responses']:6d}  "
                f"med_tok={int(row['prompt_token_count_median']):4d}  rep={row['representative_id']}"
            )
        summary["levels"][col] = {
            "unique_clusters": n_clusters,
            "dedup_factor": total_rows / n_clusters,
            "top_10": [
                {
                    "count": int(r["count"]),
                    "unique_responses": int(r["unique_responses"]),
                    "representative_id": r["representative_id"],
                }
                for r in top.head(10).iter_rows(named=True)
            ],
        }

    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary written to {out_path / 'summary.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: aggregate-clusters.py <shards_dir> <output_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
