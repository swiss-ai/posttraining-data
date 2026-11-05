"""Restore the questions of nvidia/OpenCodeReasoning-2 from the datasets they came from.

The release leaves `question` as "-" in every row and documents how to rebuild it: look up
row `index` of split `split` in the source named by `dataset` (TACO, APPS, CodeContests or
open-r1/codeforces) and take its problem statement, composed for open-r1/codeforces from the
description, formats, examples and note. This module does that once and writes a small
lookup table, one row per (dataset, split, index) the release refers to.

Only training splits are looked up. Rows from evaluation splits (APPS test, TACO test,
CodeContests valid/test) are rejected later by the preprocessor, since those questions are
benchmark problems.

Each lookup is checked against what the release says about the question (its platform and
difficulty label), so a source whose row order differs from the one the release used is
noticed instead of producing wrong questions. The build stops if a source disagrees.

Usage:
  python ocr2_questions.py --sources DIR --release DIR --output FILE.parquet
where DIR holds TACO/, apps/, code_contests/ and codeforces/ as downloaded by
01-hf-download/hf-download-raw.py (training files only).
"""

import argparse
import ast
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TRAIN_SPLITS = {
    ("taco", "train"),
    ("apps", "train"),
    ("code_contests", "train"),
    ("open-r1/codeforces", "train"),
}
EVALUATION_SPLITS = {
    ("taco", "test"),
    ("apps", "test"),
    ("code_contests", "valid"),
    ("code_contests", "test"),
    ("open-r1/codeforces", "test"),
}
CODE_CONTESTS_SOURCES = {
    1: "codechef", 2: "codeforces", 3: "hackerearth", 4: "codejam", 5: "atcoder", 6: "aizu",
}
# Below this share of agreeing labels the source order is taken to differ from the release
MIN_AGREEMENT = 0.98


def parquet_rows(pattern, columns):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    rows = []
    for path in files:
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return rows, files


def codeforces_question(problem):
    """The card's composition of an open-r1/codeforces statement."""
    if not problem["description"]:
        return None
    question = problem["description"]
    if problem["input_format"]:
        question += "\n\nInput\n\n" + problem["input_format"]
    if problem["output_format"]:
        question += "\n\nOutput\n\n" + problem["output_format"]
    examples = problem["examples"]
    if isinstance(examples, str):
        examples = ast.literal_eval(examples)
    if examples:
        question += "\n\nExamples"
        for example in examples:
            if "input" in example:
                question += "\n\nInput\n\n" + example["input"]
            if "output" in example:
                question += "\n\nOutput\n\n" + example["output"]
    if problem["note"]:
        question += "\n\nNote\n\n" + problem["note"]
    return question


def load_sources(directory):
    """{dataset: (rows, question(row), agrees(row, source, difficulty), files)}"""
    d = Path(directory)
    taco, taco_files = parquet_rows(str(d / "TACO/ALL/train-*.parquet"), ["question", "source", "difficulty"])
    apps_path = d / "apps/train.jsonl"
    with open(apps_path) as handle:
        apps = [json.loads(line) for line in handle]
    contests, contest_files = parquet_rows(
        str(d / "code_contests/data/train-*.parquet"), ["description", "source", "difficulty", "cf_rating"]
    )
    codeforces, codeforces_files = parquet_rows(
        str(d / "codeforces/data/train-*.parquet"),
        ["description", "input_format", "output_format", "examples", "note", "rating"],
    )
    return {
        "taco": (
            taco,
            lambda r: r["question"],
            lambda r, source, difficulty: r["source"] == source and r["difficulty"] == difficulty,
            taco_files,
        ),
        "apps": (
            apps,
            lambda r: r["question"],
            lambda r, source, difficulty: r.get("difficulty") == difficulty,
            [str(apps_path)],
        ),
        "code_contests": (
            contests,
            lambda r: r["description"] or None,
            lambda r, source, difficulty: CODE_CONTESTS_SOURCES.get(r["source"]) == source,
            contest_files,
        ),
        "open-r1/codeforces": (
            codeforces,
            codeforces_question,
            lambda r, source, difficulty: str(r["rating"]) == difficulty,
            codeforces_files,
        ),
    }


def referenced(release):
    """(dataset, split, index) -> (source, difficulty) for every row of the release."""
    from datasets import load_from_disk

    wanted = {}
    for split in load_from_disk(str(release)).values():
        for row in split.select_columns(["dataset", "split", "index", "source", "difficulty"]):
            key = (row["dataset"], row["split"], int(row["index"]))
            wanted.setdefault(key, (row["source"], row["difficulty"]))
    return wanted


def build(sources, release, output):
    wanted = referenced(release)
    loaded = load_sources(sources)
    records, counts = [], Counter()
    for (dataset, split, index), (source, difficulty) in sorted(wanted.items()):
        if (dataset, split) in EVALUATION_SPLITS:
            counts["evaluation split, not looked up"] += 1
            continue
        if (dataset, split) not in TRAIN_SPLITS:
            raise ValueError(f"Unknown OpenCodeReasoning-2 source split: {dataset} {split}")
        rows, question_of, agrees, _ = loaded[dataset]
        if index >= len(rows):
            raise ValueError(f"{dataset} {split} has no row {index} ({len(rows)} rows)")
        question = question_of(rows[index])
        if not question:
            raise ValueError(f"{dataset} {split} row {index} has no problem statement")
        agreed = bool(agrees(rows[index], source, difficulty))
        counts[(dataset, "labels agree" if agreed else "labels differ")] += 1
        records.append(dict(dataset=dataset, split=split, index=index, question=question, labels_agree=agreed))
    report = {}
    for dataset in loaded:
        agree, differ = counts[(dataset, "labels agree")], counts[(dataset, "labels differ")]
        if agree + differ:
            share = agree / (agree + differ)
            report[dataset] = dict(questions=agree + differ, labels_agree=round(share, 4))
            if share < MIN_AGREEMENT:
                raise ValueError(
                    f"{dataset}: only {share:.1%} of the looked-up rows match the release's labels; "
                    "the source row order differs from the one the release used"
                )
    revisions = {
        name: json.loads((Path(sources) / folder / "dataset_metadata.json").read_text()).get("commit_hash")
        for name, folder in (("taco", "TACO"), ("apps", "apps"), ("code_contests", "code_contests"),
                             ("open-r1/codeforces", "codeforces"))
    }
    metadata = {b"ocr2_question_sources": json.dumps(revisions).encode(), b"ocr2_question_report": json.dumps(report).encode()}
    table = pa.Table.from_pylist(records).replace_schema_metadata(metadata)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd")
    return dict(report=report, revisions=revisions, skipped=counts["evaluation split, not looked up"], written=len(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--release", required=True, help="the downloaded nvidia/OpenCodeReasoning-2 folder")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.sources, args.release, args.output), indent=2))
    except ValueError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
