"""Restore left-out questions of NVIDIA's competitive-programming releases from their sources.

nvidia/OpenCodeReasoning-2 leaves `question` as "-" in every row, and
nvidia/Nemotron-Competitive-Programming-v1 does the same for its TACO and APPS rows. Both refer
to the problem by `dataset`, `split` and `index`, and the OpenCodeReasoning-2 card documents how
to rebuild it: look up row `index` of split `split` in the source named by `dataset` (TACO,
APPS, CodeContests or open-r1/codeforces) and take its problem statement, composed for
open-r1/codeforces from the description, formats, examples and note. This module writes that
statement for every training problem of the four sources into one lookup table.

Only training splits are looked up. Rows from evaluation splits (APPS test, TACO test,
CodeContests valid/test) are rejected later by the preprocessor, since those questions are
benchmark problems.

The rows the releases refer to are checked against what each release says about the question
(its platform and difficulty label), so a source whose row order differs from the one the
release used is noticed instead of producing wrong questions. The build stops if a source
disagrees. A saved release is checked in full, a JSONL release on its first rows per file.

Usage:
  python ocr2_questions.py --sources DIR --release DIR_OR_JSONL [...] --output FILE.parquet
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


JSONL_CHECK_ROWS = 20000


def referenced(release, wanted=None):
    """(dataset, split, index) -> (source, difficulty) for the rows of a release: all rows of
    a saved dataset, the first JSONL_CHECK_ROWS rows of each JSONL file of a raw download."""
    wanted = {} if wanted is None else wanted
    path = Path(release)
    files = sorted(path.glob("**/*.jsonl")) if path.is_dir() else [path]
    if path.is_dir() and (path / "dataset_dict.json").exists():
        from datasets import load_from_disk

        rows = (r for split in load_from_disk(str(path)).values()
                for r in split.select_columns(["dataset", "split", "index", "source", "difficulty"]))
    else:
        def rows():
            for file in files:
                with open(file) as handle:
                    for number, line in enumerate(handle):
                        if number >= JSONL_CHECK_ROWS:
                            break
                        yield json.loads(line)
        rows = rows()
    for row in rows:
        if row.get("dataset") is None or row.get("index") in (None, ""):
            continue
        key = (row["dataset"], row["split"], int(row["index"]))
        wanted.setdefault(key, (row.get("source"), row.get("difficulty")))
    return wanted


def build(sources, releases, output):
    wanted = {}
    for release in releases:
        referenced(release, wanted)
    loaded = load_sources(sources)
    counts = Counter()
    for (dataset, split, index), (source, difficulty) in sorted(wanted.items()):
        if (dataset, split) in EVALUATION_SPLITS:
            counts["evaluation split, not looked up"] += 1
            continue
        if (dataset, split) not in TRAIN_SPLITS:
            raise ValueError(f"Unknown source split: {dataset} {split}")
        rows, question_of, agrees, _ = loaded[dataset]
        if index >= len(rows):
            raise ValueError(f"{dataset} {split} has no row {index} ({len(rows)} rows)")
        if not question_of(rows[index]):
            raise ValueError(f"{dataset} {split} row {index} has no problem statement")
        agreed = bool(agrees(rows[index], source, difficulty))
        counts[(dataset, "labels agree" if agreed else "labels differ")] += 1
    # Every training problem of the sources, so any row of either release can be restored
    records = []
    for dataset, (rows, question_of, _, _) in loaded.items():
        for index, source_row in enumerate(rows):
            question = question_of(source_row)
            if question:
                records.append(dict(dataset=dataset, split="train", index=index, question=question))
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
    return dict(report=report, revisions=revisions, evaluation_split_references=counts["evaluation split, not looked up"], written=len(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--release", nargs="+", required=True,
                        help="downloaded release folders or JSONL files whose references are checked")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.sources, args.release, args.output), indent=2))
    except ValueError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
