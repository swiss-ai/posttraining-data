#!/usr/bin/env python3
"""
Smoke test of the code preprocessing steps on a few rows of ONE downloaded dataset.

Run by the Slurm jobs that smoke-all-swe-terminal.sh submits (one job per dataset),
but it can be run by hand too. For every sample of the dataset it does:

  1. sample       take the first N rows (default 5)
  2. standardise  examples/code-sft/pipeline.py standardise  (checked Apertus 2 native rows)
  3. dedup        examples/code-sft/pipeline.py dedup        (exact dedup + benchmark decontamination)

A "sample" is one split of a dataset saved with save_to_disk, or one file of a raw
download (jsonl / parquet / json). A dataset folder that only contains sub-datasets
(commitpackft/<language>, termgrade-trajectories/<subset>) is handled one sub-dataset
after the other, up to --max-subsets of them.

Everything is kept under <out>/<dataset>/[<subset>/]<sample>/:
  sample/  native/  dedup/  standardise.log  dedup.log  result.json
and the dataset gets <out>/<dataset>/summary.json. `--report` prints a table over all
datasets that have a summary.

Exit code 0 only if every sample was processed and produced accepted rows.
"""

import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

RAW_SUFFIXES = (".jsonl", ".parquet", ".json", ".tar")

# The trajectory converter lives next to the standardiser
sys.path.insert(0, str(Path(__file__).resolve().parent / "02-standardisation" / "code-sft"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "x"


# --------------------------------------------------------------------------- discovery

def discover_units(src_dir):
    """A dataset folder is one unit, or a folder of sub-datasets (one unit each)."""
    if (src_dir / "dataset_metadata.json").exists():
        return [(None, src_dir)]
    subs = sorted(p.parent for p in src_dir.glob("*/dataset_metadata.json"))
    return [(p.name, p) for p in subs]


# --------------------------------------------------------------------------- sampling

def sample_saved(unit_dir, sample_root, rows):
    """One 5-row dataset per split of a save_to_disk dataset."""
    from datasets import DatasetDict, load_from_disk

    ds = load_from_disk(str(unit_dir))
    splits = dict(ds) if isinstance(ds, DatasetDict) else {"train": ds}
    out = []
    for split, d in splits.items():
        label = safe(split)
        n = min(rows, len(d))
        if n == 0:
            out.append({"label": label, "kind": "saved", "skipped": "split is empty"})
            continue
        target = sample_root / label
        if target.exists():
            shutil.rmtree(target)
        DatasetDict({split: d.select(range(n))}).save_to_disk(str(target))
        shutil.copy(unit_dir / "dataset_metadata.json", target / "dataset_metadata.json")
        out.append({"label": label, "kind": "saved", "input": str(target), "rows": n,
                    "source": f"split {split}"})
    return out


def sample_raw(unit_dir, meta, sample_root, rows, max_files):
    """One 5-row file per raw data file (files can have different columns)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = [f["path"] for f in meta["files"] if Path(f["path"]).suffix in RAW_SUFFIXES]
    if any(f.endswith(".tar") for f in files):
        # Trajectory packages: the tars hold the data, the other files are catalogs
        files = [f for f in files if f.endswith(".tar")]
    out = []
    for rel in files[:max_files]:
        src = unit_dir / rel
        label = safe(rel)
        target = sample_root / label
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        suffix = src.suffix
        if suffix == ".tar":
            # ATIF trajectories -> chat rows (02-standardisation/code-sft/atif_trajectories.py)
            from atif_trajectories import convert

            counts = convert([src], target, limit=rows)
            dest, n = target / "part-00000.jsonl", counts["written"]
        elif suffix == ".jsonl":
            dest, n = target / "sample.jsonl", 0
            with open(src, "rb") as fin, open(dest, "wb") as fout:
                for _ in range(rows):
                    line = fin.readline()
                    if not line:
                        break
                    fout.write(line if line.endswith(b"\n") else line + b"\n")
                    n += 1
        elif suffix == ".parquet":
            dest = target / "sample.parquet"
            batch = next(pq.ParquetFile(src).iter_batches(batch_size=rows), None)
            n = 0 if batch is None else batch.num_rows
            if batch is not None:
                pq.write_table(pa.Table.from_batches([batch]), dest)
        else:  # .json: either one JSON list of records, or JSON Lines under a .json name
            dest = target / "sample.jsonl"
            text = open(src, encoding="utf-8").read()
            try:
                data = json.loads(text)
                if not isinstance(data, list):
                    raise ValueError(f"{rel}: expected a JSON list of records, got {type(data).__name__}")
                data = data[:rows]
            except json.JSONDecodeError:
                data = [json.loads(line) for line in text.splitlines()[:rows] if line.strip()]
            with open(dest, "w", encoding="utf-8") as fout:
                for row in data:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n = len(data)
        if n == 0:
            out.append({"label": label, "kind": "raw", "skipped": "file has no rows"})
            continue
        out.append({"label": label, "kind": "raw", "input": str(dest), "rows": n,
                    "source": f"file {rel}"})
    if len(files) > max_files:
        log(f"  only the first {max_files} of {len(files)} files are sampled (--max-files)")
        out.append({"label": "_not_sampled", "kind": "raw", "skipped":
                    f"{len(files) - max_files} more files not sampled (--max-files)", "note_only": True})
    return out


# --------------------------------------------------------------------------- steps

def run(cmd, cwd, log_path, timeout):
    t0 = time.time()
    with open(log_path, "w") as lf:
        try:
            rc = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
            lf.write(f"\nTIMEOUT after {timeout} s\n")
    return rc, round(time.time() - t0, 1)


def last_json_line(path, prefix, whole=False):
    """The JSON on the last line that starts with `prefix` (the prefix is part of the JSON if whole)."""
    for line in reversed(Path(path).read_text(errors="replace").splitlines()):
        if line.startswith(prefix):
            try:
                return json.loads(line if whole else line[len(prefix):])
            except ValueError:
                return None
    return None


def rejection_reasons(native_dir, top=3):
    counter = Counter()
    for f in Path(native_dir).rglob("rejected.jsonl"):
        with open(f, errors="replace") as fh:
            for line in fh:
                try:
                    err = json.loads(line).get("error", "")
                except ValueError:
                    continue
                counter[re.sub(r"\s+", " ", str(err))[:220]] += 1
    return [{"count": c, "error": e} for e, c in counter.most_common(top)]


def pick_category(meta, requested, reference):
    """code_agentic or code_sft: explicit, or looked up in the shared reference folder."""
    if requested != "auto":
        return requested, "set by --category"
    repo = meta.get("dataset_name") or ""
    if "/" in repo and reference:
        owner, name = repo.split("/", 1)
        for category in ("code_agentic", "code_sft"):
            if (Path(reference) / category / owner / name).is_dir():
                return category, f"found in {reference}"
    return "code_sft", "default (not in the reference folder)"


def process_sample(sample, base, meta, args, unit_label):
    result = {"dataset": args.name, "unit": unit_label, "label": sample["label"],
              "kind": sample.get("kind"), "source": sample.get("source"),
              "sample_rows": sample.get("rows")}
    sdir = base / sample["label"]
    sdir.mkdir(parents=True, exist_ok=True)
    category, why = pick_category(meta, args.category, args.category_ref)
    result["category"], result["category_reason"] = category, why
    native, dedup = sdir / "native", sdir / "dedup"
    for d in (native, dedup):
        if d.exists():
            shutil.rmtree(d)

    cmd = [args.python, "pipeline.py", "standardise", "--input", sample["input"],
           "--output", str(native), "--category", category]
    if sample["kind"] == "raw":
        cmd += ["--repo", meta["dataset_name"]]
        if meta.get("commit_hash"):
            cmd += ["--revision", meta["commit_hash"]]
    rc, secs = run(cmd, args.code_sft_dir, sdir / "standardise.log", args.timeout)
    st = last_json_line(sdir / "standardise.log", '{"rows"', whole=True) or {}
    counts = st.get("counts") or {}
    result["standardise"] = {"returncode": rc, "seconds": secs, "log": str(sdir / "standardise.log"),
                             "read": counts.get("read"), "written": counts.get("written"),
                             "quarantined": counts.get("quarantined")}
    if counts.get("quarantined"):
        result["standardise"]["top_rejections"] = rejection_reasons(native)

    if rc != 0:
        result["status"], result["problem"] = "failed", "standardise crashed (see standardise.log)"
        return result
    if not counts.get("written"):
        result["status"], result["problem"] = "no_rows", "standardise accepted 0 rows"
        return result

    cmd = [args.python, "pipeline.py", "dedup", "--input", str(native), "--benchmarks", args.benchmarks,
           "--output", str(dedup), "--categories", category, "--workers", str(args.workers),
           "--memory-limit", args.memory_limit, "--no-fuzzy"]
    rc, secs = run(cmd, args.code_sft_dir, sdir / "dedup.log", args.timeout)
    dec = last_json_line(sdir / "dedup.log", "DECISIONS ") or {}
    result["dedup"] = {"returncode": rc, "seconds": secs, "log": str(sdir / "dedup.log"), "decisions": dec}
    if rc != 0:
        result["status"], result["problem"] = "failed", "dedup crashed (see dedup.log)"
    else:
        result["status"] = "ok"
    return result


# --------------------------------------------------------------------------- one dataset

def run_dataset(args):
    src = Path(args.src) / args.name
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    units = discover_units(src)
    if not units:
        raise SystemExit(f"{src}: no dataset_metadata.json here or one level down")
    skipped_units = max(0, len(units) - args.max_subsets) if units[0][0] else 0
    units = units[: args.max_subsets] if units[0][0] else units
    log(f"{args.name}: {len(units)} unit(s)" + (f" ({skipped_units} more not sampled, --max-subsets)" if skipped_units else ""))

    results, notes = [], []
    if skipped_units:
        notes.append(f"{skipped_units} sub-datasets not sampled (--max-subsets {args.max_subsets})")
    for unit_label, unit_dir in units:
        base = out / unit_label if unit_label else out
        meta = json.load(open(unit_dir / "dataset_metadata.json"))
        raw = str(meta.get("download_method", "")).startswith("snapshot_download")
        log(f"unit {unit_label or '.'}: {'raw files' if raw else 'saved dataset'}")
        try:
            if raw:
                samples = sample_raw(unit_dir, meta, base / "_samples", args.rows, args.max_files)
            else:
                samples = sample_saved(unit_dir, base / "_samples", args.rows)
        except Exception as e:
            traceback.print_exc()
            results.append({"dataset": args.name, "unit": unit_label, "label": "(sampling)",
                            "status": "failed", "problem": f"sampling failed: {e}"})
            continue
        for sample in samples:
            if sample.get("note_only"):
                notes.append(sample["skipped"])
                continue
            if sample.get("skipped"):
                results.append({"dataset": args.name, "unit": unit_label, "label": sample["label"],
                                "status": "skipped", "problem": sample["skipped"]})
                continue
            log(f"  sample {sample['label']} ({sample['rows']} rows)")
            try:
                res = process_sample(sample, base, meta, args, unit_label)
            except Exception as e:
                traceback.print_exc()
                res = {"dataset": args.name, "unit": unit_label, "label": sample["label"],
                       "status": "failed", "problem": f"{type(e).__name__}: {e}"}
            (base / sample["label"]).mkdir(parents=True, exist_ok=True)
            json.dump(res, open(base / sample["label"] / "result.json", "w"), indent=2)
            log(f"    -> {res['status']}" + (f": {res.get('problem')}" if res.get("problem") else ""))
            results.append(res)

    counts = Counter(r["status"] for r in results)
    overall = "ok" if results and set(counts) <= {"ok", "skipped"} and counts.get("ok") else "not_ok"
    summary = {"dataset": args.name, "overall": overall, "status_counts": dict(counts),
               "rows_requested": args.rows, "notes": notes, "samples": results,
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    log(f"{args.name}: {overall} {dict(counts)}")
    return 0 if overall == "ok" else 1


# --------------------------------------------------------------------------- report

def report(args):
    out = Path(args.out)
    excluded = set()
    if args.exclude_file and Path(args.exclude_file).exists():
        for line in open(args.exclude_file):
            name = line.split("#", 1)[0].strip()
            if name:
                excluded.add(name)
    summaries = {p.parent.name: json.load(open(p)) for p in sorted(out.glob("*/summary.json"))
                 if p.parent.name not in excluded}
    names = sorted(summaries)
    rows = []
    for n in names:
        s = summaries[n]
        c = s["status_counts"]
        issue = ""
        for r in s["samples"]:
            if r["status"] not in ("ok", "skipped"):
                top = (r.get("standardise") or {}).get("top_rejections") or []
                issue = r.get("problem", "") + (f" | {top[0]['error'][:110]}" if top else "")
                break
        rows.append((n, s["overall"], c.get("ok", 0), c.get("no_rows", 0), c.get("failed", 0), issue))
    width = max([len(r[0]) for r in rows] + [7])
    print(f"{'dataset':{width}}  overall  ok  no_rows  failed  first issue")
    for n, overall, ok, nr, fl, issue in rows:
        print(f"{n:{width}}  {overall:7}  {ok:2}  {nr:7}  {fl:6}  {issue}")
    total_ok = sum(1 for r in rows if r[1] == "ok")
    print(f"\n{total_ok} of {len(rows)} datasets with a summary are fully ok"
          + (f" ({len(excluded)} excluded datasets are not counted)" if excluded else ""))
    if args.src:
        missing = sorted(p.name for p in Path(args.src).iterdir()
                         if p.is_dir() and p.name not in summaries and p.name not in excluded
                         and not p.name.startswith("."))
        print(f"{len(missing)} dataset folders in {args.src} have no summary yet" +
              (f": {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}" if missing else ""))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", help="dataset folder name under --src (not needed with --report)")
    p.add_argument("--src", required=False, help="folder with the downloaded datasets")
    p.add_argument("--out", required=True, help="folder for samples, artifacts and summaries")
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--max-subsets", type=int, default=5, help="sub-datasets sampled per dataset folder")
    p.add_argument("--max-files", type=int, default=8, help="raw files sampled per raw dataset")
    p.add_argument("--python", default=sys.executable, help="python of the code_sft env")
    p.add_argument("--code-sft-dir", default=str(Path(__file__).resolve().parent / "examples" / "code-sft"))
    p.add_argument("--benchmarks", help="benchmark references (pipeline.py benchmarks output)")
    p.add_argument("--category", default="auto", choices=["auto", "code_sft", "code_agentic"])
    p.add_argument("--category-ref", default="/capstor/store/cscs/swissai/infra01/datasets/code_posttrain/sft")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--memory-limit", default="16GB")
    p.add_argument("--timeout", type=int, default=3600, help="seconds per step")
    p.add_argument("--report", action="store_true", help="print a table over all summaries")
    p.add_argument("--exclude-file", help="with --report: datasets listed here are left out of the table")
    args = p.parse_args()
    if args.report:
        sys.exit(report(args))
    if not (args.name and args.src and args.benchmarks):
        p.error("--name, --src and --benchmarks are required")
    sys.exit(run_dataset(args))


if __name__ == "__main__":
    main()
