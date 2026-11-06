#!/usr/bin/env python3
"""
Full processing of ONE downloaded dataset: every row of every split or file.

Run by the Slurm jobs that full-swe-terminal.sh submits (one job per dataset, one whole node
each). For every unit of the dataset (the dataset, or each of its sub-datasets) it does:

  1. prepare    trajectory tars -> JSONL (pipeline.py atif); a JSON list file -> JSONL
  2. standardise  examples/code-sft/pipeline.py standardise, split over N worker processes
                that run side by side on the node, each taking every N-th row of each split
                or file (N grows with the size of the data, up to --max-workers)
  3. dedup      examples/code-sft/pipeline.py dedup over this unit's output only: exact
                dedup and benchmark decontamination within the dataset, no fuzzy matching

Terminal and SWE-agent trajectories are written as OpenCode sessions (the default harness).

Everything is kept under <out>/<dataset>/[<subset>/]:
  native/  dedup/  logs/  [_prepared/]  result.json
and the dataset gets <out>/<dataset>/summary.json, in the format of the smoke test, so
`smoke-all-swe-terminal-run.py --report --out <out>` prints the same table.

A rerun continues where the last one stopped: standardise keeps every finished partition and
dedup resumes from its frozen inputs. Exit code 0 only if every unit was processed.
"""

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("smoke", REPO / "smoke-all-swe-terminal-run.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

DATA_SUFFIXES = (".jsonl", ".parquet", ".json", ".tar")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def size_of(paths):
    total = 0
    for path in paths:
        path = Path(path)
        if path.is_file():
            total += path.stat().st_size
        else:
            total += sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and f.suffix in (".arrow", ".parquet", ".jsonl"))
    return total


def convert_tar(arguments):
    tar, target = arguments
    sys.path.insert(0, str(REPO / "02-standardisation" / "code-sft"))
    from atif_trajectories import convert

    return convert([tar], target)


def prepare(unit_dir, meta, work, processes):
    """The input paths for standardise and whether they are raw files (need --repo)."""
    if not str(meta.get("download_method", "")).startswith("snapshot_download"):
        return [unit_dir], False
    files = [unit_dir / f["path"] for f in meta["files"] if Path(f["path"]).suffix in DATA_SUFFIXES]
    tars = [f for f in files if f.suffix == ".tar"]
    if tars:
        # Trajectory packages: the tars hold the data, the other files are catalogs
        target = work / "_prepared" / "atif"
        done = target / "_DONE.json"
        if not done.exists():
            if target.exists():
                shutil.rmtree(target)
            jobs = [(str(t), str(target / f"{n:05d}")) for n, t in enumerate(tars)]
            with ProcessPoolExecutor(max_workers=processes) as pool:
                counts = list(pool.map(convert_tar, jobs))
            json.dump({"tars": len(tars), "counts": counts}, open(done, "w"), indent=2)
        return sorted(target.rglob("part-*.jsonl")), True
    inputs = []
    for path in files:
        if path.suffix == ".json":
            # A JSON list of records (or JSON Lines under a .json name) -> JSON Lines
            converted = work / "_prepared" / (path.stem + ".jsonl")
            if not converted.exists():
                converted.parent.mkdir(parents=True, exist_ok=True)
                text = open(path, encoding="utf-8").read()
                try:
                    data = json.loads(text)
                    if not isinstance(data, list):
                        raise ValueError(f"{path}: expected a JSON list of records")
                except json.JSONDecodeError:
                    data = [json.loads(line) for line in text.splitlines() if line.strip()]
                partial = converted.with_suffix(".partial")
                with open(partial, "w", encoding="utf-8") as handle:
                    for row in data:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                partial.rename(converted)
            inputs.append(converted)
        else:
            inputs.append(path)
    return inputs, True


def standardise(inputs, raw, meta, category, native, logs, args, workers):
    base = [args.python, "pipeline.py", "standardise", "--input", *map(str, inputs),
            "--output", str(native), "--category", category,
            "--terminal-harness", "opencode", "--workers", str(workers)]
    if raw:
        base += ["--repo", meta["dataset_name"]]
        if meta.get("commit_hash"):
            base += ["--revision", meta["commit_hash"]]
    else:
        base += ["--split", "*"]
    logs.mkdir(parents=True, exist_ok=True)
    processes, started = [], time.time()
    for worker in range(workers):
        handle = open(logs / f"standardise-{worker:03d}.log", "w")
        processes.append((worker, handle, subprocess.Popen(
            base + ["--worker", str(worker)], cwd=args.code_sft_dir, stdout=handle, stderr=subprocess.STDOUT
        )))
    failed = []
    for worker, handle, process in processes:
        if process.wait() != 0:
            failed.append(worker)
        handle.close()
    return failed, round(time.time() - started, 1)


def process_unit(name, unit_label, unit_dir, base, args):
    meta = json.load(open(unit_dir / "dataset_metadata.json"))
    category, why = smoke.pick_category(meta, args.category, args.category_ref)
    result = {"dataset": name, "unit": unit_label, "label": "all", "kind": "full",
              "category": category, "category_reason": why}
    base.mkdir(parents=True, exist_ok=True)
    native, dedup, logs = base / "native", base / "dedup", base / "logs"

    inputs, raw = prepare(unit_dir, meta, base, args.max_workers)
    if not inputs:
        result.update(status="skipped", problem="no data files")
        return result
    gigabytes = size_of(inputs) / 1e9
    workers = max(1, min(args.max_workers, math.ceil(gigabytes / args.gb_per_worker)))
    result["input_gb"], result["workers"] = round(gigabytes, 2), workers
    log(f"  {unit_label or name}: {len(inputs)} input(s), {gigabytes:.1f} GB, {workers} worker(s), {category}")

    failed, seconds = standardise(inputs, raw, meta, category, native, logs, args, workers)
    success = list(native.glob("*/*/*/_SUCCESS.json"))
    counts = json.load(open(success[0]))["counts"] if success else {}
    result["standardise"] = {"returncode": 1 if failed else 0, "seconds": seconds, "log": str(logs),
                             "workers": workers, "failed_workers": failed,
                             "read": counts.get("read"), "written": counts.get("written"),
                             "quarantined": counts.get("quarantined")}
    if counts.get("quarantined"):
        result["standardise"]["top_rejections"] = smoke.rejection_reasons(native, top=5)
    if failed or not success:
        result["status"], result["problem"] = "failed", f"standardise failed in worker(s) {failed} (see {logs})"
        return result
    if not counts.get("written"):
        result["status"], result["problem"] = "no_rows", "standardise accepted 0 rows"
        return result

    cmd = [args.python, "pipeline.py", "dedup", "--input", str(native), "--benchmarks", args.benchmarks,
           "--output", str(dedup), "--categories", category, "--workers", str(args.dedup_workers),
           "--memory-limit", args.memory_limit, "--no-fuzzy"]
    rc, secs = smoke.run(cmd, args.code_sft_dir, logs / "dedup.log", args.timeout)
    decisions = smoke.last_json_line(logs / "dedup.log", "DECISIONS ") or {}
    result["dedup"] = {"returncode": rc, "seconds": secs, "log": str(logs / "dedup.log"), "decisions": decisions}
    result["status"] = "ok" if rc == 0 else "failed"
    if rc != 0:
        result["problem"] = "dedup failed (see dedup.log)"
    return result


def run_dataset(args):
    src = Path(args.src) / args.name
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    units = smoke.discover_units(src)
    if not units:
        raise SystemExit(f"{src}: no dataset_metadata.json here or one level down")
    log(f"{args.name}: {len(units)} unit(s)")
    results = []
    for unit_label, unit_dir in units:
        base = out / unit_label if unit_label else out
        try:
            res = process_unit(args.name, unit_label, unit_dir, base, args)
        except Exception as error:
            traceback.print_exc()
            res = {"dataset": args.name, "unit": unit_label, "label": "all", "status": "failed",
                   "problem": f"{type(error).__name__}: {error}"}
        json.dump(res, open(base / "result.json", "w"), indent=2)
        log(f"    -> {res['status']}" + (f": {res.get('problem')}" if res.get("problem") else ""))
        results.append(res)
    counts = Counter(r["status"] for r in results)
    overall = "ok" if results and set(counts) <= {"ok", "skipped"} and counts.get("ok") else "not_ok"
    summary = {"dataset": args.name, "overall": overall, "status_counts": dict(counts), "rows_requested": "all",
               "notes": [], "samples": results, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    log(f"{args.name}: {overall} {dict(counts)}")
    return 0 if overall == "ok" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="dataset folder name under --src")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--python", default=sys.executable, help="python of the code_sft env")
    p.add_argument("--code-sft-dir", default=str(REPO / "examples" / "code-sft"))
    p.add_argument("--benchmarks", required=True, help="benchmark references (pipeline.py benchmarks output)")
    p.add_argument("--category", default="auto", choices=["auto", "code_sft", "code_agentic"])
    p.add_argument("--category-ref", default="/capstor/store/cscs/swissai/infra01/datasets/code_posttrain/sft")
    p.add_argument("--max-workers", type=int, default=64, help="standardise processes per dataset unit")
    p.add_argument("--gb-per-worker", type=float, default=2.0, help="input size per standardise process")
    p.add_argument("--dedup-workers", type=int, default=32)
    p.add_argument("--memory-limit", default="128GB", help="DuckDB memory cap of dedup")
    p.add_argument("--timeout", type=int, default=12 * 3600, help="seconds for dedup")
    args = p.parse_args()
    sys.exit(run_dataset(args))


if __name__ == "__main__":
    main()
