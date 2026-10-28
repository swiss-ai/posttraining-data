"""Run the code-SFT recipe using implementations in the numbered pipeline stages."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
STANDARD = ROOT / "02-standardisation/code-sft"
DECONTAMINATION = ROOT / "04-decontamination/code-sft"
STAGES = {
    "standardise": STANDARD / "standardise_native.py",
    "unify": STANDARD / "standardise_native.py",
    "benchmarks": DECONTAMINATION / "benchmarks.py",
    "dedup": DECONTAMINATION / "dedup.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(STANDARD), str(DECONTAMINATION), env.get("PYTHONPATH", "")]
    )
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    return subprocess.call(
        [sys.executable, str(STAGES[args.stage]), *arguments], env=env
    )


if __name__ == "__main__":
    raise SystemExit(main())
