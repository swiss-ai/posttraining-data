"""Settings of the decontamination, in one place.

Every script of 04-decontamination takes its defaults from here, and submit-parallel-decontamination.sh
reads them by running this file (it prints them as shell assignments). Slurm settings (account, partition,
time limits, ...) are in slurm_config.sh.

The tokenizer and the paths can be overridden from the environment (TOKENIZER_NAME,
DECONTAMINATION_PROMPTS, DECONTAMINATION_CACHE_DIR). The matching settings cannot: change them here, so
that every script uses the same values.

The matching settings are recorded in the reports directory (decontamination_settings.json): reports are
reused on a relaunch, so reports made with other settings must not be mixed in.
"""
import json
import os
import shlex
from pathlib import Path

# ── Tokenizer and paths (overridable from the environment) ──────────────────
# Only cuts the text into comparable units (see matching_units.py), so it is independent of the model being
# trained: keep it fixed (the Apertus 1.5 tokenizer) to keep results comparable across runs.
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "swiss-ai/Apertus-8B-Instruct-2509")
DECONTAMINATION_PROMPTS = os.environ.get(
    "DECONTAMINATION_PROMPTS",
    "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts",
)
DECONTAMINATION_CACHE_DIR = os.environ.get(
    "DECONTAMINATION_CACHE_DIR",
    f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}/posttraining-data/decontamination_cache",
)

# ── Matching (see matching_units.py and decontamination_common.py) ──────────
NGRAM_LENGTH = 8         # a training prompt is a candidate if it shares an n-gram of this many units
DIFF_THRESHOLD = 0.5     # contaminated if matched runs cover at least this share of the benchmark prompt
MIN_BLOCK = 5            # only matched runs of at least this many units count towards that share
# Benchmark prompts shorter than NGRAM_LENGTH:
EXACT_MAX_UNITS = 3      # up to this many units, the training prompt must be identical
SHORT_MIN_COVERAGE = 0.5  # otherwise it must contain the benchmark prompt, making up this share of it

SETTINGS_FILE = "decontamination_settings.json"


def matching_settings(tokenizer_name, ngram_length, diff_threshold):
    """The settings that decide which samples a report flags."""
    from matching_units import MATCHING_VERSION, UNITS_VERSION

    return {
        "tokenizer_name": tokenizer_name,
        "ngram_length": ngram_length,
        "diff_threshold": diff_threshold,
        "min_block": MIN_BLOCK,
        "exact_max_units": EXACT_MAX_UNITS,
        "short_min_coverage": SHORT_MIN_COVERAGE,
        "units_version": UNITS_VERSION,
        "matching_version": MATCHING_VERSION,
    }


def check_report_settings(report_path, settings, write=False):
    """Compare settings with the ones recorded in report_path; record them if write and none are recorded.

    Returns a list of (name, recorded, current) for every setting that differs. Raises nothing: the
    callers decide whether a mismatch is an error. Reports of runs before the settings were recorded have
    no file; they are assumed to match (a warning is printed)."""
    settings_file = Path(report_path) / SETTINGS_FILE
    if settings_file.exists():
        with open(settings_file) as f:
            recorded = json.load(f)
        return [(k, recorded.get(k), v) for k, v in settings.items() if recorded.get(k) != v]
    if any(Path(report_path).glob("*__contamination_report.json")):
        print(f"Warning: {report_path} has reports but no {SETTINGS_FILE} (made before the settings were "
              f"recorded); assuming they were made with the current settings.")
    if write:
        tmp = settings_file.with_name(f"{SETTINGS_FILE}.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp, settings_file)  # parallel array jobs write the same content
    return []


def format_mismatch(report_path, mismatches):
    lines = [f"The reports in {report_path} were made with other settings ({SETTINGS_FILE}):"]
    lines += [f"  {name}: reports {recorded!r}, now {current!r}" for name, recorded, current in mismatches]
    lines.append("Use a new output path, or delete the reports to recompute them.")
    return "\n".join(lines)


if __name__ == "__main__":
    # Shell assignments for submit-parallel-decontamination.sh: eval "$(python decontamination_settings.py)"
    for name in ["TOKENIZER_NAME", "DECONTAMINATION_PROMPTS", "DECONTAMINATION_CACHE_DIR"]:
        print(f"{name}={shlex.quote(str(globals()[name]))}")
