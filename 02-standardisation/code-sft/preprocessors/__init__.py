"""Source column mappings and explicit catalog selection policies.

Original source objects are retained by unified.py before these adapters run.
"""

import json
from pathlib import Path
import yaml
from .base import DatasetPreprocessor, prepare_fields, benchmark_fields

BASE = Path(__file__).resolve().parent.parent
CATALOG = json.loads((BASE / "adapter-policies.json").read_text())
ENTRIES = {entry["repo"]: entry for entry in CATALOG["datasets"]}
BENCHMARKS = {
    entry["repo"]: entry
    for entry in yaml.safe_load(
        (BASE.parents[1] / "04-decontamination/code-sft/benchmarks.yaml").read_text()
    )["datasets"]
}


class CatalogPreprocessor(DatasetPreprocessor):
    def prepare(self, row, entry, file):
        if entry["category"] == "benchmarks":
            row = benchmark_fields(
                row, dict(BENCHMARKS.get(entry["repo"], {}), **entry), file["path"]
            )
        return prepare_fields(
            row,
            sql_context=entry["repo"] == "b-mc2/sql-create-context",
            instruction_only=entry["repo"]
            == "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        )

    def configure(self, entry):
        entry = super().configure(entry)
        if self.repo == "open-r1/Mixture-of-Thoughts" and "files" in entry:
            entry["files"] = [
                file for file in entry["files"] if file["path"].startswith("code/")
            ]
        return entry


PROCESSORS = {}
for repo, entry in {**ENTRIES, **BENCHMARKS}.items():
    cls = type(
        repo.replace("/", "_").replace("-", "_"),
        (CatalogPreprocessor,),
        {
            "repo": repo,
            "policy": entry.get("code_policy", "code_source"),
        },
    )
    PROCESSORS[repo] = cls()

# Preserve the concrete dataset adapters that were available in source form.
for path in sorted(Path(__file__).parent.glob("*.py")):
    if path.stem in {"base", "__init__"}:
        continue
    module = __import__(f"{__name__}.{path.stem}", fromlist=["*"])
    for value in vars(module).values():
        if (
            isinstance(value, type)
            and issubclass(value, DatasetPreprocessor)
            and value.repo
        ):
            PROCESSORS[value.repo] = value()


def get_processor(repo):
    return PROCESSORS.get(repo, DatasetPreprocessor())
