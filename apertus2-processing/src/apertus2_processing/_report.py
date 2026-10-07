"""Mergeable counts and streaming issue records."""

from collections import Counter
from dataclasses import asdict

from ._util import canonical


class Report:
    def __init__(self):
        self.rows = Counter()
        self.occurrences = Counter()
        self.affected = Counter()

    def add(self, location, issues, sink, *, accepted):
        self.rows["processed"] += 1
        self.rows["accepted" if accepted else "rejected"] += 1
        severities = {issue["severity"] for issue in issues}
        for severity in severities:
            self.rows[f"rows_with_{severity}"] += 1
        keys = set()
        for issue in issues:
            sink.write(canonical({**location, **issue}) + "\n")
            key = (issue["rule"], issue["severity"], location["source"], location["split"])
            self.occurrences[key] += 1
            keys.add(key)
        self.affected.update(keys)

    def as_dict(self):
        return {
            "rows": dict(self.rows),
            "issues": [
                {
                    "rule": k[0],
                    "severity": k[1],
                    "source": k[2],
                    "split": k[3],
                    "occurrences": count,
                    "affected_records": self.affected[k],
                }
                for k, count in sorted(self.occurrences.items())
            ],
        }

    def merge(self, summary):
        self.rows.update(summary["rows"])
        for issue in summary["issues"]:
            key = tuple(issue[k] for k in ("rule", "severity", "source", "split"))
            self.occurrences[key] += issue["occurrences"]
            self.affected[key] += issue["affected_records"]


def findings(report):
    return [
        {
            "rule": v.rule,
            "severity": str(v.severity),
            "location": asdict(v.location),
            "message": v.message,
        }
        for v in report.violations
    ] + [
        {
            "rule": v.rule,
            "severity": "unevaluated",
            "location": asdict(v.location),
            "message": v.message,
        }
        for v in report.unevaluated
    ]


def structural(error, rule="structure/invalid"):
    return {"rule": rule, "severity": "error", "location": None, "message": str(error)}
