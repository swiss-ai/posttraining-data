"""Incremental record/token statistics for classified mixture records."""

from __future__ import annotations

from collections import Counter
from typing import Any


LABEL_DIMENSIONS = (
    "expert_domains",
    "task_domains",
    "programming_languages",
    "work_context",
    "interaction_styles",
    "swe.change_scope",
    "swe.artifact_types",
    "terminal.activities",
    "terminal.command_names",
    "tool_use.tool_types",
    "tool_use.available_tool_names",
    "tool_use.invoked_tool_names",
    "tool_use.trajectory_properties",
)


class Summary:
    """Mergeable counters used while streaming annotations."""

    def __init__(self) -> None:
        self.records = 0
        self.source_tokens = 0
        self.training_loss_tokens = 0
        self.records_with_source_tokens = 0
        self.records_with_training_loss_tokens = 0
        self.datasets: Counter[str] = Counter()
        self.dataset_source_tokens: Counter[str] = Counter()
        self.dataset_training_loss_tokens: Counter[str] = Counter()
        self.dimensions: list[str] = []
        self.labels: dict[str, Counter[str]] = {}
        self.statuses: dict[str, Counter[str]] = {}
        self.source_token_labels: dict[str, Counter[str]] = {}
        self.training_loss_token_labels: dict[str, Counter[str]] = {}
        self.expert_intersections: Counter[str] = Counter()
        self.judge_requests = 0
        self.judge_prompt_tokens = 0
        self.judge_completion_tokens = 0
        self.judge_models: Counter[str] = Counter()
        self.judge_providers: Counter[str] = Counter()
        self.judge_errors: Counter[str] = Counter()

    def _ensure_dimension(self, dimension: str) -> None:
        if dimension in self.labels:
            return
        self.dimensions.append(dimension)
        self.labels[dimension] = Counter()
        self.statuses[dimension] = Counter()
        self.source_token_labels[dimension] = Counter()
        self.training_loss_token_labels[dimension] = Counter()

    def add(self, annotation: dict[str, Any]) -> None:
        self.records += 1
        dataset = annotation.get("source_dataset") or "unknown"
        self.datasets[dataset] += 1
        source_tokens = annotation.get("source_tokens")
        loss_tokens = annotation.get("training_loss_tokens")
        if isinstance(source_tokens, int):
            self.source_tokens += source_tokens
            self.records_with_source_tokens += 1
            self.dataset_source_tokens[dataset] += source_tokens
        if isinstance(loss_tokens, int):
            self.training_loss_tokens += loss_tokens
            self.records_with_training_loss_tokens += 1
            self.dataset_training_loss_tokens[dataset] += loss_tokens
        for dimension, classification in annotation["classifications"].items():
            self._ensure_dimension(dimension)
            self.statuses[dimension][classification["status"]] += 1
            for label in classification["labels"]:
                self.labels[dimension][label] += 1
                if isinstance(source_tokens, int):
                    self.source_token_labels[dimension][label] += source_tokens
                if isinstance(loss_tokens, int):
                    self.training_loss_token_labels[dimension][label] += loss_tokens
        experts = annotation["classifications"]["expert_domains"]["labels"]
        self.expert_intersections["+".join(sorted(experts)) or "unknown"] += 1
        judge_model = annotation.get("judge_model")
        if judge_model:
            self.judge_requests += 1
            self.judge_models[str(judge_model)] += 1
            self.judge_providers[str(annotation.get("judge_provider") or "unknown")] += 1
            if isinstance(annotation.get("judge_prompt_tokens"), int):
                self.judge_prompt_tokens += annotation["judge_prompt_tokens"]
            if isinstance(annotation.get("judge_completion_tokens"), int):
                self.judge_completion_tokens += annotation["judge_completion_tokens"]
            if annotation.get("judge_error"):
                self.judge_errors[str(annotation["judge_error"])] += 1

    @staticmethod
    def _share(count: int, total: int) -> float | None:
        return round(count / total, 8) if total else None

    def _dimension(self, dimension: str) -> dict[str, Any]:
        result = {}
        all_labels = set(self.labels[dimension])
        complete_source_tokens = self.records_with_source_tokens == self.records
        complete_loss_tokens = self.records_with_training_loss_tokens == self.records
        for label in sorted(all_labels):
            count = self.labels[dimension][label]
            source_tokens = self.source_token_labels[dimension][label]
            loss_tokens = self.training_loss_token_labels[dimension][label]
            result[label] = {
                "records": count,
                "record_share": self._share(count, self.records),
                "source_tokens": source_tokens if complete_source_tokens else None,
                "source_token_share": (
                    self._share(source_tokens, self.source_tokens) if complete_source_tokens else None
                ),
                "training_loss_tokens": loss_tokens if complete_loss_tokens else None,
                "training_loss_token_share": (
                    self._share(loss_tokens, self.training_loss_tokens)
                    if complete_loss_tokens
                    else None
                ),
            }
        return result

    def _dataset_statistics(self) -> dict[str, Any]:
        complete_source_tokens = self.records_with_source_tokens == self.records
        complete_loss_tokens = self.records_with_training_loss_tokens == self.records
        return {
            dataset: {
                "records": records,
                "record_share": self._share(records, self.records),
                "source_tokens": (
                    self.dataset_source_tokens[dataset] if complete_source_tokens else None
                ),
                "source_token_share": (
                    self._share(self.dataset_source_tokens[dataset], self.source_tokens)
                    if complete_source_tokens
                    else None
                ),
                "training_loss_tokens": (
                    self.dataset_training_loss_tokens[dataset] if complete_loss_tokens else None
                ),
                "training_loss_token_share": (
                    self._share(
                        self.dataset_training_loss_tokens[dataset], self.training_loss_tokens
                    )
                    if complete_loss_tokens
                    else None
                ),
            }
            for dataset, records in sorted(self.datasets.items())
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "source_tokens": self.source_tokens if self.records_with_source_tokens == self.records else None,
            "training_loss_tokens": (
                self.training_loss_tokens
                if self.records_with_training_loss_tokens == self.records
                else None
            ),
            "token_coverage": {
                "records_with_source_tokens": self.records_with_source_tokens,
                "records_with_training_loss_tokens": self.records_with_training_loss_tokens,
            },
            "source_datasets": dict(sorted(self.datasets.items())),
            "source_dataset_statistics": self._dataset_statistics(),
            "dimensions": {dimension: self._dimension(dimension) for dimension in self.dimensions},
            "statuses": {
                dimension: dict(sorted(counter.items()))
                for dimension, counter in self.statuses.items()
            },
            "expert_domain_intersections": dict(sorted(self.expert_intersections.items())),
            "judge_usage": {
                "requests": self.judge_requests,
                "models": dict(sorted(self.judge_models.items())),
                "providers": dict(sorted(self.judge_providers.items())),
                "prompt_tokens": self.judge_prompt_tokens,
                "completion_tokens": self.judge_completion_tokens,
                "errors": dict(sorted(self.judge_errors.items())),
            },
            "notes": [
                "Multi-label shares may sum to more than 1.0.",
                "Token statistics are null unless every input record supplies the corresponding count.",
            ],
        }
