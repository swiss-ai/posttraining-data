"""Load and validate the versioned SWE/Terminal/Tool Use taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class TaxonomyError(ValueError):
    """Raised when a taxonomy or generated annotation violates the contract."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaxonomyError(f"{path} must be a mapping")
    return value


def _labels(dimension: dict[str, Any], path: str) -> frozenset[str]:
    labels = _mapping(dimension.get("labels"), f"{path}.labels")
    if not labels:
        raise TaxonomyError(f"{path}.labels must not be empty")
    return frozenset(labels)


def _grouped_labels(dimension: dict[str, Any], path: str) -> frozenset[str]:
    groups = _mapping(dimension.get("groups"), f"{path}.groups")
    labels: set[str] = set()
    for group_name, group in groups.items():
        group = _mapping(group, f"{path}.groups.{group_name}")
        group_labels = _mapping(group.get("labels"), f"{path}.groups.{group_name}.labels")
        duplicate = labels.intersection(group_labels)
        if duplicate:
            raise TaxonomyError(f"duplicate labels in {path}: {sorted(duplicate)}")
        labels.update(group_labels)
    if not labels:
        raise TaxonomyError(f"{path}.groups must define at least one label")
    return frozenset(labels)


@dataclass(frozen=True)
class Taxonomy:
    """Validated taxonomy plus flattened label registries used by classifiers."""

    path: Path
    document: dict[str, Any]
    taxonomy_id: str
    version: str
    statuses: frozenset[str]
    labels: dict[str, frozenset[str]]
    semantic_judge_dimensions: tuple[str, ...]
    dimension_experts: dict[str, str]

    def validate_labels(self, dimension: str, values: list[str] | tuple[str, ...]) -> None:
        if dimension not in self.labels:
            raise TaxonomyError(f"unknown taxonomy dimension: {dimension}")
        unknown = set(values).difference(self.labels[dimension])
        if unknown:
            raise TaxonomyError(f"unknown {dimension} labels: {sorted(unknown)}")

    def definition(self, dimension: str, label: str) -> dict[str, Any]:
        """Return a label definition for prompt generation or human inspection."""
        dimensions = self.document["dimensions"]
        if dimension in dimensions and "groups" in dimensions[dimension]:
            for group in dimensions[dimension]["groups"].values():
                if label in group["labels"]:
                    return group["labels"][label]
        elif dimension == "programming_languages":
            if label in dimensions[dimension]["canonical_labels"]:
                return {"description": label}
        elif "." in dimension:
            section, child = dimension.split(".", 1)
            return dimensions[section]["dimensions"][child]["labels"][label]
        elif dimension in dimensions and "labels" in dimensions[dimension]:
            return dimensions[dimension]["labels"][label]
        raise TaxonomyError(f"unknown {dimension} label: {label}")


def load_taxonomy(path: str | Path) -> Taxonomy:
    path = Path(path).resolve()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TaxonomyError(f"cannot read taxonomy {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TaxonomyError(f"invalid YAML in {path}: {exc}") from exc

    document = _mapping(document, "taxonomy document")
    if str(document.get("schema_version")) != "1.0":
        raise TaxonomyError("schema_version must be 1.0")
    metadata = _mapping(document.get("taxonomy"), "taxonomy")
    taxonomy_id = metadata.get("id")
    version = metadata.get("version")
    if not isinstance(taxonomy_id, str) or not taxonomy_id:
        raise TaxonomyError("taxonomy.id must be a non-empty string")
    if not isinstance(version, str) or not version:
        raise TaxonomyError("taxonomy.version must be a non-empty string")

    statuses = frozenset(_mapping(document.get("classification_statuses"), "classification_statuses"))
    required_statuses = {"labeled", "unknown", "ambiguous", "not_applicable", "error"}
    if not required_statuses.issubset(statuses):
        raise TaxonomyError(f"missing classification statuses: {sorted(required_statuses - statuses)}")

    dimensions = _mapping(document.get("dimensions"), "dimensions")
    required_dimensions = {"expert_domains", "programming_languages"}
    missing = required_dimensions.difference(dimensions)
    if missing:
        raise TaxonomyError(f"missing dimensions: {sorted(missing)}")

    language_labels = dimensions["programming_languages"].get("canonical_labels")
    if not isinstance(language_labels, list) or not language_labels:
        raise TaxonomyError("dimensions.programming_languages.canonical_labels must be a non-empty list")
    if len(language_labels) != len(set(language_labels)):
        raise TaxonomyError("programming language labels must be unique")

    labels: dict[str, frozenset[str]] = {}
    dimension_experts: dict[str, str] = {}
    for dimension, definition_value in dimensions.items():
        definition = _mapping(definition_value, f"dimensions.{dimension}")
        if dimension == "programming_languages":
            labels[dimension] = frozenset(language_labels)
        elif "labels" in definition:
            labels[dimension] = _labels(definition, f"dimensions.{dimension}")
        elif "groups" in definition:
            labels[dimension] = _grouped_labels(definition, f"dimensions.{dimension}")
        elif "dimensions" in definition:
            children = _mapping(
                definition.get("dimensions"), f"dimensions.{dimension}.dimensions"
            )
            for child, child_value in children.items():
                child_definition = _mapping(
                    child_value, f"dimensions.{dimension}.dimensions.{child}"
                )
                if child_definition.get("cardinality") == "multi_open_vocabulary":
                    continue
                labels[f"{dimension}.{child}"] = _labels(
                    child_definition, f"dimensions.{dimension}.dimensions.{child}"
                )
        applies_to = definition.get("applies_when_expert_domain")
        if applies_to is not None:
            if not isinstance(applies_to, str):
                raise TaxonomyError(
                    f"dimensions.{dimension}.applies_when_expert_domain must be a string"
                )
            dimension_experts[dimension] = applies_to

    for dimension, applies_to in dimension_experts.items():
        if applies_to not in labels["expert_domains"]:
            raise TaxonomyError(
                f"dimensions.{dimension}.applies_when_expert_domain must name an expert label"
            )

    judge_definition = document.get("semantic_judge")
    if judge_definition is None:
        defaults = ("expert_domains", "task_domains")
        semantic_judge_dimensions = tuple(
            dimension for dimension in defaults if dimension in labels
        )
    else:
        judge_definition = _mapping(judge_definition, "semantic_judge")
        judge_dimensions = judge_definition.get("dimensions")
        if not isinstance(judge_dimensions, list) or not judge_dimensions:
            raise TaxonomyError("semantic_judge.dimensions must be a non-empty list")
        if not all(isinstance(dimension, str) for dimension in judge_dimensions):
            raise TaxonomyError("semantic_judge.dimensions must contain only strings")
        if len(judge_dimensions) != len(set(judge_dimensions)):
            raise TaxonomyError("semantic_judge.dimensions must be unique")
        unknown_judge_dimensions = set(judge_dimensions).difference(labels)
        if unknown_judge_dimensions:
            raise TaxonomyError(
                "unknown semantic judge dimensions: "
                f"{sorted(unknown_judge_dimensions)}"
            )
        semantic_judge_dimensions = tuple(judge_dimensions)

    return Taxonomy(
        path=path,
        document=document,
        taxonomy_id=taxonomy_id,
        version=version,
        statuses=statuses,
        labels=labels,
        semantic_judge_dimensions=semantic_judge_dimensions,
        dimension_experts=dimension_experts,
    )
