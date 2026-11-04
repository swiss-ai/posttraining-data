"""Repairs for tool schemas damaged by the way a source release was stored.

Each repair is scoped to the repository that shows the damage, changes nothing but the
damaged schema nodes, and returns a record of what it changed. Calls are still checked
against the repaired schema, so a call that uses a removed property is still rejected.

  nvidia/Nemotron-SWE-v1       The tool schemas were stored as one Arrow struct shared by
                               all tools. Every property of every tool exists in every
                               tool's schema; the ones a tool does not define are null
                               (`"thought": null`, `"enum": null`). A null schema node is
                               an absent definition, so it is removed.
  openbmb/UltraData-SFT-Agent-2609
                               The tool schemas were serialised with a JSON encoder that
                               replaced repeated nodes with the string "[Circular]". The
                               original node is not recoverable, so it becomes the
                               unconstrained schema {}: no type or property is invented.
"""

import copy

ARROW_NULL_REPOS = {"nvidia/Nemotron-SWE-v1"}
CIRCULAR_REPOS = {"openbmb/UltraData-SFT-Agent-2609"}
CIRCULAR = "[Circular]"
# Keywords for which null is a real value and not an absent definition
NULL_IS_VALUE = {"default", "const", "example", "examples"}
SCHEMA_VALUED = {"items", "additionalProperties", "not", "if", "then", "else", "contains"}


def drop_null_nodes(node, path, removed):
    if isinstance(node, dict):
        in_properties = path.endswith("/properties")
        result = {}
        for key, value in node.items():
            if value is None and (in_properties or key not in NULL_IS_VALUE):
                removed.append(f"{path}/{key}")
                continue
            result[key] = drop_null_nodes(value, f"{path}/{key}", removed)
        return result
    if isinstance(node, list):
        return [drop_null_nodes(v, f"{path}[{i}]", removed) for i, v in enumerate(node)]
    return node


def replace_circular(node, path, replaced, schema_position=True):
    if isinstance(node, str):
        if node == CIRCULAR and schema_position:
            replaced.append(path)
            return {}
        return node
    if isinstance(node, dict):
        in_properties = path.endswith("/properties")
        return {
            key: replace_circular(
                value,
                f"{path}/{key}",
                replaced,
                schema_position=in_properties or key in SCHEMA_VALUED,
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [
            replace_circular(v, f"{path}[{i}]", replaced, schema_position)
            for i, v in enumerate(node)
        ]
    return node


def repair_tool_schema(repo, name, schema):
    """Return (schema, repairs); the schema is untouched when the repo is not listed."""
    if repo in ARROW_NULL_REPOS:
        removed = []
        fixed = drop_null_nodes(copy.deepcopy(schema), name, removed)
        if removed:
            return fixed, [
                {
                    "kind": "removed_null_schema_nodes",
                    "tool": name,
                    "count": len(removed),
                    "nodes": removed,
                    "reason": "null marks a node the tool does not define in a schema shared by all tools",
                }
            ]
    if repo in CIRCULAR_REPOS:
        replaced = []
        fixed = replace_circular(copy.deepcopy(schema), name, replaced)
        if replaced:
            return fixed, [
                {
                    "kind": "replaced_circular_schema_nodes",
                    "tool": name,
                    "nodes": replaced,
                    "replacement": "{} (unconstrained); the original node is not recoverable",
                }
            ]
    return schema, []
