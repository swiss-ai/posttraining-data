"""Reject unresolved tool provenance and reasoning boundaries."""


def unresolved(result):
    from openai_reasoning import result_reasoning_errors

    blockers = result_reasoning_errors(result)
    if any(
        t["origin"] == "observed_signature_only"
        for t in result.get("tool_provenance", [])
    ):
        blockers.append("original_tool_schema_unavailable")
    for warning in result.get("warnings", []):
        if "Parallel result IDs assigned by source order" in warning:
            blockers.append("parallel_result_association_unverified")
        if "execution batch inferred" in warning:
            blockers.append("executed_code_selection_unrecorded")
        if "Unclosed source think block" in warning:
            blockers.append("unclosed_reasoning_boundary")
        if "World-model task" in warning:
            blockers.append("separate_world_model_task")
    return sorted(set(blockers))
