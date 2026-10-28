"""Shared loss-preserving mechanics; each dataset owns a separate adapter module."""


def prepare_fields(row, sql_context=False, instruction_only=False):
    row = dict(row)
    if "old_contents" in row and "new_contents" in row:
        row["prompt"] = (
            str(row.get("subject") or row.get("message") or "")
            + "\n\nFile: "
            + str(row.get("old_file", ""))
            + "\n\n"
            + str(row["old_contents"])
        )
        row["response"] = row["new_contents"]
    elif "sql_prompt" in row:
        row["prompt"] = (
            str(row["sql_prompt"]) + "\n\n" + str(row.get("sql_context", ""))
        )
        row["response"] = row.get("sql")
    elif sql_context:
        row["prompt"] = (
            str(row.get("question", "")) + "\n\n" + str(row.get("context", ""))
        )
    elif instruction_only:
        row["prompt"] = row.get("instruction")
    elif row.get("instruction") and row.get("input"):
        row["prompt"] = str(row["instruction"]) + "\n\n" + str(row["input"])
    if row.get("starter_code") and row.get("question"):
        row["prompt"] = row["question"] + "\n\nStarter code:\n" + row["starter_code"]
    if row.get("r1_generation"):
        row["response"] = row["r1_generation"]
    return row


def benchmark_fields(row, entry, source_file):
    result = dict(row)

    def fields(name):
        return [
            row[k]
            for k in entry.get(name, [])
            if isinstance(row.get(k), str) and row[k].strip()
        ]

    prompts = fields("prompt_fields")
    if not prompts:
        raise ValueError(
            f"No prompt field for {entry['repo']} {source_file}: {list(row)}"
        )
    result["prompt"] = prompts[0]
    answers = fields("response_fields")
    if answers:
        result["response"] = answers[0]
    tests = {k: row[k] for k in entry.get("test_fields", []) if k in row}
    if tests:
        result["tests"] = tests
    result["benchmark_prompts"] = prompts
    result["benchmark_source_file"] = source_file
    return result


class DatasetPreprocessor:
    """Fallback for uncatalogued sources; registered datasets override preparation."""

    repo = None
    policy = "code_source"
    description = "Loss-preserving unified normalization."

    def select(self, row, entry, file):
        from code_policy import decision

        policy = self.policy if self.repo else entry.get("code_policy", self.policy)
        return decision(row, dict(entry, code_policy=policy), file)

    def configure(self, entry):
        entry = dict(entry)
        if self.repo:
            entry["code_policy"] = self.policy
            if self.policy == "terminal_conversations":
                entry["code_policy_version"] = "terminal-conversations-2"
        return entry

    def prepare(self, row, entry, file):
        if entry["category"] == "benchmarks":
            row = benchmark_fields(row, entry, file["path"])
        return prepare_fields(row)

    def normalize(self, row, entry, file, index):
        from unified import normalize_v2_shared

        return normalize_v2_shared(row, entry, file, index, self)
