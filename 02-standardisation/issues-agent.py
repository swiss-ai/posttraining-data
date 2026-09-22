import re
import json
import argparse
import requests
from tqdm import tqdm
from typing import List
from dataclasses import dataclass, asdict
from datasets import Dataset, load_from_disk

from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Conversation,
    DeveloperContent,
    SystemContent,
    Message,
    Role,
    ToolDescription,
    HarmonyEncoding,
    TextContent,
    Author,
)

SUBSETS = [
    "cn_contest"
]

FUNCTION_TOOLS = [
    ToolDescription.new(
        name="browse_samples",
        description="Browse a list of samples from the dataset",
        parameters={
            "type": "object",
            "properties": {
                "num_samples": {
                    "type": "number",
                    "description": "The number of samples to browse",
                    "default": 1,
                },
                "start_idx": {
                    "type": "number",
                    "description": "The index of the first sample to browse (sample id)",
                    "default": 0,
                },
                "subset": {
                    "type": "string",
                    "description": "The subset of samples to browse",
                    "default": "all",
                    "enum": SUBSETS,
                },
                "filter": {
                    "type": "string",
                    "description": "A regex filter to select the samples. The regex is applied to the `initial_prompt`.",
                    "default": None,
                },
            },
            "required": ["num_samples", "start_idx", "subset", "filter"],
        },
    ),
    ToolDescription.new(
        name="record_issue",
        description="Record an issue found in the dataset. Use this tool to document issues you discover while browsing samples. Each issue should describe either: (1) content that should be removed (e.g., metadata after the problem, section headers, formatting artifacts, content before/after the problem statement), or (2) content that was incorrectly removed and should be restored (e.g., legitimate problem text that was deleted). Focus on structural artifacts around the text, not mathematical correctness.",
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A clear description of the issue. For content to remove: 'remove \"Time allowed: 270 minutes. Each problem is worth 10 points. ## SOLUTIONS\" and everything after it - this metadata appears after the problem statement ends', 'remove formatting tags [u] and [b]', 'remove everything after the separator pattern ---'. For content incorrectly removed: 'the condition \"such that abc = 1\" was incorrectly removed - this is part of the problem statement and should be kept', 'the text \"for all m, n ∈ ℕ\" was incorrectly removed - this is an essential part of the problem statement', 'the number \"14\" in \"14 players\" was incorrectly removed - this is part of the problem statement describing the setup, not metadata'",
                },
                "subset": {
                    "type": "string",
                    "description": "The subset where this issue was found",
                    "enum": [
                        "olympiads_ref",
                        "aops_forum",
                        "cn_contest",
                        "inequalities",
                        "cn_k12",
                        "number_theory",
                    ],
                },
                "sample_ids": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "An array of sample IDs (indices within the subset) where this issue was observed. These are the relative indices within the filtered subset, not global dataset indices.",
                },
            },
            "required": ["description", "subset", "sample_ids"],
        },
    ),
]


@dataclass
class Issue:
    description: str
    subset: str
    samples_ids: List[int]
    samples_content: List[str]


DEVELOPER_INSTRUCTIONS = """Your task is to find ALL issues in a cleaned OCR dataset. The dataset has been processed by a cleaner script, but there may still be artifacts or cleaning errors that need to be identified.

CRITICAL FOCUS: Your PRIMARY goal is to identify TWO types of issues:
1. Content that appears AFTER or AROUND the actual problem statement that should be removed, WITHOUT changing the problem text itself
2. Content that was INCORRECTLY REMOVED - legitimate problem text that the cleaning script removed but should have been kept

You are looking for structural artifacts and metadata that surround the problem, not issues with the mathematical content or LaTeX formatting within the problem.

You will be given a specific subset to analyze. Your task is to:
1. Browse samples from the given subset systematically - you MUST review samples from index 100 to index 300 (samples 100-299, which is 200 samples total)
2. Browse samples in batches of 20 at a time (use num_samples=20 and increment start_idx by 20 for each batch: start_idx=100, then 120, then 140, then 160, continuing until you reach start_idx=300)
3. Compare the "BEFORE CLEANING" (original_problem) with the "AFTER CLEANING" (cleaned content) for each sample with EXTREME ATTENTION TO DETAIL
4. Identify where the problem statement ENDS and look for content that appears AFTER it that should be removed, including:
   - Metadata lines after the problem (e.g., "Time allowed: 270 minutes", "Each problem is worth 10 points", "Points: 10", etc.)
   - Section headers that follow the problem (e.g., "## SOLUTIONS", "## Solution", "SOLUTIONS", "Answer:", etc.)
   - Separator patterns followed by unrelated content (e.g., "---", "===", repeated dashes, etc. followed by solutions, answers, or other problems)
   - Translation metadata that appears after the problem statement
   - Contest information, problem numbers, or other metadata that appears after the problem
   - Any content that is clearly not part of the problem statement itself
5. Also identify content that appears BEFORE the problem that should be removed:
   - Example numbers, problem numbers, or identifiers at the start
   - Contest names or metadata headers
   - Separator patterns before the problem
6. Identify formatting artifacts that weren't removed:
   - Formatting tags like [hide], [u], [b], [i], [list], [quote], [url], [color], [size], [font], [center], [align], [img], [attachment], [code], [math], [latex], [equation], [spoiler], [strike], [s], [sub], [sup], [table], [tr], [td], [th], [hr], [br], etc.
   - HTML/BBcode markup
7. CRITICALLY IMPORTANT: Identify cases where the cleaning script REMOVED TOO MUCH - legitimate problem content that was incorrectly deleted:
   - Legitimate problem text that was removed (e.g., parts of the problem statement, important conditions, constraints, or question text)
   - Numbers that are part of the problem statement but were removed (e.g., "14 players", "3 teams", "5 coins" - these numbers describe the problem setup and must be kept, unlike problem numbers like "Problem 14" which are metadata)
   - Mathematical expressions, equations, or formulas that are part of the problem but were removed
   - Text that was part of the problem but was incorrectly identified as metadata and removed
   - Lines or paragraphs that were part of the problem statement but were removed
   - Any content from the "BEFORE CLEANING" that should have been kept but is missing in "AFTER CLEANING"
8. Use the record_issue tool to document EVERY issue you find, including:
   - A clear description of what content should be removed/restored and where it appears (before/after the problem)
   - The subset name
   - The sample IDs (relative indices within the subset) where the issue was observed

IMPORTANT: You must browse samples from index 100 to index 300 (samples 100-299, which is 200 samples total). Browse them in batches of 20 samples at a time (num_samples=20). Start at start_idx=100 and continue browsing in batches of 20 until you reach start_idx=300 (you will browse: 100-119, 120-139, 140-159, 160-179, 180-199, 200-219, 220-239, 240-259, 260-279, 280-299).

MOST IMPORTANT: Focus on identifying TWO types of issues:

Type 1 - Content that should be REMOVED (not part of the problem):
- Content that appears AFTER the problem ends (metadata, solutions headers, etc.)
- Content that appears BEFORE the problem starts (example numbers, headers, etc.)
- Formatting artifacts that weren't cleaned
- Structural elements that indicate the end of the problem (like separators followed by unrelated content)

Type 2 - Content that was INCORRECTLY REMOVED (should be kept):
- Legitimate problem text that was removed and should be restored
- Numbers that are part of the problem statement (e.g., "14 players", "3 teams", "5 coins") that were incorrectly removed - distinguish these from problem numbers/metadata (e.g., "Problem 14") which should be removed
- Mathematical expressions, equations, or formulas that are part of the problem but were deleted
- Text that was part of the problem but was incorrectly identified as metadata and removed
- Lines or paragraphs from the problem statement that are missing in the cleaned version
- Any content that exists in "BEFORE CLEANING" but is missing in "AFTER CLEANING" and should have been kept

When comparing BEFORE vs AFTER, ask yourself:
- Is there content in BEFORE that's missing in AFTER that should have been kept? → Record as "too much removed"
- Is there content in AFTER that wasn't in BEFORE? → Usually not an issue (unless it's an artifact)
- Is there content in AFTER that should be removed? → Record as "should be removed"

Do NOT focus on:
- Mathematical correctness or logic within the problem
- LaTeX formatting issues within the problem text itself (unless they're artifacts that should be removed)
- Whether the problem makes mathematical sense

Below are examples of the types of issues you should look for:

Example 1 - Content after problem:
Problem ends with: "for all m, n ∈ ℕ. (Bulgaria)"
Then follows: "Time allowed: 270 minutes. Each problem is worth 10 points. ## SOLUTIONS"
Issue: "Remove 'Time allowed: 270 minutes. Each problem is worth 10 points. ## SOLUTIONS' and everything after it - this metadata appears after the problem statement ends"

Example 2 - Separator patterns:
Problem ends, then: "---" followed by solution or answer
Issue: "Remove everything after the separator pattern '---' (or '===') - content after separators is not part of the problem"

Example 3 - Section headers:
Problem ends, then: "## SOLUTIONS" or "SOLUTIONS:" or "Answer:"
Issue: "Remove section headers like '## SOLUTIONS' or 'Answer:' that appear after the problem statement"

Example 4 - Formatting tags:
Problem contains: "[b]text[/b]" or "[u]text[/u]"
Issue: "Remove formatting tags like [b], [u], [i] - these are BBcode artifacts that should be cleaned"

Example 5 - Translation metadata:
Problem ends, then: "Translation: ..." or lines with translation information
Issue: "Remove translation metadata that appears after the problem statement"

Example 6 - Too much removed:
BEFORE CLEANING contains: "Let a, b, c be positive real numbers such that abc = 1. Prove that..."
AFTER CLEANING contains: "Let a, b, c be positive real numbers. Prove that..."
Issue: "The condition 'such that abc = 1' was incorrectly removed - this is part of the problem statement and should be kept"

Example 7 - Too much removed:
BEFORE CLEANING contains: "Problem 1. Find all functions f: ℝ → ℝ such that..."
AFTER CLEANING contains: "Find all functions f: ℝ → ℝ such that..."
Issue: "The problem number 'Problem 1.' was correctly removed, but if any part of the actual problem statement was removed, that would be an issue"

Example 8 - Too much removed:
BEFORE CLEANING contains: "Determine all functions f: ℕ → ℕ satisfying f(f(m)² + 2f(n)²) = m² + 2n² for all m, n ∈ ℕ."
AFTER CLEANING contains: "Determine all functions f: ℕ → ℕ satisfying f(f(m)² + 2f(n)²) = m² + 2n²."
Issue: "The condition 'for all m, n ∈ ℕ' was incorrectly removed - this is an essential part of the problem statement and must be kept"

Example 9 - Too much removed (numbers in problem statement):
BEFORE CLEANING contains: "14 players are participating in a tournament..."
AFTER CLEANING contains: "players are participating in a tournament..."
Issue: "The number '14' was incorrectly removed - this is part of the problem statement (describing how many players), not metadata. Numbers that are part of the problem content must be kept"

Browse through multiple samples to identify where problems end and what content follows. Pay EXTREME attention to:
- Where the problem statement naturally concludes (usually ends with a question mark, period, or mathematical statement)
- What appears immediately after the problem ends (metadata, headers, separators, solutions)
- What appears before the problem starts (example numbers, headers, metadata)
- Patterns that indicate the end of a problem (separators, section headers, metadata lines)
- Formatting artifacts that weren't removed
- Structural elements that clearly separate the problem from other content
- CRITICAL: Compare BEFORE and AFTER line by line to identify any legitimate problem content that was removed

Remember: Your goal is to identify TWO types of issues:
1. Artifacts AROUND the text (before/after) and formatting artifacts that should be REMOVED
2. Legitimate problem content that was INCORRECTLY REMOVED and should be RESTORED

Do NOT evaluate the mathematical content itself. You must browse samples from index 100 to index 300 (samples 100-299, which is 200 samples total) in batches of 20. Be COMPREHENSIVE - record EVERY issue you encounter (both things that should be removed AND things that were incorrectly removed). Once you've thoroughly reviewed samples 100-299 and recorded ALL issues you've found, you can conclude."""


def browse_samples(
    dataset: Dataset,
    num_samples: int = 1,
    start_idx: int = 0,
    subset: str = "all",
    filter: str = None,
) -> str:
    def get_initial_prompt_content(sample):
        initial_prompt = sample.get("initial_prompt", {})
        if isinstance(initial_prompt, dict):
            return initial_prompt.get("content", "")
        return str(initial_prompt)

    def get_original_problem(sample):
        initial_prompt = sample.get("initial_prompt", {})
        if isinstance(initial_prompt, dict):
            metadata = initial_prompt.get("metadata", {})
            return metadata.get("original_problem", "")
        return ""

    def check_filter_match(example, idx, subset, filter_pattern):
        if subset != "all":
            original_metadata = example.get("original_metadata", {})
            dataset_source = example.get("dataset_source", "")
            category = original_metadata.get("category", "")
            source = original_metadata.get("source", "")

            if subset not in [category, source, dataset_source]:
                return {"filter_match": False, "original_idx": idx}

        if filter_pattern:
            regex = re.compile(filter_pattern)
            initial_prompt = example.get("initial_prompt", {})
            if isinstance(initial_prompt, dict):
                content = initial_prompt.get("content", "")
            else:
                content = str(initial_prompt)
            if not regex.search(content):
                return {"filter_match": False, "original_idx": idx}

        return {"filter_match": True, "original_idx": idx}

    filtered_dataset = dataset.map(
        check_filter_match,
        with_indices=True,
        num_proc=128,
        fn_kwargs={
            "subset": subset,
            "filter_pattern": filter,
        },
    ).filter(lambda x: x["filter_match"], num_proc=128)

    if len(filtered_dataset) == 0:
        filter_msg = f" matching filter '{filter}'" if filter else ""
        subset_msg = f" for subset '{subset}'" if subset != "all" else ""
        return f"No samples found{filter_msg}{subset_msg}"

    matching_indices = filtered_dataset["original_idx"]

    end_idx = min(start_idx + num_samples, len(matching_indices))
    selected_indices = matching_indices[start_idx:end_idx]

    lines = []
    for i, idx in enumerate(selected_indices):
        sample = dataset[idx]
        original_content = get_original_problem(sample)
        cleaned_content = get_initial_prompt_content(sample)

        relative_idx = start_idx + i
        lines.append(f"Sample {relative_idx}:")
        lines.append("--- BEFORE CLEANING ---")
        lines.append(
            original_content if original_content else "(no original content available)"
        )
        lines.append("--- AFTER CLEANING ---")
        lines.append(cleaned_content)
        if i < len(selected_indices) - 1:
            lines.append("")

    return "\n".join(lines)


def get_initial_prompt_content(sample):
    initial_prompt = sample.get("initial_prompt", {})
    if isinstance(initial_prompt, dict):
        return initial_prompt.get("content", "")
    return str(initial_prompt)


def get_sample_content_for_issue(
    dataset: Dataset,
    subset: str,
    sample_ids: List[int],
    filter: str = None,
) -> List[str]:
    """Retrieve the cleaned content for samples identified by their relative indices within a subset."""
    def get_original_problem(sample):
        initial_prompt = sample.get("initial_prompt", {})
        if isinstance(initial_prompt, dict):
            metadata = initial_prompt.get("metadata", {})
            return metadata.get("original_problem", "")
        return ""

    def check_filter_match(example, idx, subset, filter_pattern):
        if subset != "all":
            original_metadata = example.get("original_metadata", {})
            dataset_source = example.get("dataset_source", "")
            category = original_metadata.get("category", "")
            source = original_metadata.get("source", "")

            if subset not in [category, source, dataset_source]:
                return {"filter_match": False, "original_idx": idx}

        if filter_pattern:
            regex = re.compile(filter_pattern)
            initial_prompt = example.get("initial_prompt", {})
            if isinstance(initial_prompt, dict):
                content = initial_prompt.get("content", "")
            else:
                content = str(initial_prompt)
            if not regex.search(content):
                return {"filter_match": False, "original_idx": idx}

        return {"filter_match": True, "original_idx": idx}

    filtered_dataset = dataset.map(
        check_filter_match,
        with_indices=True,
        num_proc=128,
        fn_kwargs={
            "subset": subset,
            "filter_pattern": filter,
        },
    ).filter(lambda x: x["filter_match"], num_proc=128)

    if len(filtered_dataset) == 0:
        return []

    matching_indices = filtered_dataset["original_idx"]
    
    samples_content = []
    for sample_id in sample_ids:
        if 0 <= sample_id < len(matching_indices):
            idx = matching_indices[sample_id]
            sample = dataset[idx]
            original_content = get_original_problem(sample)
            cleaned_content = get_initial_prompt_content(sample)
            
            # Format as "BEFORE: ... AFTER: ..." for context
            content_str = f"BEFORE CLEANING:\n{original_content}\n\nAFTER CLEANING:\n{cleaned_content}"
            samples_content.append(content_str)
        else:
            samples_content.append(f"(Sample {sample_id} not found in subset)")
    
    return samples_content


def generate(
    encoding: HarmonyEncoding, conversation: Conversation, url: str
) -> List[Message]:
    input_ids = encoding.render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )

    total_context = 128000
    used_tokens = len(input_ids)
    usage_percent = (used_tokens / total_context) * 100
    print(
        f"Context usage: {used_tokens:,} / {total_context:,} tokens ({usage_percent:.2f}%)"
    )

    response = requests.post(
        url=f"{url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.4,
                "max_new_tokens": 4096,
                "skip_special_tokens": False,
            },
        },
    )

    output_ids = response.json()["output_ids"][:-1]

    new_messages = encoding.parse_messages_from_completion_tokens(
        output_ids, role=Role.ASSISTANT
    )
    return new_messages, encoding.decode(output_ids).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--output", type=str, default="issues.json", help="Output file path for the issues array (default: issues.json)")
    args = parser.parse_args()

    try:
        import multiprocessing
        if hasattr(multiprocessing, "set_start_method"):
            try:
                multiprocessing.set_start_method("spawn", force=True)
            except (RuntimeError, ValueError):
                pass
    except ImportError:
        pass

    dataset = load_from_disk(args.dataset)["train"]
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    all_issues = []

    for subset in tqdm(SUBSETS, desc="Processing subsets"):
        print(f"\n{'='*80}")
        print(f"Processing subset: {subset}")
        print(f"{'='*80}\n")

        conversation = Conversation.from_messages(
            [
                Message.from_role_and_content(
                    Role.SYSTEM,
                    SystemContent.new(),
                ),
                Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new()
                    .with_instructions(DEVELOPER_INSTRUCTIONS)
                    .with_function_tools(FUNCTION_TOOLS),
                ),
                Message.from_role_and_content(
                    Role.USER,
                    TextContent(text=f"Please analyze the '{subset}' subset and identify ALL issues with the cleaning process. Focus on TWO types of issues: (1) Content that appears BEFORE or AFTER the problem statement that should be removed (like metadata, section headers, separators followed by unrelated content), as well as formatting artifacts, and (2) Content that was INCORRECTLY REMOVED - legitimate problem text that should have been kept but was deleted by the cleaning script. Compare the BEFORE and AFTER versions carefully to identify both types of issues. Do NOT focus on mathematical correctness or LaTeX formatting within the problem text itself. You MUST be exhaustive and record EVERY issue you find. You must browse samples from index 100 to index 300 (samples 100-299, which is 200 samples total) in batches of 20 samples at a time. Browse through samples systematically starting at start_idx=100 with num_samples=20, then start_idx=120 with num_samples=20, then start_idx=140, continuing until you reach start_idx=300 (you will browse: 100-119, 120-139, 140-159, 160-179, 180-199, 200-219, 220-239, 240-259, 260-279, 280-299) and record ALL issues you find using the record_issue tool. Remember: your goal is to identify structural artifacts around the text, formatting issues, AND cases where too much legitimate content was removed."),
                ),
            ]
        )

        subset_issues = []

        needs_to_continue = True
        while needs_to_continue:
            needs_to_continue = False
            new_messages, new_text = generate(encoding, conversation, args.url)

            cleaned_messages = []
            for message in new_messages:
                if message.recipient and not message.recipient.startswith("functions"):
                    message.recipient = None
                message.content_type = None

                if all(c.text.strip() == "" for c in message.content):
                    continue
                cleaned_messages.append(message)

            conversation.messages.extend(cleaned_messages)

            for message in cleaned_messages:
                print(message)
                if message.author.role == Role.ASSISTANT:
                    if message.recipient and message.recipient.startswith("functions"):
                        needs_to_continue = True
                        function_name = message.recipient.split(".")[-1]
                        function_result = None

                        for content in [
                            c for c in message.content if isinstance(c, TextContent)
                        ]:
                            function_arguments = json.loads(content.text)

                            try:
                                if function_name == "browse_samples":
                                    function_result = browse_samples(
                                        dataset,
                                        num_samples=function_arguments.get(
                                            "num_samples", 1
                                        ),
                                        start_idx=function_arguments.get(
                                            "start_idx", 0
                                        ),
                                        subset=function_arguments.get("subset", "all"),
                                        filter=function_arguments.get("filter"),
                                    )
                                elif function_name == "record_issue":
                                    # Get the subset and sample IDs
                                    issue_subset = function_arguments.get("subset", subset)
                                    sample_ids = function_arguments.get("sample_ids", [])
                                    
                                    # Retrieve the actual sample content
                                    samples_content = get_sample_content_for_issue(
                                        dataset,
                                        subset=issue_subset,
                                        sample_ids=sample_ids,
                                        filter=None,  # We don't track filters per issue, use subset only
                                    )
                                    
                                    # Store the issue
                                    issue = Issue(
                                        description=function_arguments.get("description", ""),
                                        subset=issue_subset,
                                        samples_ids=sample_ids,
                                        samples_content=samples_content,
                                    )
                                    subset_issues.append(issue)
                                    function_result = f"Recorded issue: {issue.description} (affecting {len(issue.samples_ids)} sample(s))"
                                else:
                                    function_result = (
                                        f"Unknown function: {function_name}"
                                    )
                            except Exception as e:
                                function_result = (
                                    f"Error calling {function_name}: {str(e)}"
                                )

                        if function_result is None:
                            function_result = ""

                        if not isinstance(function_result, str):
                            function_result = (
                                json.dumps(function_result) if function_result else ""
                            )

                        tool_message = (
                            Message.from_author_and_content(
                                Author.new(Role.TOOL, message.recipient),
                                TextContent(text=function_result),
                            )
                            .with_channel(message.channel)
                            .with_recipient("assistant")
                        )

                        print(tool_message)

                        conversation.messages.append(tool_message)

        # Add issues from this subset to the overall list
        all_issues.extend(subset_issues)
        print(f"\nFound {len(subset_issues)} issue(s) in subset '{subset}'")

    # Save all issues to a file
    print(f"\n{'='*80}")
    print(f"Total issues found: {len(all_issues)}")
    print(f"{'='*80}\n")

    # Convert issues to dictionaries for JSON serialization
    issues_dict = [asdict(issue) for issue in all_issues]

    with open(args.output, "w") as f:
        json.dump(issues_dict, f, indent=2)

    print(f"Issues saved to {args.output}")
    
    # Also print the issues array in Python format for easy copy-paste
    print("\n" + "="*80)
    print("ISSUES ARRAY (Python format):")
    print("="*80)
    print("\nINITIAL_ISSUES = [")
    for issue in all_issues:
        print(f'    Issue(')
        print(f'        description="{issue.description}",')
        print(f'        subset="{issue.subset}",')
        print(f'        samples_ids={issue.samples_ids},')
        print(f'        samples_content={issue.samples_content},')
        print(f'    ),')
    print("]")
    print("="*80)
