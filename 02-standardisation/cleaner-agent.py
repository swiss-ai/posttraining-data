import os
import re
import sys
import json
import shutil
import tempfile
import argparse
import requests
import importlib
import subprocess
import importlib.util
from tqdm import tqdm
from typing import List
from dataclasses import dataclass
from datasets import disable_progress_bars, enable_progress_bars
from datasets import Dataset, load_from_disk, concatenate_datasets

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
                    "enum": [
                        "all",
                        "olympiads",
                        "olympiads_ref",
                        "aops_forum",
                        "cn_contest",
                        "inequalities",
                        "cn_k12",
                        "number_theory",
                    ],
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
        name="run_cleaner_script",
        description="Run the cleaner script on the dataset to process and clean the data. This tool supports two modes: fast mode for quick iteration and validation, and normal mode for final processing and saving.\n\nWorkflow:\n1. After editing the cleaner script (using edit_cleaner_script), use fast=True to quickly process the dataset and see a summary of changes without saving.\n2. Review the diff summary and browse samples to verify the changes are correct.\n3. Once satisfied with the changes, use fast=False to process the dataset and save it to disk (this is the final step).\n\nThe tool automatically reloads the cleaner script module before execution to ensure the latest changes are applied.",
        parameters={
            "type": "object",
            "properties": {
                "fast": {
                    "type": "boolean",
                    "description": "Fast mode (True): Process the dataset to a temporary location and output a summary of diffs between the processed dataset and the original one. This is much faster as it doesn't save to disk. Use this immediately after editing the cleaner script to quickly validate changes. Normal mode (False): Process the dataset and save it to disk. Use this as the final step after confirming the fast mode results are correct.",
                    "default": False,
                },
            },
            "required": ["fast"],
        },
    ),
    ToolDescription.new(
        name="edit_cleaner_script",
        description="This is a tool for making multiple edits to the cleaner script in one operation. It allows you to perform multiple find-and-replace operations efficiently. Prefer this tool when you need to make multiple edits to the cleaner script.\n\nBefore using this tool:\n\n1. Use the read_cleaner_script tool to understand the file's contents and context\n2. Verify the file path is correct\n\nTo make multiple file edits, provide the following:\n1. edits: An array of edit operations to perform, where each edit contains:\n   - old_string: The text to replace (must match the file contents EXACTLY, including all whitespace, indentation, newlines, and special characters)\n   - new_string: The edited text to replace the old_string\n   - replace_all: Replace all occurences of old_string. This parameter is optional and defaults to false.\n\n⚠️ CRITICAL: old_string MUST MATCH EXACTLY ⚠️\n- The old_string parameter MUST be copied EXACTLY as it appears in the document\n- This includes: exact whitespace (spaces, tabs), exact indentation, exact newlines, exact punctuation, exact capitalization\n- If old_string does not match EXACTLY character-by-character, the tool will FAIL\n- ALWAYS use read_cleaner_script first to get the exact text from the file\n- Copy the text directly from the file output - do not paraphrase or modify it\n- Even a single character difference (extra space, wrong indentation, different newline) will cause failure\n\nIMPORTANT:\n- All edits are applied in sequence, in the order they are provided\n- Each edit operates on the result of the previous edit\n- All edits must be valid for the operation to succeed - if any edit fails, none will be applied\n- This tool is ideal when you need to make several changes to different parts of the same file\n\nCRITICAL REQUIREMENTS:\n1. The edits are atomic - either all succeed or none are applied\n2. Plan your edits carefully to avoid conflicts between sequential operations\n3. ALWAYS verify old_string matches exactly by reading the file first\n\nWARNING:\n- The tool will FAIL if edits.old_string doesn't match the file contents EXACTLY (including whitespace, tabs, newlines, indentation)\n- The tool will fail if edits.old_string and edits.new_string are the same\n- Since edits are applied in sequence, ensure that earlier edits don't affect the text that later edits are trying to find\n- If you're unsure about the exact formatting, use read_cleaner_script to get the precise text",
        parameters={
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {
                                "type": "string",
                                "description": "⚠️ CRITICAL: The text to replace. This MUST match the file contents EXACTLY character-by-character, including all whitespace (spaces, tabs), indentation, newlines, and punctuation. If it doesn't match exactly, the operation will fail. Always use read_cleaner_script first to get the exact text from the file.",
                            },
                            "new_string": {
                                "type": "string",
                                "description": "The text to replace it with",
                            },
                            "replace_all": {
                                "type": "boolean",
                                "default": False,
                                "description": "Replace all occurences of old_string (default false).",
                            },
                        },
                        "required": ["old_string", "new_string"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "description": "Array of edit operations to perform sequentially on the file",
                },
            },
            "required": ["edits"],
        },
    ),
    ToolDescription.new(
        name="search",
        description="Search for text in the cleaner script. Returns matching lines with their line numbers.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to search for in the cleaner script.",
                },
                "path": {
                    "type": "string",
                    "description": "The path to the file to search (defaults to the cleaner script path). Can be empty string to use default.",
                    "default": "",
                },
            },
            "required": ["text", "path"],
        },
    ),
    ToolDescription.new(
        name="open_file",
        description="Read a specific portion of the cleaner script by line range, or the entire file. Useful for focusing on specific sections of the code or reading the whole file.",
        parameters={
            "type": "object",
            "properties": {
                "line_start": {
                    "type": "number",
                    "description": "The starting line number (1-indexed). If not provided or -1, starts from the beginning. Set both line_start and line_end to -1 to read all lines.",
                },
                "line_end": {
                    "type": "number",
                    "description": "The ending line number (1-indexed, inclusive). If not provided or -1, reads to the end. Set both line_start and line_end to -1 to read all lines.",
                },
            },
            "required": [],
        },
    ),
]


@dataclass
class Issue:
    description: str
    subset: str
    samples_ids: List[int]
    sample_contents: List[str]


with open("issues-10.json", "r") as f:
    INITIAL_ISSUES = json.load(f)

DEVELOPER_INSTRUCTIONS = f"Your task is to improve a cleaner script for an OCR dataset: `/users/nathanrchn/capstor_reasoning/dev/posttraining-data/02-standardisation/convert-numinamath.py`. The dataset it produces still have OCR artifacts that need to be removed.\n\nYou need to fix all the issues listed below. IMPORTANT: You must continue working until all issues are fully fixed. Do not stop until you have verified that every single issue has been resolved.\n\nYour workflow should be:\n1. Browse samples for each issue to understand the problems\n2. Patch the cleaner script to fix all issues at once (consider how fixes might interact with each other)\n3. Run the cleaner script in fast mode to validate your changes across all issues\n4. Browse samples for each issue again to verify they are all fixed\n5. If any issues remain unfixed, iterate on your fixes and repeat steps 2-4 until all issues are resolved\n6. Once satisfied that all issues are resolved, run the cleaner script in normal mode to save the changes\n\nAll issues that need to be fixed:\n{json.dumps(INITIAL_ISSUES, indent=2)}"


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


def generate_diff_summary(original_dataset: Dataset, processed_dataset: Dataset) -> str:
    if len(original_dataset) != len(processed_dataset):
        return f"Warning: Dataset sizes differ. Original: {len(original_dataset)}, Processed: {len(processed_dataset)}"

    tmp_dir = tempfile.mkdtemp(prefix="diff_summary_")
    processed_tmp_path = os.path.join(tmp_dir, "processed")
    
    try:
        processed_dataset.save_to_disk(processed_tmp_path)
        processed_dataset_loaded = load_from_disk(processed_tmp_path)
        
        original_dataset_renamed = original_dataset.rename_columns(
            {col: f"original|{col}" for col in original_dataset.column_names}
        )
        processed_dataset_renamed = processed_dataset_loaded.rename_columns(
            {col: f"processed|{col}" for col in processed_dataset_loaded.column_names}
        )
        
        combined_dataset = concatenate_datasets([original_dataset_renamed, processed_dataset_renamed], axis=1)
        
        def compare_samples(sample):
            original_content = sample.get("original|initial_prompt", {}).get("content", "")
            processed_content = sample.get("processed|initial_prompt", {}).get("content", "")
            
            return {
                "original_content": original_content,
                "processed_content": processed_content,
                "diff_equals": original_content == processed_content,
            }
        
        comparison_dataset = combined_dataset.map(compare_samples, num_proc=128, keep_in_memory=True)
        
        changed_samples = comparison_dataset.filter(lambda x: not x["diff_equals"], num_proc=128)

        total_samples = len(original_dataset)
        changed_count = len(changed_samples)
        unchanged_count = total_samples - changed_count

        summary_lines = [
            f"Dataset Comparison Summary:",
            f"  Total samples: {total_samples}",
            f"  Changed samples: {changed_count}",
            f"  Unchanged samples: {unchanged_count}",
            f"  Change rate: {(changed_count / total_samples * 100):.2f}%",
            "",
        ]

        if changed_count > 0:
            summary_lines.append(f"Sample changes (showing first 25 of {changed_count}):")
            summary_lines.append("")
            for display_idx in range(min(25, changed_count)):
                sample = changed_samples[display_idx]
                original_content = sample["original_content"]
                processed_content = sample["processed_content"]
                summary_lines.append(f"Sample {display_idx}:")
                summary_lines.append("--- BEFORE CLEANING ---")
                summary_lines.append(
                    original_content if original_content else "(no original content available)"
                )
                summary_lines.append("--- AFTER CLEANING ---")
                summary_lines.append(processed_content)
                if display_idx < min(25, changed_count) - 1:
                    summary_lines.append("")

        return "\n".join(summary_lines)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_cleaner_script(
    script_path: str,
    dataset_path: str,
    script_module,
    dataset: Dataset,
    fast: bool = False,
) -> str:

    if fast:
        processed_dataset = script_module.process_split(dataset, num_proc=128)
        diff_summary = generate_diff_summary(original_dataset=dataset, processed_dataset=processed_dataset)

        return f"Fast mode completed successfully.\n\n{diff_summary}"
    else:
        disable_progress_bars()
        result = subprocess.run(
            ["python3", script_path, "-i", dataset_path, "-o", dataset_path],
            capture_output=True,
            text=True,
        )
        enable_progress_bars()
        if result.returncode != 0:
            return result.stderr
        return result.stdout


def edit_cleaner_script(file_path: str, edits: List[dict]) -> str:
    try:
        with open(file_path, "r") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

    original_content = content

    try:
        for i, edit in enumerate(edits):
            old_string = edit.get("old_string", "")
            new_string = edit.get("new_string", "")
            replace_all = edit.get("replace_all", False)

            if old_string == new_string:
                return f"Error in edit {i + 1}: old_string and new_string are the same"

            if old_string not in content:
                return f"Error in edit {i + 1}: old_string not found in file. The edit will fail if old_string doesn't match exactly (including whitespace)."

            if replace_all:
                content = content.replace(old_string, new_string)
            else:
                content = content.replace(old_string, new_string, 1)

        with open(file_path, "w") as f:
            f.write(content)

        return f"Successfully applied {len(edits)} edit(s) to the cleaner script."

    except Exception as e:
        try:
            with open(file_path, "w") as f:
                f.write(original_content)
        except:
            pass
        return f"Error applying edits: {str(e)}. No changes were made to the file."


def search_cleaner_script(query: str, script_path: str, path: str = "") -> str:
    file_path = (
        script_path
        if not path or path == "" or path == "convert_numinamath.py"
        else path
    )

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {str(e)}"

    matches = []
    for i, line in enumerate(lines, start=1):
        if query.lower() in line.lower():
            matches.append(f"{i}: {line.rstrip()}")

    if not matches:
        return f"No matches found for query: '{query}'"

    result = f"Found {len(matches)} match(es) for query '{query}':\n\n"
    result += "\n".join(matches)
    return result


def read_cleaner_script(
    script_path: str, line_start: int = None, line_end: int = None
) -> str:
    try:
        with open(script_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {str(e)}"

    total_lines = len(lines)

    if line_start == -1 and line_end == -1:
        selected_lines = lines
        start_idx = 0
        actual_end_line = total_lines
    else:
        start_idx = (
            (line_start - 1) if line_start is not None and line_start != -1 else 0
        )
        end_idx = line_end if line_end is not None and line_end != -1 else total_lines

        if start_idx < 0:
            start_idx = 0
        if end_idx > total_lines:
            end_idx = total_lines
        if start_idx >= end_idx:
            return f"Invalid line range: {line_start} to {line_end}"

        selected_lines = lines[start_idx:end_idx]
        actual_end_line = start_idx + len(selected_lines)

    result_lines = []
    for i, line in enumerate(selected_lines, start=start_idx + 1):
        result_lines.append(f"{i}: {line.rstrip()}")

    result = f"Lines {start_idx + 1}-{actual_end_line} from {script_path}:\n\n"
    result += "\n".join(result_lines)
    return result


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
                "temperature": 0.2,
                "max_new_tokens": 8192,
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
    parser.add_argument("--script", type=str, required=True)
    parser.add_argument("--url", type=str, required=True)
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

    script_path = os.path.abspath(args.script)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)
    module_name = os.path.splitext(script_name)[0].replace("-", "_")
    
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    script_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = script_module
    spec.loader.exec_module(script_module)
    
    script_module.__name__ = module_name
    script_module.__file__ = script_path

    dataset = load_from_disk(args.dataset)["train"]
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    issues = INITIAL_ISSUES.copy()

    if not issues:
        print("No issues found. Exiting.")
        sys.exit(0)

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
                TextContent(text=f"Please fix all the issues listed above. Work through them systematically: browse samples for each issue, make comprehensive changes to the cleaner script to address all issues, validate your changes, and continue iterating until all issues are fully fixed. Only save the changes once everything is completely resolved."),
            ),
        ]
    )

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
                            elif function_name == "run_cleaner_script":
                                fast_mode = function_arguments.get("fast", False)
                                script_path_reload = os.path.abspath(args.script)
                                script_name_reload = os.path.basename(script_path_reload)
                                module_name_reload = os.path.splitext(script_name_reload)[0].replace("-", "_")
                                spec_reload = importlib.util.spec_from_file_location(module_name_reload, script_path_reload)
                                spec_reload.loader.exec_module(script_module)
                                function_result = run_cleaner_script(
                                    args.script,
                                    args.dataset,
                                    script_module=script_module,
                                    dataset=dataset,
                                    fast=fast_mode,
                                )
                                if not fast_mode:
                                    dataset = load_from_disk(args.dataset)["train"]
                            elif function_name == "edit_cleaner_script":
                                function_result = edit_cleaner_script(
                                    file_path=args.script,
                                    edits=function_arguments.get("edits", []),
                                )
                            elif function_name == "search":
                                function_result = search_cleaner_script(
                                    query=function_arguments.get("query", ""),
                                    script_path=args.script,
                                    path=function_arguments.get("path", ""),
                                )
                            elif function_name == "open_file":
                                function_result = read_cleaner_script(
                                    script_path=args.script,
                                    line_start=function_arguments.get("line_start"),
                                    line_end=function_arguments.get("line_end"),
                                )
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
