#!/usr/bin/env python3
"""
Linearise a dataset.

This script is meant to convert the stored dataset into a "standardised" format that can be used for training.

The "standardised" format is the following:

- "system" message:
```json
{
    "role": "system",
    "content": {
        // system prompt used to guide the model
        "text": "{system prompt}",
    }
}
```

- "developer" message:
```json
{
    "role": "developer",
    "content": {
        // tools available to the model as a JSON string
        "tools": "{tools}",
        // whether the model has thinking enabled
        "has_thinking": {true | false},
        // formatted tools as typescript types (used to give the schema to the model)
        "formatted_tools": "{formatted_tools}",
    }
}
```

- "user" message:
```json
{
    "role": "user",
    "content": {
        "parts": [
            {
                "type": "text",
                "text": "{user message}"
            }
        ]
    }
}
```

- "assistant" message:
```json
{
    "role": "assistant",
    "content": {
        "blocks": [
            {
                "type": "thoughts",
                // assistant thinking process
                "text": "{assistant reasoning}"
            },
            {
                "type": "tool_calls",
                // tool calls made by the assistant
                "calls": [
                    {
                        "name": "{tool name}",
                        // arguments to the tool as a JSON string
                        "arguments": "{tool arguments}"
                    }
                ]
            },
            {
                "type": "tool_outputs",
                // tool outputs from the assistant
                "outputs": [
                    {
                        // note: the name is not used in practice, but is kept for consistency
                        "name": "{tool name}",
                        // output from the tool as a JSON string
                        "output": "{tool output}"
                    }
                ]
            },
            {
                "type": "response",
                // response from the assistant to the user
                "text": "{assistant response}"
            }
        ]
    }
}
```

This script raises an error if:
- a tool output message is found without a tool call message before it
- a tool call message is found with a mesage of type "thoughts" or "response" after it
"""

import json
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Literal
from transformers.utils.chat_template_utils import render_jinja_template

from datasets import Dataset, DatasetDict, load_from_disk

# Answers tool
ANSWERS_TOOL_NAME = "display_answers"
ANSWERS_TOOL = {
    "name": ANSWERS_TOOL_NAME,
    "description": "Display the answers to the user",
    "parameters": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The answers to the user",
            },
        },
        "required": ["answers"],
    },
}
ANSWERS_TOOL_OUTPUT = '"Answers displayed"'

# Empty elements to keep the structure of the messages
EMPTY_CALLS = [{"name": "", "arguments": ""}]
EMPTY_OUTPUTS = [{"name": "", "output": ""}]

# Jinja template used to format the tools
TOOLS_TEMPLATE = '{%- macro render_typescript_type(param_spec, required_params, is_nullable=false) -%}\n    {%- if param_spec.type == "array" -%}\n        {%- if param_spec[\'items\'] -%}\n            {%- if param_spec[\'items\'][\'type\'] == "string" -%}\n                {{- "string[]" }}\n            {%- elif param_spec[\'items\'][\'type\'] == "number" -%}\n                {{- "number[]" }}\n            {%- elif param_spec[\'items\'][\'type\'] == "integer" -%}\n                {{- "number[]" }}\n            {%- elif param_spec[\'items\'][\'type\'] == "boolean" -%}\n                {{- "boolean[]" }}\n            {%- else -%}\n                {%- set inner_type = render_typescript_type(param_spec[\'items\'], required_params) -%}\n                {%- if inner_type == "object | object" or inner_type|length > 50 -%}\n                    {{- "any[]" }}\n                {%- else -%}\n                    {{- inner_type + "[]" }}\n                {%- endif -%}\n            {%- endif -%}\n            {%- if param_spec.nullable -%}\n                {{- " | null" }}\n            {%- endif -%}\n        {%- else -%}\n            {{- "any[]" }}\n            {%- if param_spec.nullable -%}\n                {{- " | null" }}\n            {%- endif -%}\n        {%- endif -%}\n    {%- elif param_spec.type is defined and param_spec.type is iterable and param_spec.type is not string and param_spec.type is not mapping and param_spec.type[0] is defined -%}\n        {#- Handle array of types like ["object", "object"] from Union[dict, list] #}\n        {%- if param_spec.type | length > 1 -%}\n            {{- param_spec.type | join(" | ") }}\n        {%- else -%}\n            {{- param_spec.type[0] }}\n        {%- endif -%}\n    {%- elif param_spec.oneOf -%}\n        {#- Handle oneOf schemas - check for complex unions and fallback to any #}\n        {%- set has_object_variants = false -%}\n        {%- for variant in param_spec.oneOf -%}\n            {%- if variant.type == "object" -%}\n                {%- set has_object_variants = true -%}\n            {%- endif -%}\n        {%- endfor -%}\n        {%- if has_object_variants and param_spec.oneOf|length > 1 -%}\n            {{- "any" }}\n        {%- else -%}\n            {%- for variant in param_spec.oneOf -%}\n                {{- render_typescript_type(variant, required_params) -}}\n                {%- if variant.description %}\n                    {{- "// " + variant.description }}\n                {%- endif -%}\n                {%- if variant.default is defined %}\n                    {{ "// default: " + variant.default|tojson }}\n                {%- endif -%}\n                {%- if not loop.last %}\n                    {{- " | " }}\n                {% endif -%}\n            {%- endfor -%}\n        {%- endif -%}\n    {%- elif param_spec.type == "string" -%}\n        {%- if param_spec.enum -%}\n            {{- \'"\' + param_spec.enum|join(\'" | "\') + \'"\' -}}\n        {%- else -%}\n            {{- "string" }}\n            {%- if param_spec.nullable %}\n                {{- " | null" }}\n            {%- endif -%}\n        {%- endif -%}\n    {%- elif param_spec.type == "number" -%}\n        {{- "number" }}\n    {%- elif param_spec.type == "integer" -%}\n        {{- "number" }}\n    {%- elif param_spec.type == "boolean" -%}\n        {{- "boolean" }}\n    {%- elif param_spec.type == "object" -%}\n        {%- if param_spec.properties -%}\n            {{- "{\\n" }}\n            {%- for prop_name, prop_spec in param_spec.properties.items() -%}\n                {{- prop_name -}}\n                {%- if prop_name not in (param_spec.required or []) -%}\n                    {{- "?" }}\n                {%- endif -%}\n                {{- ": " }}\n                {{ render_typescript_type(prop_spec, param_spec.required or []) }}\n                {%- if not loop.last -%}\n                    {{-", " }}\n                {%- endif -%}\n            {%- endfor -%}\n            {{- "}" }}\n        {%- else -%}\n            {{- "object" }}\n        {%- endif -%}\n    {%- else -%}\n        {{- "any" }}\n    {%- endif -%}\n{%- endmacro -%}\n\n{%- macro render_tools(tools) -%}\n    {%- for tool in tools %}\n        {{- "// " + tool.description + "\\n" }}\n        {{- "type "+ tool.name + " = " }}\n        {%- if tool.parameters and tool.parameters.properties %}\n            {{- "(_: {\\n" }}\n            {%- for param_name, param_spec in tool.parameters.properties.items() %}\n                {%- if param_spec.description %}\n                    {{- "// " + param_spec.description + "\\n" }}\n                {%- endif %}\n                {{- param_name }}\n                {%- if param_name not in (tool.parameters.required or []) -%}\n                    {{- "?" }}\n                {%- endif -%}\n                {{- ": " }}\n                {{- render_typescript_type(param_spec, tool.parameters.required or []) }}\n                {%- if param_spec.default is defined -%}\n                    {%- if param_spec.enum %}\n                        {{- ", // default: " + param_spec.default }}\n                    {%- elif param_spec.oneOf %}\n                        {{- "// default: " + param_spec.default }}\n                    {%- else %}\n                        {{- ", // default: " + param_spec.default|tojson }}\n                    {%- endif -%}\n                {%- endif -%}\n                {%- if not loop.last %}\n                    {{- ",\\n" }}\n                {%- else %}\n                    {{- "\\n" }}\n                {%- endif -%}\n            {%- endfor %}\n            {{- "}) => any;" }}\n        {%- else -%}\n            {{- "() => any;" }}\n        {%- endif -%}\n        {%- if not loop.last -%}\n            {{- "\\n" }}\n        {%- endif -%}\n    {%- endfor %}\n{%- endmacro -%}\n{{ render_tools(messages) }}\n'


def load_existing_metadata(dataset_path: Path) -> Dict[str, Any]:
    """Load existing metadata from dataset directory."""
    metadata_file = dataset_path / "dataset_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            return json.load(f)
    return {}


def save_dataset_with_processing_log(
    dataset_dict: DatasetDict, output_path: Path, input_path: str, total_samples: int
):
    """Save dataset and update processing log."""
    # Load existing metadata
    original_metadata = load_existing_metadata(Path(input_path))

    # Create processing log entry
    processing_entry = {
        "operation": "linearise_dataset",
        "script": "linearise_dataset.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": input_path,
        "output_path": str(output_path),
        "samples_processed": total_samples,
        "conversion_details": {
            "from_format": "parts_structure",
            "to_format": "linear_structure",
            "description": "Convert assistant and user messages from parts list structure to linear structure",
        },
    }

    # Update metadata
    metadata = {
        **original_metadata,
        "processing_log": original_metadata.get("processing_log", [])
        + [processing_entry],
    }

    # Save dataset and metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    with open(output_path / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def fix_bool_required(obj: Any) -> Any:
    """
    Fix the required field of a tool parameters object.
    Recursively processes nested objects to fix boolean required fields.
    """
    if not isinstance(obj, dict):
        return obj

    if "type" in obj and obj["type"] == "object":
        if "required" in obj and isinstance(obj["required"], bool):
            # Convert boolean required to list of all property keys
            if "properties" in obj and isinstance(obj["properties"], dict):
                obj["required"] = list(obj["properties"].keys())
            else:
                obj["required"] = []

        # Recursively fix nested objects in properties
        if "properties" in obj and isinstance(obj["properties"], dict):
            for key, value in obj["properties"].items():
                if isinstance(value, dict):
                    obj["properties"][key] = fix_bool_required(value)

    # Also check for nested structures in other parts of the object
    for key, value in obj.items():
        if (
            isinstance(value, dict) and key != "properties"
        ):  # Avoid double-processing properties
            obj[key] = fix_bool_required(value)

    return obj


def create_block(
    type: Literal["thoughts", "tool_calls", "tool_outputs", "response"],
    text: str = "",
    calls: List[Dict[str, Any]] = EMPTY_CALLS,
    outputs: List[Dict[str, Any]] = EMPTY_OUTPUTS,
) -> Dict[str, Any]:
    """
    Create a block for the assistant message.
    """
    return {
        "type": type,
        "text": text,
        "calls": calls,
        "outputs": outputs,
    }


def linearise_sample_for_sft(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Linearise a sample for SFT training.
    """
    output_messages = []

    if "system_prompt" in sample and sample["system_prompt"]:
        output_messages.append(
            {"role": "system", "content": {"text": sample["system_prompt"]["content"]}}
        )

    developer_message = {
        "role": "developer",
        "content": {
            "tools": "",
            "has_thinking": False,
            "formatted_tools": "",
        },
    }

    if "available_functions" in sample and sample["available_functions"]:
        dict_tools = []
        for tool in sample["available_functions"]:
            tool_parameters = tool["parameters"]

            if tool_parameters is None:
                tool_parameters = {}

            if isinstance(tool_parameters, str):
                tool_parameters = json.loads(tool_parameters)

            if (
                tool_parameters
                and "properties" in tool_parameters
                and isinstance(tool_parameters["properties"], str)
            ):
                tool_parameters["properties"] = json.loads(
                    tool_parameters["properties"]
                )

            if (
                tool_parameters
                and "required" in tool_parameters
                and isinstance(tool_parameters["required"], bool)
            ):
                tool_parameters = fix_bool_required(tool_parameters)

            dict_tools.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool_parameters,
                }
            )

        # Add the answers tool if the assistant message contains verifiable responses
        for message in sample["conversation_branches"][0]["messages"]:
            if message["role"] == "assistant" and any(
                part["type"] == "verifiable-responses" for part in message["parts"]
            ):
                dict_tools.append(ANSWERS_TOOL)
                break

        formatted_tools, _ = render_jinja_template(
            conversations=[dict_tools],
            chat_template=TOOLS_TEMPLATE,
            add_generation_prompt=False,
        )

        developer_message["content"]["tools"] = json.dumps(dict_tools)
        developer_message["content"]["formatted_tools"] = formatted_tools[0]

    output_messages.append(developer_message)

    if "initial_prompt" in sample and sample["initial_prompt"]:
        output_messages.append(
            {
                "role": "user",
                "content": {
                    "parts": [
                        {"type": "text", "text": sample["initial_prompt"]["content"]}
                    ]
                },
            }
        )

    conversation_branch = sample["conversation_branches"][0]

    for message in conversation_branch["messages"]:
        message_role = message["role"]

        if message_role == "user":
            for part in message["parts"]:
                output_messages.append(
                    {
                        "role": "user",
                        "content": {
                            "parts": [{"type": "text", "text": part["content"]}]
                        },
                    }
                )

        elif message_role == "assistant":
            assistant_blocks = []
            tool_calls_buffer = []
            tool_outputs_buffer = []
            for part in message["parts"]:
                part_type = part["type"]

                if part_type == "thought":
                    if not developer_message["content"]["has_thinking"]:
                        developer_message["content"]["has_thinking"] = True

                    if tool_calls_buffer:
                        tqdm.write(
                            f"[{sample['conversation_id']}]: Warning: tool calls without tool output. Skipping the remaining parts."
                        )
                        break

                    if tool_outputs_buffer:
                        assistant_blocks.append(
                            create_block(
                                type="tool_outputs", outputs=tool_outputs_buffer
                            )
                        )

                        tool_outputs_buffer = []

                    assistant_blocks.append(
                        create_block(type="thoughts", text=part["content"])
                    )

                elif part_type == "function-call":
                    if tool_outputs_buffer:
                        assistant_blocks.append(
                            create_block(
                                type="tool_outputs", outputs=tool_outputs_buffer
                            )
                        )

                        tool_outputs_buffer = []

                    tool_arguments = (
                        part["args"] or ""
                    )  # To avoid having None as arguments

                    if isinstance(tool_arguments, dict):
                        try:
                            tool_arguments = json.dumps(tool_arguments)
                        except json.JSONDecodeError:
                            tqdm.write(
                                f"[{sample['conversation_id']}]: Warning: Failed to parse tool arguments: {tool_arguments}"
                            )
                            tool_arguments = "{}"

                    tool_calls_buffer.append(
                        {
                            "name": part["name"],
                            "arguments": tool_arguments,
                        }
                    )
                elif part_type == "function-output":
                    # In some datasets, we have empty function outputs, which we skip
                    if part["content"] is None:
                        continue

                    if tool_calls_buffer:
                        assistant_blocks.append(
                            create_block(type="tool_calls", calls=tool_calls_buffer)
                        )
                        tool_calls_buffer = []
                    else:
                        tqdm.write(
                            f"[{sample['conversation_id']}]: Warning: No tool calls buffer found for tool output: {part['content']}. Skipping the remaining parts."
                        )
                        break

                    tool_outputs_buffer.append(
                        {
                            "name": "",  # We don't have the name of the function that was called
                            "output": part["content"],
                        }
                    )
                elif part_type == "response":
                    if tool_calls_buffer:
                        tqdm.write(
                            f"[{sample['conversation_id']}]: Warning: Tool calls without tool output. Skipping the remaining parts."
                        )
                        break

                    if tool_outputs_buffer:
                        assistant_blocks.append(
                            create_block(
                                type="tool_outputs", outputs=tool_outputs_buffer
                            )
                        )
                        tool_outputs_buffer = []

                    assistant_blocks.append(
                        create_block(type="response", text=part["content"])
                    )
                elif part_type == "verifiable-responses":
                    if tool_calls_buffer:
                        tqdm.write(
                            f"[{sample['conversation_id']}]: Warning: Tool calls without tool output. Skipping the remaining parts."
                        )
                        break

                    if tool_outputs_buffer:
                        assistant_blocks.append(
                            create_block(
                                type="tool_outputs", outputs=tool_outputs_buffer
                            )
                        )
                        tool_outputs_buffer = []

                    # The model gives the answers as a tool call
                    assistant_blocks.append(
                        create_block(
                            type="tool_calls",
                            calls=[
                                {
                                    "name": ANSWERS_TOOL_NAME,
                                    "arguments": json.dumps(
                                        {"answers": part["answers"]}
                                    ),
                                }
                            ],
                        )
                    )
                    assistant_blocks.append(
                        create_block(
                            type="tool_outputs",
                            outputs=[
                                {
                                    "name": ANSWERS_TOOL_NAME,
                                    "output": ANSWERS_TOOL_OUTPUT,
                                }
                            ],
                        )
                    )

            if tool_calls_buffer:
                assistant_blocks.append(
                    create_block(type="tool_calls", calls=tool_calls_buffer)
                )

            if tool_outputs_buffer:
                assistant_blocks.append(
                    create_block(type="tool_outputs", outputs=tool_outputs_buffer)
                )

            output_messages.append(
                {"role": "assistant", "content": {"blocks": assistant_blocks}}
            )

    return output_messages


def linearise_sample_for_dpo(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Linearise a sample for DPO training.
    """
    raise NotImplementedError("DPO training is not implemented yet")


FORMAT_FUNCTIONS = {
    "sft": linearise_sample_for_sft,
    "dpo": linearise_sample_for_dpo,
}


def process_dataset(dataset: Dataset, training_type: str) -> Dataset:
    """
    Process a single dataset split, formatting all samples using parallel processing.
    """
    format_fn = FORMAT_FUNCTIONS[training_type]

    columns_to_keep = [
        "conversation_id",
        "dataset_source",
        "original_metadata",
        "created_timestamp",
    ]

    def format_sample(sample):
        """Format a single sample and keep only required columns."""
        formatted_messages = format_fn(sample)

        result = {}
        for col in columns_to_keep:
            if col in sample:
                result[col] = sample[col]
        result["messages"] = formatted_messages

        return result

    formatted_dataset = dataset.map(
        format_sample,
        desc="Linearising dataset",
        num_proc=16,
        remove_columns=[
            col for col in dataset.column_names if col not in columns_to_keep
        ],
    )

    print(f"Conversion complete: {len(formatted_dataset)} samples processed")
    return formatted_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Linearise a dataset")
    parser.add_argument(
        "input_path", type=str, help="Path to input dataset in old format"
    )
    parser.add_argument("output_path", type=str, help="Path to save linearised dataset")
    parser.add_argument(
        "--training-type",
        type=str,
        default="sft",
        choices=["sft"],
        help="Type of training to linearise dataset for (default: sft)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only convert first N samples (for testing)",
    )

    args = parser.parse_args()

    print(f"Loading dataset from: {args.input_path}")
    dataset = load_from_disk(args.input_path)

    if isinstance(dataset, DatasetDict):
        print(f"Processing DatasetDict with splits: {list(dataset.keys())}")
        formatted_datasets = {}

        for split_name, split_dataset in dataset.items():
            print(f"\nProcessing split: {split_name}")

            # Sample if requested
            if args.sample:
                split_dataset = split_dataset.select(
                    range(min(args.sample, len(split_dataset)))
                )
                print(f"  Sampling first {len(split_dataset)} examples")

            # Format the split
            formatted_split = process_dataset(
                split_dataset, training_type=args.training_type
            )
            formatted_datasets[split_name] = formatted_split

            print(f"  Formatted {len(formatted_split)} samples")

        # Create new DatasetDict
        formatted_dataset = DatasetDict(formatted_datasets)
    else:
        # Single Dataset
        print("Processing single Dataset")

        # Sample if requested
        if args.sample:
            dataset = dataset.select(range(min(args.sample, len(dataset))))
            print(f"  Sampling first {len(dataset)} examples")

        # Format the dataset
        formatted_dataset = process_dataset(dataset, training_type=args.training_type)
        formatted_dataset = DatasetDict({"train": formatted_dataset})

    # Calculate total samples for processing log
    total_samples = sum(
        len(split_dataset) for split_dataset in formatted_dataset.values()
    )

    # Save converted dataset
    output_path = Path(args.output_path)

    # If output_path is a directory, use the same dataset name as input
    if output_path.is_dir() or args.output_path.endswith("/"):
        input_name = Path(args.input_path).name
        output_path = output_path / input_name

    print(f"\nSaving converted dataset to: {output_path}")
    save_dataset_with_processing_log(
        formatted_dataset, output_path, args.input_path, total_samples
    )

    # Print summary
    print("\n" + "=" * 50)
    print("Linearising Summary:")
    print("=" * 50)
    for split_name, split_dataset in formatted_dataset.items():
        print(f"  {split_name}: {len(split_dataset)} samples")

    print("\nLinearising complete!")


if __name__ == "__main__":
    main()
