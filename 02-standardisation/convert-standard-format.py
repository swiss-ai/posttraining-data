#!/usr/bin/env python3
"""
This script converts datasets from various formats (chat messages, ShareGPT, 
instruction-response, preference pairs) into the our dataset schema
for function calls, thinking, and verifiable responses.
"""

import sys
import json
import argparse
import hashlib
import re
from pathlib import Path
from subprocess import run
from datetime import datetime
from collections import Counter
from datasets import load_from_disk
from typing import Dict, Any, Optional, List, Tuple


def lint_code(code: str) -> Optional[str]:
    """Lint Python code using ruff format. Returns formatted code wrapped in markdown code block."""
    result = run(
        ["ruff", "format", "-"],
        input=code.strip().encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return f"```python\n{result.stdout.decode()}```"


def lint_python_code_blocks(content: str) -> str:
    """
    Find and lint all Python code blocks in a content string.
    Code blocks are expected to be in markdown format: ```python ... ```
    """
    if not content or "```python" not in content:
        return content
    
    parts = content.split("```python")
    if len(parts) <= 1:
        return content
    
    # Process each code block
    processed_parts = [parts[0]]  # First part (before any code)
    
    for i in range(1, len(parts)):
        if "```" in parts[i]:
            code_part, after_code = parts[i].split("```", 1)
            linted_code = lint_code(code_part)
            if linted_code is not None:
                processed_parts.append(linted_code)
            else:
                # If linting fails, keep original code block
                processed_parts.append(f"```python{code_part}```")
            processed_parts.append(after_code)
        else:
            # Malformed code block (no closing ```), keep as is
            processed_parts.append(parts[i])
    
    return "".join(processed_parts)


# Common programming languages for code block detection
# List of 20 common languages: python, javascript, java, cpp, c, csharp, go, rust, 
# ruby, php, swift, kotlin, typescript, html, css, sql, bash, r, scala, perl
CODE_BLOCK_PATTERN = re.compile(r'```[a-zA-Z][a-zA-Z0-9_+-]*')


def sample_contains_code(sample: Dict[str, Any]) -> bool:
    """
    Check if a sample contains any code blocks with language identifiers (```language).
    Does NOT match plain ``` blocks without a language.
    Checks system_prompt, initial_prompt, and all conversation branch messages.
    
    Common languages detected: python, javascript, java, cpp, c, csharp, go, rust,
    ruby, php, swift, kotlin, typescript, html, css, sql, bash, r, scala, perl, etc.
    """
    def has_code_block(text: str) -> bool:
        """Check if text contains a code block with language identifier."""
        if not isinstance(text, str):
            return False
        return bool(CODE_BLOCK_PATTERN.search(text))
    
    # Check system prompt content
    if "system_prompt" in sample and isinstance(sample["system_prompt"], dict):
        if "content" in sample["system_prompt"]:
            if has_code_block(sample["system_prompt"]["content"]):
                return True
    
    # Check initial prompt content
    if "initial_prompt" in sample and isinstance(sample["initial_prompt"], dict):
        if "content" in sample["initial_prompt"]:
            if has_code_block(sample["initial_prompt"]["content"]):
                return True
    
    # Check all conversation branches
    if "conversation_branches" in sample and isinstance(sample["conversation_branches"], list):
        for branch in sample["conversation_branches"]:
            if "messages" in branch and isinstance(branch["messages"], list):
                for message in branch["messages"]:
                    if "parts" in message and isinstance(message["parts"], list):
                        for part in message["parts"]:
                            if isinstance(part, dict) and "content" in part:
                                if has_code_block(part["content"]):
                                    return True
    
    return False


def lint_sample_python_code(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lint all Python code blocks in a standardized sample.
    Processes system_prompt, initial_prompt, and all conversation branch messages.
    """
    # Lint system prompt content
    if "system_prompt" in sample and isinstance(sample["system_prompt"], dict):
        if "content" in sample["system_prompt"]:
            sample["system_prompt"]["content"] = lint_python_code_blocks(
                sample["system_prompt"]["content"]
            )
    
    # Lint initial prompt content
    if "initial_prompt" in sample and isinstance(sample["initial_prompt"], dict):
        if "content" in sample["initial_prompt"]:
            sample["initial_prompt"]["content"] = lint_python_code_blocks(
                sample["initial_prompt"]["content"]
            )
    
    # Lint all conversation branches
    if "conversation_branches" in sample and isinstance(sample["conversation_branches"], list):
        for branch in sample["conversation_branches"]:
            if "messages" in branch and isinstance(branch["messages"], list):
                for message in branch["messages"]:
                    if "parts" in message and isinstance(message["parts"], list):
                        for part in message["parts"]:
                            if isinstance(part, dict) and "content" in part:
                                part["content"] = lint_python_code_blocks(part["content"])
    
    return sample


def extract_sample_id(row: Dict[str, Any], row_index: Optional[int] = None) -> Optional[str]:
    """Extract sample ID from row, with fallback to row index."""
    # Try common ID field names
    for id_field in ['id', 'sample_id', 'conversation_id', 'idx', 'index']:
        if id_field in row and row[id_field] is not None:
            return str(row[id_field])
    
    # Fallback to row index if provided
    if row_index is not None:
        return f"row_{row_index}"
    
    return None


def create_schema_compliant_part(part_type: str, content: str = "", metadata: Optional[Dict] = None, 
                                 name: str = "", args: str = "") -> Dict[str, Any]:
    """Create a schema-compliant part with all required fields for Arrow compatibility."""
    return {
        "type": part_type,
        "content": content,
        "metadata": metadata or {},
        "name": name,
        "args": args
    }


def extract_custom_instructions(row: Dict[str, Any]) -> Optional[str]:
    """Extract custom_instructions from chat_template_kwargs if present."""
    chat_template_kwargs = row.get("chat_template_kwargs")
    if chat_template_kwargs:
        if isinstance(chat_template_kwargs, dict):
            return chat_template_kwargs.get("custom_instructions")
        elif isinstance(chat_template_kwargs, str):
            try:
                parsed = json.loads(chat_template_kwargs)
                if isinstance(parsed, dict):
                    return parsed.get("custom_instructions")
            except (json.JSONDecodeError, TypeError):
                pass
    return None


def merge_system_prompt(existing_prompt: Optional[Dict[str, Any]], 
                        custom_instructions: Optional[str]) -> Dict[str, Any]:
    """Merge custom_instructions into system prompt, combining with existing if present."""
    existing_content = existing_prompt.get("content", "") if existing_prompt else ""
    
    if custom_instructions:
        if existing_content:
            # Combine existing and custom instructions with a separator
            combined_content = f"{existing_content}\n\n{custom_instructions}"
        else:
            combined_content = custom_instructions
    else:
        combined_content = existing_content
    
    return {
        "content": combined_content,
        "metadata": existing_prompt.get("metadata", {}) if existing_prompt else {}
    }


def validate_conversation_pattern(conversation_messages: List[Dict], conversation_id: str = "unknown") -> bool:
    """Validate conversation message pattern and return True if valid, False if should skip."""
    if not conversation_messages:
        return True  # Empty conversation is valid (prompt-only)
    
    try:
        # First message should be assistant (responding to initial user prompt)
        if conversation_messages[0]["role"] != "assistant":
            print(f"Warning: Skipping sample {conversation_id} - first message must be from assistant, got '{conversation_messages[0]['role']}'")
            return False
        
        # Check alternating pattern and valid roles
        for i in range(1, len(conversation_messages)):
            current_role = conversation_messages[i]["role"]
            previous_role = conversation_messages[i-1]["role"]
            
            # Check for valid roles
            if current_role not in ["user", "assistant"]:
                print(f"Warning: Skipping sample {conversation_id} - invalid role '{current_role}' at message {i}")
                return False
            
            # Check alternating pattern
            if current_role == previous_role:
                print(f"Warning: Skipping sample {conversation_id} - consecutive messages from same role '{current_role}' at messages {i-1}-{i}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Warning: Skipping sample {conversation_id} - validation error: {e}")
        return False


def validate_standardized_sample(sample: Dict[str, Any]) -> bool:
    """Validate a sample is in our post train data schema."""
    if not sample or sample is None:
        return False
    
    # Check for required fields
    if "conversation_id" not in sample:
        print(f"Warning: Sample missing conversation_id field")
        return False
    
    conversation_id = sample.get("conversation_id", "unknown")
    
    # Validate each conversation branch
    for i, branch in enumerate(sample.get("conversation_branches", [])):
        messages = branch.get("messages", [])
        branch_id = f"{conversation_id}_branch{i}" if len(sample.get("conversation_branches", [])) > 1 else conversation_id
        
        if not validate_conversation_pattern(messages, branch_id):
            return False
    
    return True


def generate_conversation_id(dataset_source: str, content: str, sample_id: Optional[str] = None) -> str:
    """Generate a globally unique conversation ID."""
    dataset_prefix = dataset_source.replace('/', '_').replace('-', '_')
    
    if sample_id:
        # Use provided ID + short content hash for verification
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:8]
        return f"{dataset_prefix}_{sample_id}_{content_hash}"
    else:
        # Fallback to longer content hash for uniqueness
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        return f"{dataset_prefix}_{content_hash}"


def convert_chat_messages(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None, 
                          include_ctk: bool = False, remove_sp: bool = False) -> Dict[str, Any]:
    """
    Convert standard chat format (messages array with role/content) to our schema.
    Used for: smoltalk, tulu-3-sft-mixture, etc.
    """
    messages = row.get("messages", [])
    if not messages or not isinstance(messages, list):
        raise ValueError("No valid messages array found")
    
    # Parse messages if stored as JSON string
    if isinstance(messages, str):
        messages = json.loads(messages)
    
    # Find system prompt, initial user prompt, and conversation messages
    system_prompt = None
    initial_prompt = None
    conversation_messages = []
    
    for msg in messages:
        if not isinstance(msg, dict):
            continue
            
        role = msg.get("role", "").strip()
        content = msg.get("content", "")
        
        if not role:
            continue
        
        # Identify message types
        if role == "system" and system_prompt is None:
            system_prompt = {
                "content": content,
                "metadata": {}
            }
        elif role == "user" and initial_prompt is None:
            initial_prompt = {
                "role": "user",
                "content": content,
                "metadata": {}
            }
        else:
            # Convert to parts structure with schema compliance
            message_parts = [create_schema_compliant_part("response", content, {})]
            
            conversation_messages.append({
                "role": role,
                "parts": message_parts
            })
    
    if initial_prompt is None:
        raise ValueError("No initial user prompt found")
    
    # Extract custom instructions from chat_template_kwargs if enabled
    custom_instructions = None
    if include_ctk:
        custom_instructions = extract_custom_instructions(row)
    
    # Merge custom instructions into system prompt
    if include_ctk and custom_instructions:
        system_prompt = merge_system_prompt(system_prompt, custom_instructions)
    
    # Remove system prompt if requested
    if remove_sp:
        system_prompt = {"content": "", "metadata": {}}
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, initial_prompt["content"], sample_id)
    
    
    # Preserve original metadata (exclude messages field)
    original_metadata = {k: v for k, v in row.items() if k != "messages"}
    
    # Create single conversation branch
    conversation_branches = [{
        "messages": conversation_messages
    }]
    
    result = {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": system_prompt or {"content": "", "metadata": {}},
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }
    
    return result


def convert_nemotron_format(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None,
                             include_ctk: bool = False, remove_sp: bool = False) -> Dict[str, Any]:
    """
    Convert Nemotron format (input array + output string + system_prompt) to new parts format.
    Used for: Llama-Nemotron-Post-Training-Dataset
    
    Format:
    - input: [{"role": "user", "content": "..."}] (list with single user message)
    - output: "..." (assistant response string)
    - system_prompt: "..." (optional system prompt)
    - Additional metadata: category, license, reasoning, generator, etc.
    """
    input_messages = row.get("input", [])
    output = row.get("output", "")
    system_prompt_content = row.get("system_prompt", "")
    
    if not input_messages or not isinstance(input_messages, list) or len(input_messages) == 0:
        raise ValueError("No valid input messages array found")
    
    # Extract user message from nested input structure
    user_msg_data = input_messages[0]
    if not isinstance(user_msg_data, dict) or "content" not in user_msg_data:
        raise ValueError("Invalid input message structure")
        
    user_content = user_msg_data.get("content", "")
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, user_content, sample_id)
    
    # Preserve original metadata (exclude input, output, system_prompt)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["input", "output", "system_prompt"]}
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": user_content,
        "metadata": {}
    }
    
    # Extract custom instructions from chat_template_kwargs if enabled
    custom_instructions = None
    if include_ctk:
        custom_instructions = extract_custom_instructions(row)
    
    # Create system prompt and merge custom instructions if present
    existing_system_prompt = {"content": system_prompt_content, "metadata": {}} if system_prompt_content else None
    if include_ctk and custom_instructions:
        system_prompt = merge_system_prompt(existing_system_prompt, custom_instructions)
    else:
        system_prompt = existing_system_prompt or {"content": "", "metadata": {}}
    
    # Remove system prompt if requested
    if remove_sp:
        system_prompt = {"content": "", "metadata": {}}
    
    # Create conversation with assistant response using parts
    conversation_messages = [{
        "role": "assistant",
        "parts": [create_schema_compliant_part("response", output, {})]
    }]
    
    
    # Create single conversation branch
    conversation_branches = [{
        "messages": conversation_messages
    }]
    
    result = {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": system_prompt,
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }
    
    return result


def convert_sharegpt_format(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None,
                            include_ctk: bool = False, remove_sp: bool = False) -> Dict[str, Any]:
    """
    Convert ShareGPT format (conversations with from/value fields) to new parts format.
    Used for: The-Tome, EuroBlocks-SFT-Synthetic-1124, etc.
    
    Format:
    - conversations: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
    """
    conversations = row.get("conversations", [])
    if not conversations:
        raise ValueError("No valid conversations array found")
    
    # Parse conversations if stored as JSON string
    if isinstance(conversations, str):
        try:
            conversations = json.loads(conversations)
        except (json.JSONDecodeError, TypeError):
            raise ValueError("conversations field is a string but not valid JSON")
    
    if not isinstance(conversations, list):
        raise ValueError("conversations must be a list")
    
    # Role mapping for ShareGPT format
    role_map = {"human": "user", "gpt": "assistant"}
    
    # Find system prompt, initial user prompt, and conversation messages
    system_prompt = None
    initial_prompt = None
    conversation_messages = []
    
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
            
        from_role = msg.get("from", "").strip()
        value_content = msg.get("value", "")
        
        if not from_role:
            continue
        
        # Map role
        role = role_map.get(from_role, from_role)
        
        # Identify message types
        if role == "system" and system_prompt is None:
            system_prompt = {
                "content": value_content,
                "metadata": {}
            }
        elif role == "user" and initial_prompt is None:
            initial_prompt = {
                "role": "user",
                "content": value_content,
                "metadata": {}
            }
        else:
            # Convert to parts structure with schema compliance
            message_parts = [create_schema_compliant_part("response", value_content, {})]
            
            conversation_messages.append({
                "role": role,
                "parts": message_parts
            })
    
    if initial_prompt is None:
        raise ValueError("No initial user prompt found")
    
    # Extract custom instructions from chat_template_kwargs if enabled
    custom_instructions = None
    if include_ctk:
        custom_instructions = extract_custom_instructions(row)
    
    # Merge custom instructions into system prompt
    if include_ctk and custom_instructions:
        system_prompt = merge_system_prompt(system_prompt, custom_instructions)
    
    # Remove system prompt if requested
    if remove_sp:
        system_prompt = {"content": "", "metadata": {}}
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, initial_prompt["content"], sample_id)
    
    
    # Preserve original metadata (exclude conversations field)
    original_metadata = {k: v for k, v in row.items() if k != "conversations"}
    
    # Create single conversation branch
    conversation_branches = [{
        "messages": conversation_messages
    }]
    
    result = {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": system_prompt or {"content": "", "metadata": {}},
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }
    
    return result


def convert_preference_format(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None,
                              include_ctk: bool = False, remove_sp: bool = False) -> Dict[str, Any]:
    """
    Convert preference format (prompt with chosen/rejected responses) to new parts format.
    Used for: DPO datasets, preference pairs, etc.
    """
    prompt = row.get("prompt", "")
    chosen = row.get("chosen", "")
    rejected = row.get("rejected", "")
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, prompt, sample_id)
    
    # Preserve original metadata (exclude prompt, chosen, rejected)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["prompt", "chosen", "rejected"]}
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": prompt,
        "metadata": {}
    }
    
    # Extract custom instructions from chat_template_kwargs if enabled
    custom_instructions = None
    if include_ctk:
        custom_instructions = extract_custom_instructions(row)
    
    # Create system prompt with custom instructions if present
    if include_ctk and custom_instructions:
        system_prompt = merge_system_prompt(None, custom_instructions)
    else:
        system_prompt = {"content": "", "metadata": {}}
    
    # Remove system prompt if requested
    if remove_sp:
        system_prompt = {"content": "", "metadata": {}}
    
    # Create conversation branches with parts structure (chosen first = most preferred)
    conversation_branches = [
        {
            "messages": [{
                "role": "assistant",
                "parts": [create_schema_compliant_part("response", chosen, {})]
            }]
        },
        {
            "messages": [{
                "role": "assistant", 
                "parts": [create_schema_compliant_part("response", rejected, {})]
            }]
        }
    ]
    
    return {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": system_prompt,
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }


def convert_instruction_response(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None,
                                  include_ctk: bool = False, remove_sp: bool = False) -> Dict[str, Any]:
    """
    Convert instruction-response format (instruction/input → output) to new parts format.
    Used for: alpaca-style datasets, etc.
    """
    # Combine instruction and input into user prompt
    instruction = row.get("instruction", "")
    input_text = row.get("input", "")
    output = row.get("output", "")
    
    # Create user prompt (combine instruction and input)
    user_content_parts = [instruction]
    if input_text:
        user_content_parts.append(input_text)
    user_content = "\n\n".join(user_content_parts)
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, user_content, sample_id)
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": user_content,
        "metadata": {}
    }
    
    # Extract custom instructions from chat_template_kwargs if enabled
    custom_instructions = None
    if include_ctk:
        custom_instructions = extract_custom_instructions(row)
    
    # Create system prompt with custom instructions if present
    if include_ctk and custom_instructions:
        system_prompt = merge_system_prompt(None, custom_instructions)
    else:
        system_prompt = {"content": "", "metadata": {}}
    
    # Remove system prompt if requested
    if remove_sp:
        system_prompt = {"content": "", "metadata": {}}
    
    # Create single conversation branch with assistant response using parts
    conversation_branches = [{
        "messages": [{
            "role": "assistant",
            "parts": [create_schema_compliant_part("response", output, {})]
        }]
    }]
    
    # Preserve original metadata (exclude instruction, input, output)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["instruction", "input", "output"]}
    
    return {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": system_prompt,
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }


def convert_inputs_labels_format(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None,
                                   include_ctk: bool = False, remove_sp: bool = False) -> Dict[str, Any]:
    """
    Convert inputs/labels format to new parts format.
    Used for: DataProvenanceInitiative Commercial-Flan-Collection datasets.
    """
    inputs = row.get("inputs", "")
    labels = row.get("labels", "")
    
    if not inputs or not labels:
        raise ValueError("Missing required fields: inputs or labels")
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, inputs, sample_id)
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": inputs,
        "metadata": {}
    }
    
    # Extract custom instructions from chat_template_kwargs if enabled
    custom_instructions = None
    if include_ctk:
        custom_instructions = extract_custom_instructions(row)
    
    # Create system prompt with custom instructions if present
    if include_ctk and custom_instructions:
        system_prompt = merge_system_prompt(None, custom_instructions)
    else:
        system_prompt = {"content": "", "metadata": {}}
    
    # Remove system prompt if requested
    if remove_sp:
        system_prompt = {"content": "", "metadata": {}}
    
    # Create single conversation branch with assistant response using parts
    conversation_branches = [{
        "messages": [{
            "role": "assistant",
            "parts": [create_schema_compliant_part("response", labels, {})]
        }]
    }]
    
    # Preserve original metadata (exclude inputs, labels)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["inputs", "labels"]}
    
    return {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": system_prompt,
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }


def convert_input_output_format(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None,
                                  include_ctk: bool = False, remove_sp: bool = False) -> Dict[str, Any]:
    """
    Convert input/output format to new parts format.
    Used for: muri-it and similar datasets with simple input->output structure.
    """
    input_text = row.get("input", "")
    output_text = row.get("output", "")
    
    if not input_text or not output_text:
        raise ValueError("Missing required fields: input or output")
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, input_text, sample_id)
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": input_text,
        "metadata": {}
    }
    
    # Extract custom instructions from chat_template_kwargs if enabled
    custom_instructions = None
    if include_ctk:
        custom_instructions = extract_custom_instructions(row)
    
    # Create system prompt with custom instructions if present
    if include_ctk and custom_instructions:
        system_prompt = merge_system_prompt(None, custom_instructions)
    else:
        system_prompt = {"content": "", "metadata": {}}
    
    # Remove system prompt if requested
    if remove_sp:
        system_prompt = {"content": "", "metadata": {}}
    
    # Create single conversation branch with assistant response using parts
    conversation_branches = [{
        "messages": [{
            "role": "assistant",
            "parts": [create_schema_compliant_part("response", output_text, {})]
        }]
    }]
    
    # Preserve original metadata (exclude input, output)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["input", "output"]}
    
    return {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": system_prompt,
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }

def auto_detect_converter(dataset):
    """Auto-detect appropriate converter based on dataset structure."""
    # Get a sample row to inspect structure
    if hasattr(dataset, 'keys'):  # DatasetDict
        sample = dataset[list(dataset.keys())[0]][0]
    else:  # Single Dataset
        sample = dataset[0]
    
    # Check for ShareGPT format first (conversations with from/value structure)
    # This takes priority because it's a distinct format
    if "conversations" in sample:
        conversations = sample.get("conversations", [])
        if isinstance(conversations, list) and len(conversations) > 0:
            # Check if it has the ShareGPT structure (from/value fields)
            first_msg = conversations[0] if isinstance(conversations[0], dict) else {}
            if "from" in first_msg and "value" in first_msg:
                return convert_sharegpt_format
    
    # Check for standard chat messages format
    if "messages" in sample:
        messages = sample.get("messages", [])
        if isinstance(messages, list) and len(messages) > 0:
            # Check if it has the standard chat format (role/content fields)
            first_msg = messages[0] if isinstance(messages[0], dict) else {}
            if "role" in first_msg and "content" in first_msg:
                return convert_chat_messages
    
    # Check for Nemotron format (input as list + output)
    if "input" in sample and "output" in sample and isinstance(sample["input"], list):
        return convert_nemotron_format
    
    # Check for preference format
    if "prompt" in sample and "chosen" in sample and "rejected" in sample:
        return convert_preference_format
    
    # Check for instruction-response format
    if "instruction" in sample and "output" in sample:
        return convert_instruction_response
    
    # Check for inputs/labels format
    if "inputs" in sample and "labels" in sample:
        return convert_inputs_labels_format
    
    # Check for input/output format
    if "input" in sample and "output" in sample:
        return convert_input_output_format
    
    # If no format matches, raise an error
    raise ValueError(f"Cannot auto-detect format for sample with keys: {list(sample.keys())}")


def get_nested_value(obj: Dict[str, Any], field_path: str) -> Any:
    """
    Get value from nested dictionary using dot notation and array indexing.
    
    Args:
        obj: Dictionary to search in
        field_path: Dot-separated path like 'original_metadata.category' or 'messages[0].role'
    
    Returns:
        Value at the specified path, or None if not found
    """
    try:
        current = obj

        # Replace array notation with dots for parsing: messages[0].role -> messages.0.role
        normalized_path = field_path.replace('[', '.').replace(']', '')
        parts = normalized_path.split('.')

        for part in parts:
            if part.isdigit():
                # It's an array index
                index = int(part)
                if isinstance(current, list) and 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            else:
                # It's a dictionary key
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None

        return current
    except (KeyError, TypeError, AttributeError, IndexError):
        return None


def check_filter_match(sample: Dict[str, Any], field_path: Optional[str],
                       include_values: Optional[List[str]],
                       exclude_values: Optional[List[str]]) -> bool:
    """
    Check if a sample matches the filter criteria.
    
    Args:
        sample: Sample to check
        field_path: Field path to check
        include_values: Values to include (if specified, sample must have one of these)
        exclude_values: Values to exclude (if specified, sample must not have any of these)
    
    Returns:
        True if sample matches filter criteria, False otherwise
    """
    # If no filter specified, match everything
    if not field_path:
        return True

    # Get the field value
    field_value = get_nested_value(sample, field_path)
    field_value_str = str(field_value) if field_value is not None else "<NULL>"

    # Check exclude list first (takes precedence)
    if exclude_values and field_value_str in exclude_values:
        return False

    # Check include list
    if include_values and field_value_str not in include_values:
        return False

    return True


def load_existing_metadata(input_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    metadata_file = Path(input_path) / "dataset_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def convert_dataset_simple(dataset, converter, dataset_source: str,
                          field_path: Optional[str] = None,
                          include_values: Optional[List[str]] = None,
                          exclude_values: Optional[List[str]] = None,
                          include_ctk: bool = False,
                          filter_code: bool = False,
                          remove_sp: bool = False):
    """Convert dataset using simple dataset.map() approach with validation and optional filtering."""
    
    # Apply filtering before conversion if filter is specified
    if field_path:
        initial_count = len(dataset)
        print(f"Filtering dataset: {initial_count:,} samples before filter")
        
        # Add filter match flag to each example
        def add_filter_match(example, idx):
            return {
                "filter_match": check_filter_match(dict(example), field_path, include_values, exclude_values),
                "original_idx": idx
            }
        
        # Map to add filter flags
        dataset_with_filter = dataset.map(
            add_filter_match,
            with_indices=True,
            desc="Checking filter",
            num_proc=16
        )
        
        # Filter based on match
        dataset = dataset_with_filter.filter(
            lambda x: x["filter_match"],
            desc="Filtering",
            num_proc=16
        )
        
        # Remove temporary filter columns
        if "filter_match" in dataset.column_names:
            dataset = dataset.remove_columns(["filter_match", "original_idx"])
        
        filtered_count = len(dataset)
        print(f"Filtered dataset: {filtered_count:,} samples after filter")
        
        if filtered_count == 0:
            print("No samples match the filter criteria. Nothing to convert.")
            return None
    
    def convert_with_validation(example, idx):
        try:
            # Convert first, passing include_ctk and remove_sp flags
            converted = converter(example, dataset_source, idx, include_ctk=include_ctk, remove_sp=remove_sp)
            
            if converted is None:
                return None  # Return None for filtering
            
            # Check if conversion produced valid structure
            if not isinstance(converted, dict):
                print(f"Warning: Converter returned non-dict for sample {idx}: {type(converted)}")
                return None
            
            # Lint all Python code blocks in the converted sample
            try:
                converted = lint_sample_python_code(converted)
            except Exception as lint_error:
                print(f"Warning: Failed to lint Python code for sample {idx}: {lint_error}")
                # Continue with un-linted sample rather than failing
            
            # Validate the standardized format
            try:
                if not validate_standardized_sample(converted):
                    return None  # Return None for filtering
            except Exception as validation_error:
                print(f"Warning: Validation failed for sample {idx}: {validation_error}")
                print(f"Sample keys: {list(converted.keys()) if isinstance(converted, dict) else 'Not a dict'}")
                return None
            
            return converted
            
        except Exception as e:
            print(f"Warning: Failed to convert sample {idx}: {e}")
            return None  # Return None for filtering
    
    print(f"\nConverting, linting Python code, and validating {len(dataset):,} samples...")
    if include_ctk:
        print("Including custom_instructions from chat_template_kwargs in system prompts")
    if remove_sp:
        print("Removing system prompts from all samples")
    if filter_code:
        print("Filtering out samples containing code blocks")
    
    # Use dataset.map with enumeration for index
    converted = dataset.map(
        convert_with_validation,
        with_indices=True,
        desc="Converting",
        remove_columns=dataset.column_names,  # Remove original columns
        num_proc=16
    )
    
    # Filter out None results (invalid/failed conversions)
    initial_count = len(converted)
    converted = converted.filter(lambda x: x is not None)
    final_count = len(converted)
    
    if initial_count > final_count:
        print(f"Filtered out {initial_count - final_count:,} invalid samples ({final_count:,} remain)")
    
    # Filter out samples with code blocks if filter_code is enabled (after conversion)
    if filter_code:
        before_code_filter = len(converted)
        converted = converted.filter(
            lambda x: not sample_contains_code(x),
            desc="Filtering out samples with code blocks",
            num_proc=16
        )
        after_code_filter = len(converted)
        if before_code_filter > after_code_filter:
            print(f"Filtered out {before_code_filter - after_code_filter:,} samples containing code blocks ({after_code_filter:,} remain)")
    
    return converted


def get_system_prompt_stats(dataset) -> Tuple[Dict[str, int], int]:
    """
    Get system prompt statistics from a dataset.
    
    Returns:
        tuple: (dict mapping system prompt content to count, total sample count)
    """
    from datasets import DatasetDict
    
    system_prompt_counts = Counter()
    total_samples = 0
    
    # Handle DatasetDict vs single Dataset
    if isinstance(dataset, DatasetDict):
        datasets_to_check = dataset.values()
    else:
        datasets_to_check = [dataset]
    
    for ds in datasets_to_check:
        for sample in ds:
            total_samples += 1
            if "system_prompt" in sample and isinstance(sample["system_prompt"], dict):
                content = sample["system_prompt"].get("content", "")
                system_prompt_counts[content] += 1
            else:
                # Count missing system prompts as empty
                system_prompt_counts[""] += 1
    
    return dict(system_prompt_counts), total_samples


def save_dataset_and_metadata(dataset, output_path: Path, dataset_name: str, input_path: Path):
    """Save converted dataset and update metadata."""
    from datasets import DatasetDict
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Always save as DatasetDict with 'train' split for pipeline compatibility
    if isinstance(dataset, DatasetDict):
        output_dataset = dataset
    else:
        output_dataset = DatasetDict({"train": dataset})
    
    # Save dataset
    print(f"Saving dataset to {output_path}...")
    output_dataset.save_to_disk(str(output_path))
    
    # Load or create metadata
    metadata = load_existing_metadata(input_path) or {}
    
    # Add processing log entry
    processing_entry = {
        "operation": "standardisation",
        "script": "convert_to_chat.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "samples_processed": len(dataset) if not isinstance(dataset, DatasetDict) else sum(len(split) for split in dataset.values()),
        "conversion_success": True,
        "target_schema": "chat_format_v1.0"
    }
    
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Save updated metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")




def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace datasets to our dataset schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convert-standard-format.py ../data/01-hf-data/smoltalk ../data/02-standardised/smoltalk
  python convert-standard-format.py ../data/01-hf-data/tulu-3-sft-mixture ../data/02-standardised/tulu-newformat --split validation
  
  # Filter by field - include only specific categories
  python convert-standard-format.py ../data/01-hf-data/dataset ../data/02-standardised/dataset-filtered \\
    --field "category" --include "math" "science"
  
  # Filter by field - exclude specific categories
  python convert-standard-format.py ../data/01-hf-data/dataset ../data/02-standardised/dataset-filtered \\
    --field "category" --exclude "refusal" "low_quality"
  
  # Filter by nested metadata field
  python convert-standard-format.py ../data/01-hf-data/dataset ../data/02-standardised/dataset-filtered \\
    --field "metadata.quality_score" --include "high" "medium"
  
  # Include custom_instructions from chat_template_kwargs as system prompt
  python convert-standard-format.py ../data/01-hf-data/smoltalk ../data/02-standardised/smoltalk \\
    --include_ctk
  
  # Filter out samples containing code blocks
  python convert-standard-format.py ../data/01-hf-data/dataset ../data/02-standardised/dataset-no-code \\
    --filter_code
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Path to input HuggingFace dataset directory"
    )
    parser.add_argument(
        "output_path", 
        help="Path for output dataset directory"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)"
    )
    
    # Field filtering arguments
    parser.add_argument(
        "--field",
        type=str,
        help="Field path for filtering (e.g., 'category' or 'metadata.quality_score')"
    )
    parser.add_argument(
        "--include",
        nargs="+",
        help="Values to include (samples must have one of these values)"
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Values to exclude (samples must not have any of these values)"
    )
    
    parser.add_argument(
        "--include_ctk",
        action="store_true",
        help="Include custom_instructions from chat_template_kwargs as system prompt"
    )
    
    parser.add_argument(
        "--filter_code",
        action="store_true",
        help="Filter out all samples containing code blocks with language identifiers (```language)"
    )
    
    parser.add_argument(
        "--show_sp",
        action="store_true",
        help="Print the number of unique system prompts at the end of conversion"
    )
    
    parser.add_argument(
        "--remove_sp",
        action="store_true",
        help="Remove system prompts from all converted samples (set to empty)"
    )
    
    return parser.parse_args()


def main():
    """Main conversion function."""
    args = parse_arguments()
    
    # Validate filter arguments
    if (args.include or args.exclude) and not args.field:
        print("Error: --field must be specified when using --include or --exclude")
        sys.exit(1)
    
    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Get dataset name from input path
    dataset_name = input_path.name
    print(f"Converting dataset: {dataset_name}")
    print(f"Input:  {input_path}")
    
    print(f"Output: {args.output_path}")
    
    # Load input dataset
    print(f"Loading dataset from: {input_path}")
    try:
        dataset = load_from_disk(str(input_path))
        
        # Auto-detect converter function
        try:
            converter = auto_detect_converter(dataset)
            print(f"Auto-detected format: {converter.__name__}")
        except ValueError as e:
            print(f"Error: {e}")
            print("Supported formats: messages, conversations, input+output (Nemotron), prompt+chosen+rejected, instruction+output, inputs+labels")
            sys.exit(1)
        
        # Handle DatasetDict vs single Dataset
        if hasattr(dataset, 'keys'):
            # DatasetDict - use specified split or fallback to first available
            available_splits = list(dataset.keys())
            print(f"Found DatasetDict with splits: {available_splits}")
            
            if args.split in available_splits:
                input_dataset = dataset[args.split]
                print(f"Using '{args.split}' split with {len(input_dataset):,} samples")
            else:
                print(f"Warning: Requested split '{args.split}' not found. Available splits: {available_splits}")
                split_name = available_splits[0]
                input_dataset = dataset[split_name]
                print(f"Using '{split_name}' split with {len(input_dataset):,} samples")
        else:
            # Single Dataset
            input_dataset = dataset
            print(f"Loaded single dataset with {len(input_dataset):,} samples")
        
        # Convert dataset
        print(f"Converting using {converter.__name__}...")
        converted_dataset = convert_dataset_simple(
            input_dataset, 
            converter, 
            dataset_name,
            field_path=args.field,
            include_values=args.include,
            exclude_values=args.exclude,
            include_ctk=args.include_ctk,
            filter_code=args.filter_code,
            remove_sp=args.remove_sp
        )
        
        # Check if filtering resulted in no samples
        if converted_dataset is None:
            print(f"\n⚠️  No samples to convert after filtering.")
            sys.exit(0)
        
        # Save output
        output_path = Path(args.output_path)
        save_dataset_and_metadata(converted_dataset, output_path, dataset_name, input_path)
        
        print(f"\n✅ Conversion complete!")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Samples: {len(converted_dataset):,}")
        
        # Print system prompt statistics if requested
        if args.show_sp:
            sp_counts, total_samples = get_system_prompt_stats(converted_dataset)
            unique_sp_count = len(sp_counts)
            print(f"\nSystem Prompt Statistics:")
            print(f"Unique system prompts: {unique_sp_count:,}")
            print(f"Total samples: {total_samples:,}")
            print(f"\nSystem prompt proportions:")
            
            # Sort by count (descending) for better readability
            sorted_sp = sorted(sp_counts.items(), key=lambda x: x[1], reverse=True)
            
            for sp_content, count in sorted_sp:
                proportion = count / total_samples if total_samples > 0 else 0.0
                percentage = proportion * 100
                
                # Truncate long system prompts for display
                display_content = sp_content if len(sp_content) <= 100 else sp_content[:97] + "..."
                # Replace newlines with spaces for cleaner display
                display_content = display_content.replace("\n", " ").replace("\r", "")
                
                if not sp_content:
                    display_content = "<empty>"
                
                print(f"  [{count:,} samples ({percentage:.2f}%)] {display_content}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
