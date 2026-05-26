"""Shared utilities for converting medical parquets to the Apertus linearised format."""

from typing import Dict, List, Any, Optional

# Empty elements matching 07-dataset-aggregation/linearise-dataset.py lines 129-130
EMPTY_CALLS = [{"name": "", "arguments": ""}]
EMPTY_OUTPUTS = [{"name": "", "output": ""}]


def create_block(
    block_type: str,
    text: str = "",
    calls: Optional[List[Dict[str, Any]]] = None,
    outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create a block for the assistant message."""
    return {
        "type": block_type,
        "text": text,
        "calls": calls if calls is not None else EMPTY_CALLS,
        "outputs": outputs if outputs is not None else EMPTY_OUTPUTS,
    }


def build_linearised_messages(
    user_content: str,
    assistant_content: str,
    reasoning: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build the linearised message list for a single-turn medical sample.

    Args:
        user_content: The user prompt text.
        assistant_content: The assistant response text.
        reasoning: Optional chain-of-thought reasoning (produces a thoughts block).

    Returns:
        List of developer, user, and assistant messages in linearised format.
    """
    has_thinking = reasoning is not None and len(reasoning) > 0

    developer_message = {
        "role": "developer",
        "content": {
            "tools": "",
            "has_thinking": has_thinking,
            "formatted_tools": "",
        },
    }

    user_message = {
        "role": "user",
        "content": {
            "parts": [{"type": "text", "text": user_content}],
        },
    }

    blocks = []
    if has_thinking:
        blocks.append(create_block(block_type="thoughts", text=reasoning))
    blocks.append(create_block(block_type="response", text=assistant_content))

    assistant_message = {
        "role": "assistant",
        "content": {"blocks": blocks},
    }

    return [developer_message, user_message, assistant_message]
