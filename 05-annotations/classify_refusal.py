#!/usr/bin/env python3
"""
Refusal Classification Script

Classifies assistant messages in chat format datasets (with parts structure) to identify
refusal responses where the assistant declines to provide information or assistance due to
safety, ethical, or capability constraints.

Only processes response-type parts, excluding function calls, function outputs,
and verifiable answers.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import copy

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from base_classifier import BaseClassifier
from llm_classifier import extract_message_context, should_classify_message


class RefusalClassifier(BaseClassifier):
    """Classifier for identifying refusal responses in assistant messages."""

    def __init__(self):
        super().__init__(
            classifier_name="refusal_classification",
            template_filename="refusal.txt",
            valid_categories=["refusal", "no_refusal"],
            description="Classify refusal responses in chat format datasets (with parts structure)"
        )

    def collect_tasks(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect all assistant response parts that need refusal classification from a sample.

        Args:
            sample: Chat format sample with parts structure

        Returns:
            List of classification tasks with response parts and context
        """
        tasks = []

        # Build conversation context
        conversation_context = []

        # Add system prompt if present
        if sample.get("system_prompt") and sample["system_prompt"].get("content", "").strip():
            conversation_context.append(f"System: {sample['system_prompt']['content']}")

        # Add initial prompt if present
        if sample.get("initial_prompt") and sample["initial_prompt"].get("content", "").strip():
            role = sample["initial_prompt"].get("role", "user").title()
            content = sample["initial_prompt"]["content"]
            conversation_context.append(f"{role}: {content}")

        # Process messages from all conversation branches
        for branch_idx, branch in enumerate(sample.get("conversation_branches", [])):
            for msg_idx, message in enumerate(branch.get("messages", [])):
                role = message.get("role", "").lower()

                if "parts" in message and isinstance(message["parts"], list):
                    # New format with parts
                    # Extract response parts for context and classification
                    response_parts = []
                    for part in message["parts"]:
                        if (isinstance(part, dict) and
                            part.get("type") == "response" and
                            part.get("content", "").strip()):
                            response_parts.append(part["content"])

                    # Add to context if we have response content
                    if response_parts:
                        role_display = role.title()
                        full_content = " ".join(response_parts)
                        conversation_context.append(f"{role_display}: {full_content}")

                    # Create classification tasks for assistant response parts
                    if role == "assistant":
                        for part_idx, part in enumerate(message["parts"]):
                            if (isinstance(part, dict) and
                                part.get("type") == "response" and
                                part.get("content", "").strip()):
                                # Use the conversation context built so far (excluding this response)
                                context_text = "\n".join(conversation_context[:-1]) if len(conversation_context) > 1 else "[No previous context]"

                                tasks.append({
                                    "sample": sample,
                                    "branch_index": branch_idx,
                                    "message_index": msg_idx,
                                    "part_index": part_idx,
                                    "part": part,
                                    "question": context_text,
                                    "answer": part["content"]
                                })

        return tasks

    def apply_results(self, tasks: List[Dict[str, Any]], results: List[Any],
                      model: str) -> List[Dict[str, Any]]:
        """
        Apply classification results to the original samples (with parts structure).

        Args:
            tasks: Original classification tasks
            results: Classification results from LLM
            model: Model name used for classification

        Returns:
            List of updated samples with classification metadata in response parts
        """
        # Group tasks by sample to avoid duplicates
        sample_updates = {}

        for task, result in zip(tasks, results):
            sample_id = task["sample"]["conversation_id"]
            if sample_id not in sample_updates:
                sample_updates[sample_id] = {
                    "sample": copy.deepcopy(task["sample"]),  # Deep copy to avoid Arrow corruption
                    "updates": []
                }

            # Create classification metadata
            classification_metadata = {
                "classification": result.classification,
                "reasoning": result.reasoning
            }

            if not result.success:
                classification_metadata["error"] = result.error if result.error else "Classification failed"

            sample_updates[sample_id]["updates"].append({
                "branch_index": task["branch_index"],
                "message_index": task["message_index"],
                "part_index": task["part_index"],
                "metadata": classification_metadata
            })

        # Apply updates to samples
        updated_samples = []
        for sample_data in sample_updates.values():
            sample = sample_data["sample"]

            # Apply metadata updates to response parts or messages
            for update in sample_data["updates"]:
                branch_index = update["branch_index"]
                message_index = update["message_index"]
                part_index = update["part_index"]
                classification_metadata = update["metadata"]

                # Update specific part metadata
                message = sample["conversation_branches"][branch_index]["messages"][message_index]

                # Initialize part metadata if not present
                if "metadata" not in message["parts"][part_index]:
                    message["parts"][part_index]["metadata"] = {}

                # Create nested structure: refusal_classification -> model -> data
                if "refusal_classification" not in message["parts"][part_index]["metadata"]:
                    message["parts"][part_index]["metadata"]["refusal_classification"] = {}

                message["parts"][part_index]["metadata"]["refusal_classification"][model] = classification_metadata

            updated_samples.append(sample)

        return updated_samples


def main():
    """Main function."""
    classifier = RefusalClassifier()
    return classifier.run_classification()


if __name__ == "__main__":
    sys.exit(main())