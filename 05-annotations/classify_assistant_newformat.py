#!/usr/bin/env python3
"""
AI Assistant Classification Script (New Format)

Classifies question-answer pairs in new chat format datasets (with parts structure) 
to identify content involving AI assistants, chatbots, or language models.

Only processes response-type parts, excluding function calls, function outputs, 
and verifiable answers.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from base_classifier import BaseClassifier
from llm_classifier import extract_message_context, should_classify_message


class AssistantClassifier(BaseClassifier):
    """Classifier for identifying AI assistant-related content in new format datasets."""
    
    def __init__(self):
        super().__init__(
            classifier_name="assistant_classification",
            template_filename="assistant.txt",
            valid_categories=["ai_assistant_related", "non_ai_assistant"],
            description="Classify AI assistant-related content in new chat format datasets (with parts structure)"
        )
    
    def collect_tasks(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect all response parts from assistant messages that need AI assistant classification.
        
        Args:
            sample: New chat format sample with parts structure
            
        Returns:
            List of classification tasks with response parts and context
        """
        tasks = []
        
        
        # Build conversation context
        conversation_context = []
        
        # Add system prompt if present
        if sample.get("system_prompt") and sample["system_prompt"].get("content", "").strip():
            conversation_context.append(f"System: {sample['system_prompt']['content']}")
        
        # Add initial prompt if present and create classification task for it
        if sample.get("initial_prompt") and sample["initial_prompt"].get("content", "").strip():
            role = sample["initial_prompt"].get("role", "user").title()
            content = sample["initial_prompt"]["content"]
            conversation_context.append(f"{role}: {content}")
            
            # Create classification task for initial prompt
            context_text = "\n".join(conversation_context[:-1]) if len(conversation_context) > 1 else "[No previous context]"
            tasks.append({
                "sample": sample,
                "message": sample["initial_prompt"],
                "part_index": None,  # Not a part, it's the full initial prompt
                "part": None,
                "question": context_text,
                "answer": content,
                "is_initial_prompt": True
            })
        
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
                                context_text = "\\n".join(conversation_context[:-1]) if len(conversation_context) > 1 else "[No previous context]"
                                
                                tasks.append({
                                    "sample": sample,
                                    "branch_index": branch_idx,
                                    "message_index": msg_idx,
                                    "part_index": part_idx,
                                    "part": part,
                                    "question": context_text,
                                    "answer": part["content"]
                                })
                
                elif message.get("content", "").strip():
                    # Legacy format or direct content
                    role_display = role.title()
                    conversation_context.append(f"{role_display}: {message['content']}")
        
        return tasks
    
    def apply_results(self, tasks: List[Dict[str, Any]], results: List[Any], 
                     model: str) -> List[Dict[str, Any]]:
        """
        Apply classification results to the original samples (new format with parts).
        
        Args:
            tasks: Original classification tasks
            results: Classification results from LLM
            model: Model name used for classification
            
        Returns:
            List of updated samples with classification metadata in response parts
        """
        import copy
        
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
                "branch_index": task.get("branch_index"),
                "message_index": task.get("message_index"),
                "part_index": task["part_index"],
                "metadata": classification_metadata,
                "is_initial_prompt": task.get("is_initial_prompt", False)
            })
        
        # Apply updates to samples
        updated_samples = []
        for sample_data in sample_updates.values():
            sample = sample_data["sample"]
            
            # Apply metadata updates to response parts or initial prompt
            for update in sample_data["updates"]:
                branch_index = update["branch_index"]
                message_index = update["message_index"]
                part_index = update["part_index"]
                classification_metadata = update["metadata"]
                is_initial_prompt = update["is_initial_prompt"]
                
                if is_initial_prompt:
                    # Update initial prompt metadata
                    if "metadata" not in sample["initial_prompt"]:
                        sample["initial_prompt"]["metadata"] = {}
                    
                    # Create nested structure: assistant_classification -> model -> data
                    if "assistant_classification" not in sample["initial_prompt"]["metadata"]:
                        sample["initial_prompt"]["metadata"]["assistant_classification"] = {}
                    
                    sample["initial_prompt"]["metadata"]["assistant_classification"][model] = classification_metadata
                else:
                    # Use indices to directly access the correct part
                    message = sample["conversation_branches"][branch_index]["messages"][message_index]
                    
                    # Initialize part metadata if not present
                    if "metadata" not in message["parts"][part_index]:
                        message["parts"][part_index]["metadata"] = {}
                    
                    # Create nested structure: assistant_classification -> model -> data
                    if "assistant_classification" not in message["parts"][part_index]["metadata"]:
                        message["parts"][part_index]["metadata"]["assistant_classification"] = {}
                    
                    message["parts"][part_index]["metadata"]["assistant_classification"][model] = classification_metadata
            
            updated_samples.append(sample)
        
        return updated_samples


def main():
    """Main function."""
    classifier = AssistantClassifier()
    return classifier.run_classification()


if __name__ == "__main__":
    sys.exit(main())