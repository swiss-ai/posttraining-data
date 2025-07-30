#!/usr/bin/env python3
"""
Refusal Classification Script (BaseClassifier Version)

Classifies assistant messages in chat format datasets to identify refusal responses 
where the assistant declines to provide information or assistance due to safety, 
ethical, or capability constraints.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

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
            description="Classify refusal responses in chat format datasets"
        )
    
    def collect_tasks(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect all assistant messages that need refusal classification from a sample.
        
        Args:
            sample: Chat format sample
            
        Returns:
            List of classification tasks with message and context
        """
        tasks = []
        
        # Build conversation message list for context extraction
        conversation_messages = []
        
        # Add system prompt if present
        if sample.get("system_prompt"):
            conversation_messages.append({
                "role": "system",
                "content": sample["system_prompt"]["content"]
            })
        
        # Add initial prompt
        if sample.get("initial_prompt"):
            conversation_messages.append({
                "role": sample["initial_prompt"]["role"],
                "content": sample["initial_prompt"]["content"]
            })
        
        # Add messages from all conversation branches
        for branch in sample.get("conversation_branches", []):
            for message in branch.get("messages", []):
                conversation_messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })
                
                # Check if this assistant message needs classification
                if should_classify_message(message, ["assistant"]):
                    context = extract_message_context(message, conversation_messages)
                    
                    tasks.append({
                        "sample": sample,
                        "message": message,
                        "question": context,
                        "answer": message["content"]
                    })
        
        return tasks
    
    def apply_results(self, tasks: List[Dict[str, Any]], results: List[Any], 
                     model: str) -> List[Dict[str, Any]]:
        """
        Apply classification results to the original samples.
        
        Args:
            tasks: Original classification tasks
            results: Classification results from LLM
            model: Model name used for classification
            
        Returns:
            List of updated samples
        """
        # Group tasks by sample
        sample_updates = {}
        
        for task, result in zip(tasks, results):
            sample_id = task["sample"]["conversation_id"]
            if sample_id not in sample_updates:
                sample_updates[sample_id] = {
                    "sample": task["sample"],
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
                "message": task["message"],
                "metadata": classification_metadata
            })
        
        # Apply updates to samples
        updated_samples = []
        for sample_data in sample_updates.values():
            sample = sample_data["sample"].copy()
            
            # Apply metadata updates to messages
            for update in sample_data["updates"]:
                target_message = update["message"]
                classification_metadata = update["metadata"]
                
                # Find and update the message in conversation branches
                for branch in sample.get("conversation_branches", []):
                    for i, message in enumerate(branch.get("messages", [])):
                        if (message["content"] == target_message["content"] and 
                            message["role"] == target_message["role"]):
                            
                            # Initialize metadata if not present
                            if "metadata" not in branch["messages"][i]:
                                branch["messages"][i]["metadata"] = {}
                            
                            # Create nested structure: refusal_classification -> model -> data
                            if "refusal_classification" not in branch["messages"][i]["metadata"]:
                                branch["messages"][i]["metadata"]["refusal_classification"] = {}
                            
                            branch["messages"][i]["metadata"]["refusal_classification"][model] = classification_metadata
                            break
            
            updated_samples.append(sample)
        
        return updated_samples


def main():
    """Main function."""
    classifier = RefusalClassifier()
    return classifier.run_classification()


if __name__ == "__main__":
    sys.exit(main())