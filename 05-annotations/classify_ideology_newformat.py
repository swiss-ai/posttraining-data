#!/usr/bin/env python3
"""
Ideological Classification Script

Classifies the ideological sensitivity of initial prompts in chat format datasets
(with parts structure) on a scale of 0-3, where 0 indicates no ideological sensitivity 
and 3 indicates high sensitivity where ideological position would fundamentally shape responses.

Only processes initial prompts.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import copy

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from base_classifier import BaseClassifier


class IdeologyClassifier(BaseClassifier):
    """Classifier for ideological sensitivity of initial prompts."""
    
    def __init__(self):
        super().__init__(
            classifier_name="ideological_classification",
            template_filename="ideological.txt",
            valid_categories=["0", "1", "2", "3"],
            description="Classify ideological sensitivity in chat format datasets (with parts structure)"
        )
    
    def collect_tasks(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect classification task for the initial prompt of a sample.
        
        Args:
            sample: Chat format sample with parts structure
            
        Returns:
            List containing single task dict, or empty list if no initial prompt
        """
        # Extract system prompt content (if exists)
        system_content = ""
        if sample.get("system_prompt") and sample["system_prompt"].get("content"):
            system_content = sample["system_prompt"]["content"]
        
        # Extract initial prompt content
        initial_content = ""
        if sample.get("initial_prompt") and sample["initial_prompt"].get("content"):
            initial_content = sample["initial_prompt"]["content"]
        
        # Return list with single task
        return [{
            "sample": sample,
            "system_prompt": system_content,
            "initial_prompt": initial_content
        }]
    
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
        updated_samples = []
        
        for task, result in zip(tasks, results):
            sample = copy.deepcopy(task["sample"])  # Prevent Arrow corruption
            
            # Convert string classification to integer
            try:
                classification_int = int(result.classification)
            except (ValueError, TypeError):
                classification_int = 0  # Default to 0 if parsing fails
            
            # Create classification data structure
            classification_data = {
                "classification": classification_int,
                "reasoning": result.reasoning
            }
            
            # Add error info if classification failed
            if not result.success:
                classification_data["error"] = result.error if result.error else "Classification failed"
                classification_data["success"] = False
            
            # Add to initial_prompt metadata with nested structure
            if "initial_prompt" in sample and sample["initial_prompt"]:
                if "metadata" not in sample["initial_prompt"]:
                    sample["initial_prompt"]["metadata"] = {}
                
                # Create nested structure: ideological_classification -> model -> data
                if "ideological_classification" not in sample["initial_prompt"]["metadata"]:
                    sample["initial_prompt"]["metadata"]["ideological_classification"] = {}
                
                sample["initial_prompt"]["metadata"]["ideological_classification"][model] = classification_data
            
            updated_samples.append(sample)
        
        return updated_samples


def main():
    """Main function."""
    classifier = IdeologyClassifier()
    return classifier.run_classification()


if __name__ == "__main__":
    sys.exit(main())