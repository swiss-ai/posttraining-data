#!/usr/bin/env python3
"""
Question Complexity Classification Script

Evaluates initial prompts in chat format datasets (with parts structure) 
across three dimensions: complexity, completeness, and quality.
Adds computed fields for complexity × quality and complexity × quality × completeness.

Only processes initial prompts.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import copy

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from base_classifier import BaseClassifier
import json
import asyncio
from tqdm import tqdm


class ComplexityClassifier(BaseClassifier):
    """Classifier for evaluating question complexity, completeness, and quality."""
    
    def __init__(self):
        super().__init__(
            classifier_name="complexity_classification",
            template_filename="complexity_quality_completeness.txt",
            valid_categories=[],  # Complexity has structured JSON output, not simple categories
            description="Evaluate question complexity, completeness, and quality in chat format datasets (with parts structure)"
        )
    
    def collect_tasks(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect classification task for the initial prompt complexity evaluation.
        
        Args:
            sample: Chat format sample with parts structure
            
        Returns:
            List containing single task dict, or empty list if no initial prompt
        """
        # Extract initial prompt content
        initial_content = ""
        if sample.get("initial_prompt") and sample["initial_prompt"].get("content"):
            initial_content = sample["initial_prompt"]["content"]
        
        if not initial_content.strip():
            return []
        
        # Return list with single task
        return [{
            "sample": sample,
            "question": initial_content
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
            
            if result.success and result.reasoning:
                # The complexity structure is in the reasoning field as JSON string
                try:
                    complexity_structure = json.loads(result.reasoning)
                    complexity_data = complexity_structure
                    
                    # Add computed fields
                    complexity_score = complexity_data.get("complexity", {}).get("score")
                    completeness_score = complexity_data.get("completeness", {}).get("score") 
                    quality_score = complexity_data.get("quality", {}).get("score")
                    
                    # Calculate computed fields if all scores are numeric
                    if (isinstance(complexity_score, int) and 
                        isinstance(completeness_score, int) and 
                        isinstance(quality_score, int)):
                        
                        complexity_data["complexity_x_quality"] = complexity_score * quality_score
                        complexity_data["complexity_x_quality_x_completeness"] = complexity_score * quality_score * completeness_score
                    else:
                        # Set computed fields to error if any input score is non-numeric
                        complexity_data["complexity_x_quality"] = "error"
                        complexity_data["complexity_x_quality_x_completeness"] = "error"
                        
                except json.JSONDecodeError:
                    complexity_data = {
                        "error": "Failed to parse complexity JSON response",
                        "raw_response": result.reasoning
                    }
            else:
                # Classification failed - provide error structure matching expected format
                error_msg = result.error if result.error else "Complexity assessment failed"
                complexity_data = {
                    "complexity": {"reasoning": error_msg, "score": "error"},
                    "completeness": {"reasoning": error_msg, "score": "error"},
                    "quality": {"reasoning": error_msg, "score": "error"},
                    "complexity_x_quality": "error",
                    "complexity_x_quality_x_completeness": "error"
                }
            
            # Add to initial_prompt metadata with nested structure
            if "initial_prompt" in sample and sample["initial_prompt"]:
                if "metadata" not in sample["initial_prompt"]:
                    sample["initial_prompt"]["metadata"] = {}
                
                # Create nested structure: complexity_classification -> model -> data
                if "complexity_classification" not in sample["initial_prompt"]["metadata"]:
                    sample["initial_prompt"]["metadata"]["complexity_classification"] = {}
                
                sample["initial_prompt"]["metadata"]["complexity_classification"][model] = complexity_data
            
            updated_samples.append(sample)
        
        return updated_samples


def main():
    """Main function."""
    classifier = ComplexityClassifier()
    return classifier.run_classification()


if __name__ == "__main__":
    sys.exit(main())