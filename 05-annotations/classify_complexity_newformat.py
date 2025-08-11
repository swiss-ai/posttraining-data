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
                    
                    # Get scores for computation BEFORE converting to strings
                    complexity_score = complexity_data.get("complexity", {}).get("score")
                    completeness_score = complexity_data.get("completeness", {}).get("score") 
                    quality_score = complexity_data.get("quality", {}).get("score")
                    
                    # Try to parse scores as integers for computation
                    try:
                        complexity_int = int(complexity_score) if complexity_score is not None else None
                        completeness_int = int(completeness_score) if completeness_score is not None else None
                        quality_int = int(quality_score) if quality_score is not None else None
                        
                        if complexity_int is not None and completeness_int is not None and quality_int is not None:
                            complexity_data["complexity_x_quality"] = str(complexity_int * quality_int)
                            complexity_data["complexity_x_quality_x_completeness"] = str(complexity_int * quality_int * completeness_int)
                        else:
                            complexity_data["complexity_x_quality"] = "error"
                            complexity_data["complexity_x_quality_x_completeness"] = "error"
                    except (ValueError, TypeError):
                        # If any score cannot be parsed as int, set computed fields to error
                        complexity_data["complexity_x_quality"] = "error"
                        complexity_data["complexity_x_quality_x_completeness"] = "error"
                    
                    # NOW convert all score fields to strings for consistency
                    if "complexity" in complexity_data and isinstance(complexity_data["complexity"], dict):
                        if "score" in complexity_data["complexity"]:
                            complexity_data["complexity"]["score"] = str(complexity_data["complexity"]["score"])
                    
                    if "completeness" in complexity_data and isinstance(complexity_data["completeness"], dict):
                        if "score" in complexity_data["completeness"]:
                            complexity_data["completeness"]["score"] = str(complexity_data["completeness"]["score"])
                    
                    if "quality" in complexity_data and isinstance(complexity_data["quality"], dict):
                        if "score" in complexity_data["quality"]:
                            complexity_data["quality"]["score"] = str(complexity_data["quality"]["score"])
                        
                except json.JSONDecodeError:
                    complexity_data = {
                        "error": "Failed to parse complexity JSON response",
                        "raw_response": result.reasoning,
                        "complexity": {"reasoning": "JSON parsing failed", "score": "error"},
                        "completeness": {"reasoning": "JSON parsing failed", "score": "error"},
                        "quality": {"reasoning": "JSON parsing failed", "score": "error"},
                        "complexity_x_quality": "error",
                        "complexity_x_quality_x_completeness": "error"
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