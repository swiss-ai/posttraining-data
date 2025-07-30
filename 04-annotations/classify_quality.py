#!/usr/bin/env python3
"""
Question Quality Classification Script

Evaluates the quality of initial prompts in chat format datasets across four
dimensions: well-formedness, clarity of intent, answerable scope, and completeness.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from base_classifier import BaseClassifier
import json
import asyncio
from tqdm import tqdm


class QualityClassifier(BaseClassifier):
    """Classifier for evaluating question quality across multiple dimensions."""
    
    def __init__(self):
        super().__init__(
            classifier_name="quality_classification",
            template_filename="quality.txt",
            valid_categories=[],  # Quality has structured JSON output, not simple categories
            description="Evaluate question quality in chat format datasets"
        )
    
    def collect_tasks(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Collect classification task for the initial prompt quality evaluation.
        
        Args:
            sample: Chat format sample
            
        Returns:
            Single task dict or None if no initial prompt
        """
        # Extract initial prompt content
        initial_content = ""
        if sample.get("initial_prompt") and sample["initial_prompt"].get("content"):
            initial_content = sample["initial_prompt"]["content"]
        
        if not initial_content.strip():
            return None
        
        # Return task with question for evaluation
        return {
            "sample": sample,
            "question": initial_content
        }
    
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
            sample = task["sample"].copy()
            
            if result.success and result.reasoning:
                # The quality structure is in the reasoning field as JSON string
                try:
                    quality_structure = json.loads(result.reasoning)
                    quality_data = quality_structure
                except json.JSONDecodeError:
                    quality_data = {
                        "error": "Failed to parse quality JSON response",
                        "raw_response": result.reasoning
                    }
            else:
                # Classification failed - provide error structure matching expected format
                error_msg = result.error if result.error else "Quality assessment failed"
                quality_data = {
                    "well_formedness": {"reasoning": error_msg, "score": "error"},
                    "clarity_of_intent": {"reasoning": error_msg, "score": "error"},
                    "answerable_scope": {"reasoning": error_msg, "score": "error"},
                    "completeness": {"reasoning": error_msg, "score": "error"}
                }
            
            # Add to initial_prompt metadata with nested structure
            if "initial_prompt" in sample and sample["initial_prompt"]:
                if "metadata" not in sample["initial_prompt"]:
                    sample["initial_prompt"]["metadata"] = {}
                
                # Create nested structure: quality_classification -> model -> data
                if "quality_classification" not in sample["initial_prompt"]["metadata"]:
                    sample["initial_prompt"]["metadata"]["quality_classification"] = {}
                
                sample["initial_prompt"]["metadata"]["quality_classification"][model] = quality_data
            
            updated_samples.append(sample)
        
        return updated_samples


def main():
    """Main function."""
    classifier = QualityClassifier()
    return classifier.run_classification()


if __name__ == "__main__":
    sys.exit(main())