#!/usr/bin/env python3
"""
Judge LLM Ranking Evaluation

Evaluates a judge model's ability to rank 9 completions from best to worst.
Uses iterations 0-8 from synthetic preference dataset, mapping to ranks 1-9.

Usage:
    python judge_llm_ranking.py --samples 3  # Dev mode
    python judge_llm_ranking.py --samples -1  # Full evaluation
"""

import re
import random
import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, kendalltau

from judge_lib import (
    SyntheticDatasetLoader, InstructionsLoader, EvaluationAnalyzer,
    ReportGenerator, LLMClient, ConcurrentEvaluator, JudgeEvaluationUtils
)

# Configuration
DATASET_PATH = "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-preference-quality-short-1100-synthetic"
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_MAX_RETRIES = 2


class JudgeRankingEvaluator:
    """Evaluates judge model's ranking ability using composition."""
    
    def __init__(self, model: str = DEFAULT_MODEL, label_type: str = "numeric", 
                 instructions_path: str = "judge_instructions/default.txt", 
                 max_retries: int = DEFAULT_MAX_RETRIES, max_concurrent: int = 50,
                 debug_file_path: Optional[Path] = None):
        self.model = model
        self.label_type = label_type
        self.instructions_path = instructions_path
        
        # Initialize composable components
        self.llm_client = LLMClient(model=model, max_retries=max_retries, debug_file_path=debug_file_path)
        self.concurrent_evaluator = ConcurrentEvaluator(max_concurrent=max_concurrent)
        
        # Load judge instructions
        script_dir = Path(__file__).parent
        self.judge_instructions = InstructionsLoader.load_instructions(
            instructions_path, script_dir
        )
        
        # Label mappings
        if label_type == "alphabetic":
            self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
            self.label_to_num = {label: i+1 for i, label in enumerate(self.labels)}
        else:
            self.labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            self.label_to_num = {label: int(label) for label in self.labels}
    
    def create_prompt(self, sample: Dict, presentation_order: List[int]) -> str:
        """Create ranking prompt with clear boundaries."""
        prompt_parts = [
            self.judge_instructions,
            "",
            "Rank the following 9 completions from best to worst.",
            "",
            f"Question: {sample['question']}",
            ""
        ]
        
        # Add completions in presentation order (presentation_order contains completion indices)
        for idx, completion_idx in enumerate(presentation_order):
            label = self.labels[idx]
            content = sample['completions'][completion_idx]['content']
            
            prompt_parts.append(f"=== COMPLETION {label} ===")
            prompt_parts.append(content)
            prompt_parts.append(f"=== END COMPLETION {label} ===")
            prompt_parts.append("")
        
        # Add instructions with clear template
        if self.label_type == "numeric":
            prompt_parts.extend([
                "Instructions:",
                "Follow this exact format:",
                "",
                "REASONING:",
                "[Provide your detailed reasoning for ranking each completion]",
                "",
                "RANKING: [x, x, x, x, x, x, x, x, x]",
                "",
                "Ranking requirements:",
                "- Each x must be a single digit from 1-9 (1=best, 9=worst)",
                "- Must use exactly 9 numbers, no more, no less",
                "- Each number 1-9 must appear exactly once (no duplicates, no missing values)",
                "- The position in the list corresponds to the completion (1st position = Completion 1, 2nd position = Completion 2, etc.)"
            ])
        else:
            prompt_parts.extend([
                "Instructions:",
                "Follow this exact format:",
                "",
                "REASONING:",
                "[Provide your detailed reasoning for ranking each completion]",
                "",
                "RANKING: [x, x, x, x, x, x, x, x, x]",
                "",
                "Ranking requirements:",
                "- Each x must be a single letter from A-I (A=best, I=worst)",
                "- Must use exactly 9 letters, no more, no less",
                "- Each letter A-I must appear exactly once (no duplicates, no missing values)",
                "- The position in the list corresponds to the completion (1st position = Completion A, 2nd position = Completion B, etc.)"
            ])
        
        return "\n".join(prompt_parts)
    
    def parse_ranking(self, response: str) -> Tuple[Optional[List[int]], Optional[str]]:
        """Parse ranking from model response. Returns (ranking, error_detail)."""
        # Check for REASONING: marker (optional, but good to track)
        has_reasoning = re.search(r'REASONING:', response, re.IGNORECASE)
        
        # Look for RANKING: marker
        ranking_match = re.search(r'RANKING:\s*\[(.*?)\]', response, re.IGNORECASE | re.DOTALL)
        if not ranking_match:
            error_msg = "No RANKING: [...] found in response"
            if not has_reasoning:
                error_msg += " (also missing REASONING: section)"
            return None, error_msg
        
        ranking_str = ranking_match.group(1)
        
        # Parse based on label type
        try:
            if self.label_type == "alphabetic":
                # Parse alphabetic labels
                items = [item.strip() for item in ranking_str.split(',')]
                ranking = []
                invalid_labels = []
                for item in items:
                    # Handle potential quotes
                    item = item.strip('"\'')
                    if item in self.label_to_num:
                        ranking.append(self.label_to_num[item])
                    else:
                        invalid_labels.append(item)
                
                if invalid_labels:
                    return None, f"Invalid alphabetic labels: {invalid_labels}"
            else:
                # Parse numeric labels
                try:
                    ranking = [int(item.strip()) for item in ranking_str.split(',')]
                except ValueError as e:
                    return None, f"Non-numeric values in ranking: {ranking_str}"
            
            # Validate ranking
            if len(ranking) != 9:
                return None, f"Wrong number of elements: expected 9, got {len(ranking)}"
            
            # Check for duplicates
            if len(set(ranking)) != 9:
                duplicates = [x for x in ranking if ranking.count(x) > 1]
                return None, f"Duplicate values: {list(set(duplicates))}"
            
            # Check range
            invalid_range = [x for x in ranking if x < 1 or x > 9]
            if invalid_range:
                return None, f"Values outside 1-9 range: {invalid_range}"
            
            # Check for missing values
            missing = list(set(range(1, 10)) - set(ranking))
            if missing:
                return None, f"Missing values: {missing}"
            
            return ranking, None
            
        except Exception as e:
            return None, f"Parse exception: {str(e)}"
    
    async def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample."""
        # Create random presentation order to avoid position bias (shuffle completion indices)
        presentation_order = list(range(9))  # Indices 0-8 for the 9 completions
        random.shuffle(presentation_order)
        
        # Create prompt
        prompt = self.create_prompt(sample, presentation_order)
        
        # Make LLM request using the centralized client
        llm_result = await self.llm_client.make_request([{
            "role": "user", 
            "content": prompt
        }], temperature=0.0, max_tokens=10000)
        
        # Base result structure
        result = {
            "sample_id": sample['id'],
            "success": False,
            "tokens": llm_result['tokens'],
            "presentation_order": presentation_order
        }
        
        # Add retry info if present
        if 'retries' in llm_result:
            result['retries'] = llm_result['retries']
        
        # Handle LLM request failure
        if not llm_result['success']:
            result.update({
                "error": "API Error",
                "error_detail": llm_result['error'],
                "raw_response": None
            })
            return result
        
        raw_response = llm_result['content']
        result['raw_response'] = raw_response
        
        # Parse ranking
        predicted_ranking, error_detail = self.parse_ranking(raw_response)
        
        if predicted_ranking is None:
            result.update({
                "error": "Failed to parse valid ranking",
                "error_detail": error_detail
            })
            return result
        
        # Map predicted ranking back to original completion order
        # predicted_ranking[i] is the rank assigned to completion at presentation position i
        # We need to map this back to the original completion order (0-8)
        completion_ranking = [0] * 9
        for pos_idx, rank in enumerate(predicted_ranking):
            completion_idx = presentation_order[pos_idx]  # Which original completion is at this position
            completion_ranking[completion_idx] = rank     # Assign rank to original completion
        
        # Ground truth: actual quality ranks from dataset (already in completion order)
        ground_truth = sample['ground_truth']
        
        # Calculate metrics
        spearman_corr, _ = spearmanr(ground_truth, completion_ranking)
        kendall_corr, _ = kendalltau(ground_truth, completion_ranking)
        
        # Top-k accuracy: count how many of the top 3 actual best completions got ranks 1-3
        # Ground truth ranks: lower is better (1=best, 9=worst)
        # Find which completions should be top 3
        best_3_indices = sorted(range(9), key=lambda i: ground_truth[i])[:3]
        top3_correct = sum(1 for i in best_3_indices if completion_ranking[i] <= 3)
        
        result.update({
            "success": True,
            "ground_truth": ground_truth,
            "predicted": completion_ranking,
            "spearman": spearman_corr,
            "kendall": kendall_corr,
            "top3_correct": top3_correct,
            "error": None
        })
        
        return result
    
    async def evaluate_all(self, samples: List[Dict], max_concurrent: Optional[int] = None) -> List[Dict]:
        """Evaluate all samples using concurrent processing."""
        return await self.concurrent_evaluator.evaluate_all_concurrent(
            samples, self.evaluate_sample, max_concurrent
        )


async def main():
    parser = argparse.ArgumentParser(description="Evaluate judge model ranking ability")
    parser.add_argument("--samples", type=int, default=3, 
                       help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help="Model to use for evaluation")
    parser.add_argument("--label-type", type=str, default="numeric",
                       choices=["numeric", "alphabetic"],
                       help="Label type for completions")
    parser.add_argument("--concurrent", type=int, default=50,
                       help="Maximum concurrent API requests")
    parser.add_argument("--instructions", type=str, default="judge_instructions/default.txt",
                       help="Path to judge instructions file")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                       help="Maximum number of retries for failed samples")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging of all LLM requests and responses")
    
    args = parser.parse_args()
    
    # Initialize components
    loader = SyntheticDatasetLoader(DATASET_PATH)
    analyzer = EvaluationAnalyzer()
    reporter = ReportGenerator()
    utils = JudgeEvaluationUtils()
    
    # Load dataset to get sample count for filename generation
    print(f"Loading dataset...")
    samples = loader.load_dataset(args.samples)
    print(f"Loaded {len(samples)} samples")
    
    # Create debug file path if debug is enabled
    debug_file_path = None
    if args.debug:
        config = {
            'model': args.model,
            'label_type': args.label_type,
            'instructions': args.instructions
        }
        base_name = utils.generate_output_filename("llm_ranking", args.model, config, len(samples))
        debug_file_path = utils.create_debug_file_path(__file__, base_name)
        print(f"Debug logging enabled: {debug_file_path}")
    
    evaluator = JudgeRankingEvaluator(
        model=args.model, 
        label_type=args.label_type, 
        instructions_path=args.instructions, 
        max_retries=args.max_retries,
        max_concurrent=args.concurrent,
        debug_file_path=debug_file_path
    )
    
    # Evaluate
    results = await evaluator.evaluate_all(samples, max_concurrent=args.concurrent)
    
    # Calculate metrics
    metrics = analyzer.calculate_overall_metrics(results)
    
    # Print final results to terminal
    print("\n=== FINAL RESULTS ===")
    print(f"Success Rate: {metrics['success_rate']:.1%} ({metrics['successful']}/{metrics['total_samples']})")
    print(f"Failed: {metrics['failed_rate']:.1%} ({metrics['failed']}/{metrics['total_samples']})")
    if metrics['successful'] > 0:
        print(f"Mean Spearman: {metrics['mean_spearman']:.3f}")
        print(f"Mean Kendall's Tau: {metrics['mean_kendall']:.3f}")
    
    # Generate output filenames
    config = {
        'model': args.model,
        'label_type': args.label_type,
        'instructions': args.instructions
    }
    base_name = utils.generate_output_filename("llm_ranking", args.model, config, len(samples))
    output_dir = utils.create_output_directory(__file__)
    
    # Save results
    jsonl_path = output_dir / f"{base_name}.jsonl"
    with open(jsonl_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Generate report
    report_path = output_dir / f"{base_name}.md"
    reporter.generate_report(results, metrics, report_path, config)
    
    print(f"\nReport saved to: {report_path}")
    print(f"Results saved to: {jsonl_path}")


if __name__ == "__main__":
    asyncio.run(main())