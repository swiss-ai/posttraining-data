#!/usr/bin/env python3
"""
Judge LLM Ranking Direct Evaluation

Evaluates a judge model's ability to rank 9 completions using a naive scoring method.
The model scores each completion individually with a single digit number (1-9),
then we use those scores to create a ranking.

Usage:
    python judge_llm_ranking_direct.py --samples 3  # Dev mode
    python judge_llm_ranking_direct.py --samples -1  # Full evaluation
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

from lib import (
    SyntheticDatasetLoader, InstructionsLoader, EvaluationAnalyzer,
    ReportGenerator, LLMClient, ConcurrentEvaluator, JudgeEvaluationUtils
)

# Configuration
DATASET_PATH = "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-preference-quality-short-1100-synthetic"
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_MAX_RETRIES = 2


class JudgeDirectScoringEvaluator:
    """Evaluates judge model's ranking ability using direct scoring method."""
    
    def __init__(self, model: str = DEFAULT_MODEL, 
                 instructions_path: str = "prompts/principles.txt", 
                 max_retries: int = DEFAULT_MAX_RETRIES, max_concurrent: int = 50,
                 debug_file_path: Optional[Path] = None):
        self.model = model
        self.instructions_path = instructions_path
        
        # Initialize composable components
        self.llm_client = LLMClient(model=model, max_retries=max_retries, debug_file_path=debug_file_path)
        self.concurrent_evaluator = ConcurrentEvaluator(max_concurrent=max_concurrent)
        
        # Load judge instructions
        script_dir = Path(__file__).parent
        self.judge_instructions = InstructionsLoader.load_instructions(
            instructions_path, script_dir
        )
    
    def create_scoring_prompt(self, sample: Dict, completion_idx: int) -> str:
        """Create scoring prompt for a single completion."""
        completion = sample['completions'][completion_idx]
        
        prompt_parts = [
            self.judge_instructions,
            "",
            "Score the following completion on a scale of 1-9, where 9 is the best and 1 is the worst.",
            "",
            f"Question: {sample['question']}",
            "",
            "=== COMPLETION ===",
            completion['content'],
            "=== END COMPLETION ===",
            "",
            "Instructions:",
            "Provide a single digit score from 1-9 where:",
            "- 9 = Excellent (best possible response)",
            "- 2 = Very Good",
            "- 3 = Good", 
            "- 4 = Above Average",
            "- 5 = Average",
            "- 6 = Below Average",
            "- 7 = Poor",
            "- 8 = Very Poor",
            "- 1 = Terrible (worst possible response)",
            "",
            "IMPORTANT: Respond with ONLY a single digit from 1-9.",
            "Example: 5"
        ]
        
        return "\n".join(prompt_parts)
    
    def parse_score(self, response: str) -> Tuple[Optional[int], Optional[str]]:
        """Parse score from model response. Returns (score, error_detail)."""
        # Clean the response
        response = response.strip()
        
        # First, try to find a single digit at the beginning or end (most common case)
        lines = response.split('\n')
        
        # Check first line for a single digit
        if lines and lines[0].strip().isdigit():
            score = int(lines[0].strip())
            if 1 <= score <= 9:
                return score, None
        
        # Check last line for a single digit
        if lines and lines[-1].strip().isdigit():
            score = int(lines[-1].strip())
            if 1 <= score <= 9:
                return score, None
        
        # Look for any single digit in the response
        digit_match = re.search(r'\b(\d)\b', response)
        if digit_match:
            score = int(digit_match.group(1))
            if 1 <= score <= 9:
                return score, None
        
        # Fallback: try to find SCORE: marker (in case model still uses it)
        score_match = re.search(r'SCORE:\s*(\d)', response, re.IGNORECASE)
        if score_match:
            try:
                score = int(score_match.group(1))
                if score < 1 or score > 9:
                    return None, f"Score outside 1-9 range: {score}"
                return score, None
            except ValueError:
                return None, f"Invalid score format: {score_match.group(1)}"
        
        return None, "No valid score found in response"
    
    async def score_completion(self, sample: Dict, completion_idx: int) -> Tuple[Optional[int], Dict]:
        """Score a single completion."""
        prompt = self.create_scoring_prompt(sample, completion_idx)
        
        # Make LLM request
        llm_result = await self.llm_client.make_request([{
            "role": "user", 
            "content": prompt
        }], temperature=0.0, max_tokens=1000)
        
        result = {
            "completion_idx": completion_idx,
            "success": False,
            "tokens": llm_result['tokens']
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
            return None, result
        
        raw_response = llm_result['content']
        result['raw_response'] = raw_response
        
        # Parse score (handle None response)
        if raw_response is None:
            result.update({
                "error": "Failed to parse valid score",
                "error_detail": "No response content received from LLM"
            })
            return None, result
        
        score, error_detail = self.parse_score(raw_response)
        
        if score is None:
            result.update({
                "error": "Failed to parse valid score",
                "error_detail": error_detail
            })
            return None, result
        
        result.update({
            "success": True,
            "score": score,
            "error": None
        })
        
        return score, result
    
    async def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample by scoring each completion individually."""
        # Score each completion
        scores = []
        scoring_results = []
        all_tokens = {'prompt': 0, 'completion': 0, 'total': 0}
        
        for completion_idx in range(9):
            score, result = await self.score_completion(sample, completion_idx)
            scoring_results.append(result)
            
            # Accumulate tokens
            all_tokens['prompt'] += result['tokens']['prompt']
            all_tokens['completion'] += result['tokens']['completion']
            all_tokens['total'] += result['tokens']['total']
            
            if score is not None:
                scores.append((completion_idx, score))
            else:
                # If any completion fails to score, the whole sample fails
                # Make sure we have token info for all completed requests
                for remaining_idx in range(completion_idx + 1, 9):
                    # Add zero tokens for remaining completions that weren't attempted
                    all_tokens['prompt'] += 0
                    all_tokens['completion'] += 0
                    all_tokens['total'] += 0
                
                return {
                    "sample_id": sample['id'],
                    "success": False,
                    "error": "Failed to score all completions",
                    "error_detail": f"Failed to score completion {completion_idx}: {result.get('error_detail', 'Unknown error')}",
                    "tokens": all_tokens,
                    "scoring_results": scoring_results
                }
        
        # Create ranking from scores (higher scores = better ranks)
        # Sort by score (descending) and assign ranks
        scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score
        
        # Create ranking: completion_idx -> rank
        completion_ranking = [0] * 9
        for rank, (completion_idx, _) in enumerate(scores, 1):
            completion_ranking[completion_idx] = rank
        
        # Ground truth: actual quality ranks from dataset
        ground_truth = sample['ground_truth']
        
        # Calculate metrics
        spearman_corr, _ = spearmanr(ground_truth, completion_ranking)
        kendall_corr, _ = kendalltau(ground_truth, completion_ranking)
        
        # Top-k accuracy: count how many of the top 3 actual best completions got ranks 7-9
        best_3_indices = sorted(range(9), key=lambda i: ground_truth[i], reverse=True)[:3]
        top3_correct = sum(1 for i in best_3_indices if completion_ranking[i] >= 7)
        
        return {
            "sample_id": sample['id'],
            "success": True,
            "ground_truth": ground_truth,
            "predicted": completion_ranking,
            "scores": dict(scores),  # completion_idx -> score
            "spearman": spearman_corr,
            "kendall": kendall_corr,
            "top3_correct": top3_correct,
            "tokens": all_tokens,
            "scoring_results": scoring_results,
            "error": None
        }
    
    async def evaluate_all(self, samples: List[Dict], max_concurrent: Optional[int] = None) -> List[Dict]:
        """Evaluate all samples using concurrent processing."""
        return await self.concurrent_evaluator.evaluate_all_concurrent(
            samples, self.evaluate_sample, max_concurrent
        )


async def main():
    parser = argparse.ArgumentParser(description="Evaluate judge model ranking ability using direct scoring")
    parser.add_argument("--samples", type=int, default=3, 
                       help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help="Model to use for evaluation")
    parser.add_argument("--concurrent", type=int, default=50,
                       help="Maximum concurrent API requests")
    parser.add_argument("--instructions", type=str, default="prompts/principles.txt",
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
            'method': 'direct_scoring',
            'instructions': args.instructions
        }
        base_name = utils.generate_output_filename("llm_ranking_direct", args.model, config, len(samples))
        debug_file_path = utils.create_debug_file_path(__file__, base_name)
        print(f"Debug logging enabled: {debug_file_path}")
    
    evaluator = JudgeDirectScoringEvaluator(
        model=args.model, 
        instructions_path=args.instructions, 
        max_retries=args.max_retries,
        max_concurrent=args.concurrent,
        debug_file_path=debug_file_path
    )
    
    # Evaluate
    results = await evaluator.evaluate_all(samples, max_concurrent=args.concurrent)
    
    # Calculate metrics
    metrics = analyzer.calculate_overall_metrics(results)
    
    # Ensure token metrics are always present (in case all results failed)
    if 'total_tokens' not in metrics:
        total_tokens = sum(r['tokens']['total'] for r in results)
        metrics.update({
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": total_tokens / len(results) if results else 0,
            "avg_prompt_tokens": np.mean([r['tokens']['prompt'] for r in results]) if results else 0,
            "avg_completion_tokens": np.mean([r['tokens']['completion'] for r in results]) if results else 0
        })
    
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
        'method': 'direct_scoring',
        'instructions': args.instructions
    }
    base_name = utils.generate_output_filename("llm_ranking_direct", args.model, config, len(samples))
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
