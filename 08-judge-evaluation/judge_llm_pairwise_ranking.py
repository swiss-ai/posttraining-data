#!/usr/bin/env python3
"""
Judge LLM Pairwise Ranking Evaluation

Evaluates a judge model's ability to rank completions using pairwise comparisons.
For each pair of completions, creates prompts in both directions:
- A: text1, B: text2 -> P(text1 > text2) estimate 1
- A: text2, B: text1 -> P(text1 > text2) estimate 2 = 1 - P(text2 > text1)
- Averages the two estimates to get final P(text1 > text2)
- Uses Bradley-Terry model via choix library to produce final ranking

Usage:
    python judge_llm_pairwise_ranking.py --samples 3  # Dev mode
    python judge_llm_pairwise_ranking.py --samples -1  # Full evaluation
"""

import argparse
import asyncio
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.stats import spearmanr, kendalltau
import choix

from judge_lib import (
    SyntheticDatasetLoader, InstructionsLoader, EvaluationAnalyzer,
    ReportGenerator, LLMClient, ConcurrentEvaluator, JudgeEvaluationUtils
)

# Configuration
DATASET_PATH = "./data/olmo-2-preference-quality-short-1100-synthetic"
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_MAX_RETRIES = 2


def bradley_terry_ranking(pairwise_probs: np.ndarray, alpha: float = 0.0001, 
                         prob_scale: int = 50, use_binary: bool = True) -> np.ndarray:
    """
    Use Bradley-Terry model via choix to rank items from pairwise win probabilities.
    
    Args:
        pairwise_probs: n x n matrix where pairwise_probs[i, j] = P(item i beats item j)
        alpha: Regularization parameter for LSR
        prob_scale: Scale factor to convert probabilities to integer counts (ignored if use_binary=True)
        use_binary: If True, convert probabilities to binary outcomes (>0.5 = win, else loss)
    
    Returns:
        Array of scores
    """
    n = pairwise_probs.shape[0]
    
    if use_binary:
        # Binary: convert probabilities to binary win matrix
        win_matrix = (pairwise_probs > 0.5).astype(int)
    else:
        # Probabilistic: scale probabilities to integer counts
        win_matrix = np.round(pairwise_probs * prob_scale).astype(int)
    
    # Use choix LSR to estimate parameters
    scores = choix.ilsr_pairwise_dense(win_matrix, alpha=alpha)
    return scores



class JudgePairwiseRankingEvaluator:
    """Evaluates judge model with pairwise comparisons."""
    
    def __init__(self, model: str = DEFAULT_MODEL, enable_thinking: bool = False,
                 instructions_path: str = "judge_instructions/default.txt",
                 max_retries: int = DEFAULT_MAX_RETRIES, max_concurrent: int = 50,
                 debug_file_path: Optional[Path] = None):
        self.model = model
        self.enable_thinking = enable_thinking
        self.instructions_path = instructions_path
        
        # Initialize composable components
        self.llm_client = LLMClient(model=model, max_retries=max_retries, debug_file_path=debug_file_path)
        self.concurrent_evaluator = ConcurrentEvaluator(max_concurrent=max_concurrent)
        
        # Load judge instructions
        script_dir = Path(__file__).parent
        self.judge_instructions = InstructionsLoader.load_instructions(
            instructions_path, script_dir
        )
    
    def create_pairwise_prompts(self, sample: Dict) -> List[Tuple[str, int, int, bool]]:
        """
        Create pairwise comparison prompts for all completion pairs in both directions.
        
        Returns:
            List of (prompt, i, j, is_forward) where:
            - prompt: the comparison prompt
            - i, j: completion indices being compared
            - is_forward: True if A=completion_i, B=completion_j; False if A=completion_j, B=completion_i
        """
        completions = sample["completions"]
        n_completions = len(completions)
        prompts = []
        
        # For each unique pair, create both directions
        for i in range(n_completions):
            for j in range(i + 1, n_completions):
                # Forward direction: A=completion_i, B=completion_j
                completion_i = completions[i]['content']
                completion_j = completions[j]['content']
                
                prompt_forward = self._create_comparison_prompt(sample['question'], completion_i, completion_j)
                prompts.append((prompt_forward, i, j, True))
                
                # Reverse direction: A=completion_j, B=completion_i  
                prompt_reverse = self._create_comparison_prompt(sample['question'], completion_j, completion_i)
                prompts.append((prompt_reverse, i, j, False))
        
        return prompts
    
    def _create_comparison_prompt(self, question: str, completion_a: str, completion_b: str) -> str:
        """Create a single pairwise comparison prompt."""
        prompt_parts = [
            self.judge_instructions,
            "",
            "Compare the two completions below and determine which one is better.",
            "Respond with only 'A' if the first completion is better, or 'B' if the second completion is better.",
            "",
            "======",
            f"QUESTION: {question}",
            "======",
            f"COMPLETION A: {completion_a}",
            "======", 
            f"COMPLETION B: {completion_b}",
            "======",
        ]
        
        if not self.enable_thinking:
            prompt_parts.append("BETTER COMPLETION: ")
        
        return "\n".join(prompt_parts)
    
    def parse_preference(self, logprobs: list) -> Tuple[Optional[float], Optional[str]]:
        """Parse preference from model response. Returns (prob_A_better, error_detail)."""
        
        in_thinking = False
        for token_info in logprobs:
            tok = token_info.token.strip()
            if tok == "<think>":
                in_thinking = True
            elif tok == "</think>":
                in_thinking = False
            elif not in_thinking and tok.upper() in ['A', 'B']:
                # Found the preference token, now get probabilities
                prob_A = 0.0
                prob_B = 0.0
                total_prob = 0.0
                
                for logprob_info in token_info.top_logprobs:
                    lp_tok = logprob_info.token.strip().upper()
                    if lp_tok in ['A', 'B']:
                        prob = math.exp(logprob_info.logprob)
                        total_prob += prob
                        if lp_tok == 'A':
                            prob_A += prob
                        elif lp_tok == 'B':
                            prob_B += prob
                
                if total_prob > 0:
                    # Normalize and return probability that A is better
                    return prob_A / total_prob, None
                
        return None, "No valid preference (A/B) found in response."
    
    async def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample using pairwise comparisons."""
        # Create pairwise prompts
        pairwise_prompts = self.create_pairwise_prompts(sample)
        n_completions = len(sample["completions"])
        
        # Base result structure
        result = {
            "sample_id": sample['id'],
            "success": False,
            "retries": 0,
        }
        
        # Store estimates for each pair
        # pairwise_probs[i, j] will store list of estimates for P(completion_i > completion_j)
        pairwise_probs = np.zeros((n_completions, n_completions))
        debug_probs = []
        
        if self.enable_thinking:
            max_tokens = 2048
            temperature = 0.6
        else:
            max_tokens = 16
            temperature = 0.0
        
        # Evaluate all pairwise comparisons
        for prompt, i, j, is_forward in pairwise_prompts:
            llm_result = await self.llm_client.make_request(
                [{
                    "role": "user", 
                    "content": prompt,
                }],
                max_tokens=max_tokens,
                temperature=temperature,
                openai_kwargs={
                    "logprobs": True,
                    "top_logprobs": 20,
                    "extra_body": {
                        "chat_template_kwargs": {
                            "enable_thinking": self.enable_thinking,
                        }
                    }
                }
            )
            
            # Sum up tokens across all comparisons
            if "tokens" not in result:
                result["tokens"] = llm_result["tokens"]
            else:
                for k, v in llm_result["tokens"].items():
                    result["tokens"][k] += v
            
            # Add retry info if present
            if 'retries' in llm_result:
                result['retries'] += llm_result['retries']
            
            # Handle LLM request failure
            if not llm_result['success']:
                result.update({
                    "error": f"API Error on comparison {i} vs {j} (forward={is_forward})",
                    "error_detail": llm_result['error'],
                    "raw_response": None
                })
                return result
            
            # Parse preference probability
            prob_A_better, error_detail = self.parse_preference(llm_result['logprobs'])
            if prob_A_better is None:
                result.update({
                    "error": f"Failed to parse preference for comparison {i} vs {j} (forward={is_forward})",
                    "error_detail": error_detail
                })
                return result
            
            # Convert to estimate of P(completion_i > completion_j)
            if is_forward:
                # A=completion_i, B=completion_j, so P(A > B) = P(completion_i > completion_j)
                prob_i_beats_j = prob_A_better
            else:
                # A=completion_j, B=completion_i, so P(A > B) = P(completion_j > completion_i)
                # Therefore P(completion_i > completion_j) = 1 - P(completion_j > completion_i)
                prob_i_beats_j = 1.0 - prob_A_better
            
            # Store the estimatepai
            pairwise_probs[i,j] += prob_i_beats_j
            pairwise_probs[j,i] += 1.0 - prob_i_beats_j
            debug_probs.append((i, j, is_forward, prob_i_beats_j))
        
        # Use Bradley-Terry to get final ranking
        pairwise_probs /= 2 # Average the (A, B) and (B, A) estimates
        try:
            completion_scores = bradley_terry_ranking(pairwise_probs)
        except Exception as e:
            result.update({
                "error": "Bradley-Terry ranking failed",
                "error_detail": str(e)
            })
            return result
        
        # Ground truth: actual quality ranks from dataset
        ground_truth = sample['ground_truth']
        
        spearman_corr, _ = spearmanr(ground_truth, completion_scores)
        kendall_corr, _ = kendalltau(ground_truth, completion_scores)
        
        # Top-k accuracy
        completion_ranking = sorted(range(n_completions), key=lambda i: -completion_scores[i])  # Higher score = better completion
        best_3_indices = sorted(range(n_completions), key=lambda i: ground_truth[i])[:3]
        top3_correct = sum(1 for i in best_3_indices if completion_ranking[i] <= 3)
        
        result.update({
            "success": True,
            "ground_truth": ground_truth,
            "predicted": completion_ranking,
            "scores": completion_scores.tolist(),
            "pairwise_probs": pairwise_probs.tolist(),
            "spearman": -spearman_corr,
            "kendall": -kendall_corr,
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
    parser = argparse.ArgumentParser(description="Evaluate judge model pairwise ranking ability")
    parser.add_argument("--samples", type=int, default=3, 
                       help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help="Model to use for evaluation")
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                       help="Whether to turn on thinking")
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
    
    # Load dataset
    print(f"Loading dataset...")
    samples = loader.load_dataset(args.samples)
    print(f"Loaded {len(samples)} samples")
    
    # Create debug file path if debug is enabled
    debug_file_path = None
    if args.debug:
        config = {
            'model': args.model,
            'instructions': args.instructions,
            'thinking': args.enable_thinking,
        }
        base_name = utils.generate_output_filename("llm_pairwise_ranking", args.model, config, len(samples))
        debug_file_path = utils.create_debug_file_path(__file__, base_name)
        print(f"Debug logging enabled: {debug_file_path}")
    
    evaluator = JudgePairwiseRankingEvaluator(
        model=args.model, 
        enable_thinking=args.enable_thinking,
        instructions_path=args.instructions,
        max_retries=args.max_retries,
        max_concurrent=args.concurrent,
        debug_file_path=debug_file_path,
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
        'instructions': args.instructions,
        'thinking': args.enable_thinking,
    }
    base_name = utils.generate_output_filename("llm_pairwise_ranking", args.model, config, len(samples))
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