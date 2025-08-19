#!/usr/bin/env python3
"""
Judge LLM Scoring Evaluation

Evaluates a judge model's ability to score completions using probability-weighted scoring.
Scores each completion individually on a 1-9 scale (9=best, 1=worst) and can either:
- Use the modal response (single predicted score)
- Calculate weighted mean using probability distribution across valid scores

Usage:
    python judge_llm_scoring.py --samples 3  # Dev mode with probability weighting
    python judge_llm_scoring.py --samples -1  # Full evaluation
    python judge_llm_scoring.py --samples 10 --use-modal-response  # Use single score
"""

import re
import random
import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from scipy.stats import spearmanr, kendalltau, rankdata

from lib import (
    SyntheticDatasetLoader, InstructionsLoader, EvaluationAnalyzer,
    ReportGenerator, LLMClient, ConcurrentEvaluator, JudgeEvaluationUtils
)

# Configuration
DATASET_PATH = "./data/olmo-2-preference-quality-short-1100-synthetic"
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_MAX_RETRIES = 2


class JudgeScoringEvaluator:
    """Evaluates judge model with direct scoring."""
    
    def __init__(self, model: str = DEFAULT_MODEL, enable_thinking: bool = False, scale_range: Tuple[int, int] = (1, 9), 
                 instructions_path: str = "prompts/principles.txt", use_modal_response: bool = False,
                 max_retries: int = DEFAULT_MAX_RETRIES, max_concurrent: int = 50,
                 debug_file_path: Optional[Path] = None):
        self.model = model
        self.enable_thinking = enable_thinking
        self.instructions_path = instructions_path
        self.use_modal_response = use_modal_response
        
        # Initialize composable components
        self.llm_client = LLMClient(model=model, max_retries=max_retries, debug_file_path=debug_file_path)
        self.concurrent_evaluator = ConcurrentEvaluator(max_concurrent=max_concurrent)
        
        # Load judge instructions
        script_dir = Path(__file__).parent
        self.judge_instructions = InstructionsLoader.load_instructions(
            instructions_path, script_dir
        )
        
        # Label mappings
        assert scale_range[0] < scale_range[1], "Scale range must be valid (start < end)"
        self.scale = list(range(*scale_range))
    
    def create_prompts(self, sample: Dict) -> str:
        """Create scoring prompt with clear boundaries."""
        l, h = min(self.scale), max(self.scale)
        prompts = []

        for completion in sample["completions"]:
            prompt_parts = [
                self.judge_instructions,
                "",
                f"Based on these quality indicators, label the following completion on a scale from {l} to {h}, where {h} is best and {l} is worst.",
                "Do not use any reasoning and respond with only the quality label."
                "",
                "======",
                f"QUESTION: {sample['question']}",
                ""
            ]

            # TODO: add few-shot examples
        
            content = completion['content']
            prompt_parts.append(f"======")
            prompt_parts.append(f"COMPLETION: {content}")   
            prompt_parts.append(f"======")
            if not self.enable_thinking:
                prompt_parts.append("LABEL: ")
            prompts.append("\n".join(prompt_parts))
        return prompts
    
    def parse_score(self, logprobs: list) -> Tuple[Optional[List[int]], Optional[str]]:
        """Parse ranking from model response. Returns (ranking, error_detail)."""

        # identify the token with the answer by finding the first occurence of an integer
        in_thinking = False
        scale_str = [str(x) for x in self.scale]
        for token_info in logprobs:
            tok = token_info.token.strip()
            if tok == "<think>":
                in_thinking = True
            elif tok == "</think>":
                in_thinking = False

            # parse the score
            elif not in_thinking and tok in scale_str:
                score = int(tok)
                if self.use_modal_response:
                    return score, None
                else:
                    # take the expectation over token probabilities
                    total_prob = 0
                    scale_probs = {}
                    for logprob_info in token_info.top_logprobs:
                        lp_tok = logprob_info.token.strip()
                        if lp_tok in scale_str:
                            prob = math.exp(logprob_info.logprob)
                            total_prob += prob
                            scale_probs[int(lp_tok)] = prob
                    weighted_mean = sum(s * (p / total_prob) for s, p in scale_probs.items())
                    return weighted_mean, None
            
        return None, f"No valid score found in response."
    
    async def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample."""
        # Create prompt
        prompts = self.create_prompts(sample)

        # Base result structure
        result = {
            "sample_id": sample['id'],
            "success": False,
            "retries": 0,
        }
        completion_scores = []

        if self.enable_thinking:
            max_tokens = 2048
            temperature = 0.6
            # TOFIX: For thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0
        else:
            max_tokens = 256
            temperature = 0.0

        # Make LLM request using the centralized client
        for idx, prompt in enumerate(prompts):
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
            
            # Sum up tokens across all completions
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
                    "error": f"API Error on idx {idx}",
                    "error_detail": llm_result['error'],
                    "raw_response": None
                })
                return result
            
            raw_response = llm_result['content']
            result['raw_response'] = raw_response

            # Parse score
            predicted_score, error_detail = self.parse_score(llm_result['logprobs'])
            if predicted_score is None:
                result.update({
                    "error": f"Failed to parse valid score for {idx}",
                    "error_detail": error_detail
                })
                return result
            
            completion_scores.append(predicted_score)
        
        # Ground truth: actual quality ranks from dataset (already in completion order)
        ground_truth = sample['ground_truth']
        
        # Calculate metrics
        if len(set(completion_scores)) == 1:
            # All scores identical, correlation undefined (could also set to 0?)
            result.update({
                "error": "All predicted scores identical, correlation undefined",
                "error_detail": "Undefined correlation"
            })
            return result
        else:
            spearman_corr, _ = spearmanr(ground_truth, completion_scores)
            kendall_corr, _ = kendalltau(ground_truth, completion_scores)
        
        # Top-k accuracy: count how many of the top 3 actual best completions got ranks 7-9
        # Ground truth ranks: higher is better (9=best, 1=worst)
        # Scores: higher is better (9=best, 1=worst) - same scale now
        # Find which completions should be top 3
        completion_ranking = rankdata(completion_scores).astype(int).tolist()
        n_completions = len(completion_ranking)
        best_3_indices = sorted(range(n_completions), key=lambda i: ground_truth[i], reverse=True)[:3]
        top3_threshold = max(1, n_completions - 2)
        top3_correct = sum(1 for i in best_3_indices if completion_ranking[i] >= top3_threshold)

        result.update({
            "success": True,
            "ground_truth": ground_truth,
            "predicted": completion_ranking,
            "scores": completion_scores,
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
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                       help="Whether to turn on thinking")
    parser.add_argument("--use-modal-response", action="store_true", default=False,
                       help="Whether to use the modal response rather than the logprobs")
    parser.add_argument("--scale-range", type=int, nargs=2, default=(1, 9),
                       help="Scale range for ranking (start end), e.g. 1 9")
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
            'instructions': args.instructions,
            'modal': args.use_modal_response,
            'thinking': args.enable_thinking,
        }
        base_name = utils.generate_output_filename("llm_scoring", args.model, config, len(samples))
        debug_file_path = utils.create_debug_file_path(__file__, base_name)
        print(f"Debug logging enabled: {debug_file_path}")
    
    evaluator = JudgeScoringEvaluator(
        model=args.model, 
        enable_thinking=args.enable_thinking,
        scale_range=args.scale_range, 
        instructions_path=args.instructions,
        use_modal_response=args.use_modal_response,
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
        'modal': args.use_modal_response,
        'thinking': args.enable_thinking,
    }
    base_name = utils.generate_output_filename("llm_scoring", args.model, config, len(samples))
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