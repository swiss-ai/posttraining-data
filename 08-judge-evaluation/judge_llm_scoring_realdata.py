#!/usr/bin/env python3
"""
Judge LLM Scoring for Real Data

Applies individual LLM-based scoring (1-9 scale) to real datasets with multiple
conversation branches. Scores each branch independently and saves the annotated
dataset with score metadata.

Usage:
    python judge_llm_scoring_realdata.py --source-dataset /path/to/dataset --target-path /path/to/output --samples 10
    python judge_llm_scoring_realdata.py --source-dataset /path/to/dataset --target-path /path/to/output --samples -1  # Full dataset
"""

import re
import random
import argparse
import asyncio
import json
import math
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import numpy as np
from tqdm.asyncio import tqdm
from datasets import load_from_disk, Dataset, DatasetDict

from lib import (
    InstructionsLoader, EvaluationAnalyzer,
    ReportGenerator, LLMClient, ConcurrentEvaluator, JudgeEvaluationUtils
)

# Configuration
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_MAX_RETRIES = 3


class RealDatasetLoader:
    """Handles loading and preparation of real datasets with multiple conversation branches."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
    
    def load_dataset(self, num_samples: int) -> List[Dict]:
        """Load and prepare dataset samples."""
        ds = load_from_disk(self.dataset_path)
        
        # Determine how many samples to use
        total_samples = len(ds['train'])
        if num_samples == -1:
            num_samples = total_samples
        else:
            num_samples = min(num_samples, total_samples)
        
        samples = []
        for i in range(num_samples):
            sample = ds['train'][i]
            
            # Extract completions from all conversation branches
            completions = []
            
            for branch_idx, branch in enumerate(sample['conversation_branches']):
                if branch['messages']:
                    message = branch['messages'][0]  # Assume single message per branch
                    
                    # Handle both old and new chat formats
                    if 'parts' in message:
                        # New format with parts
                        for part in message['parts']:
                            if part['type'] == 'response':
                                completion_data = {
                                    'content': part['content'],
                                    'branch_idx': branch_idx,
                                    'original_metadata': part.get('metadata', {})
                                }
                                completions.append(completion_data)
                                break
                    else:
                        # Old format with direct content
                        completion_data = {
                            'content': message.get('content', ''),
                            'branch_idx': branch_idx,
                            'original_metadata': message.get('metadata', {})
                        }
                        completions.append(completion_data)
            
            # Filter out empty completions and ensure we have valid content
            valid_completions = []
            for comp in completions:
                if comp['content'] and comp['content'].strip():
                    valid_completions.append(comp)
            
            if len(valid_completions) >= 1:  # Need at least 1 completion to score
                samples.append({
                    'id': f"{i:04d}",
                    'conversation_id': sample['conversation_id'],
                    'question': sample['initial_prompt']['content'],
                    'completions': valid_completions,
                    'original_sample': sample  # Preserve original for output
                })
        
        return samples


class DatasetScorer:
    """Handles saving scored datasets with preserved schema."""
    
    def __init__(self, target_path: str, judge_model: str, scoring_method: str):
        self.target_path = target_path
        self.judge_model = judge_model
        self.scoring_method = scoring_method
        self.timestamp = datetime.now().isoformat()
    
    def save_scored_dataset(self, samples: List[Dict], scoring_results: List[Dict]):
        """Save dataset with score annotations."""
        # Create mapping from sample_id to scoring results
        scoring_map = {r['sample_id']: r for r in scoring_results}
        
        # Process each sample and add scores
        processed_samples = []
        
        for sample in samples:
            sample_id = sample['id']
            original_sample = copy.deepcopy(sample['original_sample'])
            
            if sample_id in scoring_map and scoring_map[sample_id]['success']:
                scoring_result = scoring_map[sample_id]
                scores = scoring_result['scores']
                
                # Add score metadata to each conversation branch
                for comp_idx, completion in enumerate(sample['completions']):
                    branch_idx = completion['branch_idx']
                    
                    # Find the corresponding score for this completion
                    if comp_idx < len(scores):
                        score = scores[comp_idx]
                        
                        # Add score metadata to the original sample's branch
                        if branch_idx < len(original_sample['conversation_branches']):
                            branch = original_sample['conversation_branches'][branch_idx]
                            
                            # Add score metadata to the appropriate part/message
                            if branch['messages']:
                                message = branch['messages'][0]
                                
                                if 'parts' in message:
                                    # New format - add to response part
                                    for part in message['parts']:
                                        if part['type'] == 'response':
                                            if 'metadata' not in part:
                                                part['metadata'] = {}
                                            part['metadata']['score'] = score
                                            part['metadata']['scoring_metadata'] = {
                                                'judge_model': self.judge_model,
                                                'scoring_method': self.scoring_method,
                                                'timestamp': self.timestamp,
                                                'total_branches_scored': len(sample['completions'])
                                            }
                                            break
                                else:
                                    # Old format - add to message metadata
                                    if 'metadata' not in message:
                                        message['metadata'] = {}
                                    message['metadata']['score'] = score
                                    message['metadata']['scoring_metadata'] = {
                                        'judge_model': self.judge_model,
                                        'scoring_method': self.scoring_method,
                                        'timestamp': self.timestamp,
                                        'total_branches_scored': len(sample['completions'])
                                    }
            
            processed_samples.append(original_sample)
        
        # Create new dataset
        new_dataset = Dataset.from_list(processed_samples)
        dataset_dict = DatasetDict({'train': new_dataset})
        
        # Save to disk
        dataset_dict.save_to_disk(self.target_path)
        print(f"Scored dataset saved to: {self.target_path}")
        print(f"Total samples: {len(processed_samples)}")


class JudgeScoringRealDataEvaluator:
    """Evaluates real conversation branches using individual LLM scoring."""
    
    def __init__(self, model: str = DEFAULT_MODEL, enable_thinking: bool = False, scale_range: Tuple[int, int] = (1, 9), 
                 instructions_path: str = "08-judge-evaluation/prompts/charter-generic.txt", use_modal_response: bool = False,
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
        self.judge_instructions = InstructionsLoader.load_instructions(
            instructions_path, None
        )
        
        # Label mappings
        assert scale_range[0] < scale_range[1], "Scale range must be valid (start < end)"
        self.scale = list(range(*scale_range))
    
    def create_prompts(self, sample: Dict) -> List[str]:
        """Create scoring prompts for each completion."""
        l, h = min(self.scale), max(self.scale)
        prompts = []

        for completion in sample["completions"]:
            content = completion['content']
            
            prompt = f"""{self.judge_instructions}

Based on the above principles, rate the following response to the question on a scale from {l} to {h}, where {h} is best and {l} is worst.

QUESTION: {sample['question']}

RESPONSE: {content}

Rate this response from {l} to {h}.

Don't think or explain. Answer with only the number."""

            if not self.enable_thinking:
                prompt += "\n\nRATING: "
                
            prompts.append(prompt)
        return prompts
    
    def parse_score(self, logprobs: list) -> Tuple[Optional[float], Optional[str]]:
        """Parse score from model response. Returns (score, error_detail)."""

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
    
    async def score_sample(self, sample: Dict, debug_mode: bool = False, pbar=None) -> Dict:
        """Score all completions in a sample."""
        # Create prompts
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
        else:
            max_tokens = 256
            temperature = 0.0

        # Make LLM request for each completion
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
            
            # Update progress bar
            if pbar is not None:
                pbar.update(1)
            
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
                    "error": f"API Error on completion idx {idx}",
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
                    "error": f"Failed to parse valid score for completion {idx}",
                    "error_detail": error_detail
                })
                return result
            
            completion_scores.append(predicted_score)
        
        if debug_mode:
            print(f"\nSample {sample['id']} scored:")
            for idx, (completion, score) in enumerate(zip(sample['completions'], completion_scores)):
                print(f"  Completion {idx}: {score:.2f}")
        
        result.update({
            "success": True,
            "scores": completion_scores,
            "total_completions": len(completion_scores),
            "mean_score": np.mean(completion_scores),
            "error": None
        })
        
        return result
    
    async def evaluate_all(self, samples: List[Dict], max_concurrent: Optional[int] = None) -> List[Dict]:
        """Evaluate all samples with progress tracking."""
        # Calculate total LLM requests (one per completion)
        total_llm_requests = sum(len(sample['completions']) for sample in samples)
        
        print(f"Total LLM requests to make: {total_llm_requests}")
        
        pbar = tqdm(total=total_llm_requests, desc="LLM Requests", unit="req")
        
        async def evaluate_with_progress(sample):
            debug_mode = len(samples) == 1
            result = await self.score_sample(sample, debug_mode, pbar)
            return result
        
        # Process samples
        if max_concurrent:
            results = []
            for i in range(0, len(samples), max_concurrent):
                batch = samples[i:i+max_concurrent]
                batch_results = await asyncio.gather(*[evaluate_with_progress(s) for s in batch])
                results.extend(batch_results)
        else:
            results = await asyncio.gather(*[evaluate_with_progress(s) for s in samples])
        
        pbar.close()
        return results


async def main():
    parser = argparse.ArgumentParser(description="Score real dataset conversation branches using individual LLM scoring")
    parser.add_argument("--source-dataset", type=str, required=True,
                       help="Path to source dataset to score")
    parser.add_argument("--target-path", type=str, required=True,
                       help="Path where scored dataset will be saved")
    parser.add_argument("--samples", type=int, default=10, 
                       help="Number of samples to process (-1 for all)")
    parser.add_argument("--skip", type=int, default=0,
                       help="Skip the first N samples")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help="Model to use for evaluation")
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                       help="Whether to turn on thinking")
    parser.add_argument("--use-modal-response", action="store_true", default=False,
                       help="Whether to use the modal response rather than the logprobs")
    parser.add_argument("--scale-range", type=int, nargs=2, default=(1, 9),
                       help="Scale range for scoring (start end), e.g. 1 9")
    parser.add_argument("--concurrent", type=int, default=50,
                       help="Maximum concurrent API requests")
    parser.add_argument("--instructions", type=str, default="08-judge-evaluation/prompts/charter-generic.txt",
                       help="Path to judge instructions file (from repo root)")
    parser.add_argument("--charter-path", type=str, default="08-judge-evaluation/prompts/charter-generic.txt",
                       help="Path to judge charter file (from repo root, alias for --instructions)")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                       help="Maximum number of retries for failed samples")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging of all LLM requests and responses")
    
    args = parser.parse_args()
    
    # Initialize components
    loader = RealDatasetLoader(args.source_dataset)
    analyzer = EvaluationAnalyzer()
    reporter = ReportGenerator()
    utils = JudgeEvaluationUtils()
    
    # Prefer charter-path argument, fall back to instructions
    # Both default to the same value, but charter-path takes precedence if explicitly set
    instructions_path = args.charter_path
    
    print(f"Using judge instructions from: {instructions_path}")
    
    # Determine scoring method for metadata
    scoring_method = "modal_scoring" if args.use_modal_response else "weighted_mean_scoring"
    if args.enable_thinking:
        scoring_method += "_with_thinking"
    
    scorer = DatasetScorer(args.target_path, args.model, scoring_method)
    
    # Load dataset
    print(f"Loading dataset from: {args.source_dataset}")
    if args.skip > 0:
        total_needed = args.samples + args.skip if args.samples != -1 else -1
        all_samples = loader.load_dataset(total_needed)
        samples = all_samples[args.skip:args.skip + args.samples] if args.samples != -1 else all_samples[args.skip:]
        print(f"Loaded {len(all_samples)} samples total, skipped first {args.skip}, using {len(samples)} samples")
    else:
        samples = loader.load_dataset(args.samples)
        print(f"Loaded {len(samples)} samples")
    
    if not samples:
        print("No valid samples found. Exiting.")
        return
    
    # Show sample statistics
    completion_counts = [len(s['completions']) for s in samples]
    print(f"Completions per sample: min={min(completion_counts)}, max={max(completion_counts)}, avg={np.mean(completion_counts):.1f}")
    
    # Create debug file path if debug is enabled
    debug_file_path = None
    if args.debug:
        config = {
            'model': args.model,
            'modal': args.use_modal_response,
            'thinking': args.enable_thinking,
            'instructions': instructions_path
        }
        base_name = utils.generate_output_filename("llm_scoring_realdata", args.model, config, len(samples))
        debug_file_path = utils.create_debug_file_path(__file__, base_name)
        print(f"Debug logging enabled: {debug_file_path}")
    
    evaluator = JudgeScoringRealDataEvaluator(
        model=args.model,
        enable_thinking=args.enable_thinking,
        scale_range=args.scale_range,
        instructions_path=instructions_path,
        use_modal_response=args.use_modal_response,
        max_retries=args.max_retries,
        max_concurrent=args.concurrent,
        debug_file_path=debug_file_path
    )
    
    # Evaluate and score
    print("\nStarting scoring evaluation...")
    results = await evaluator.evaluate_all(samples, max_concurrent=args.concurrent)
    
    # Calculate metrics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n=== SCORING RESULTS ===")
    print(f"Success Rate: {len(successful)/len(results):.1%} ({len(successful)}/{len(results)})")
    print(f"Failed: {len(failed)}")
    
    if successful:
        all_scores = []
        for r in successful:
            all_scores.extend(r['scores'])
        
        print(f"Total completions scored: {len(all_scores)}")
        print(f"Mean score: {np.mean(all_scores):.2f}")
        print(f"Score std: {np.std(all_scores):.2f}")
        print(f"Score range: {np.min(all_scores):.2f} - {np.max(all_scores):.2f}")
    
    # Save scored dataset
    print(f"\nSaving scored dataset to: {args.target_path}")
    scorer.save_scored_dataset(samples, results)
    
    # Generate evaluation report
    config = {
        'model': args.model,
        'modal': args.use_modal_response,
        'thinking': args.enable_thinking,
        'instructions': instructions_path,
        'source_dataset': args.source_dataset,
        'target_path': args.target_path,
        'scoring_method': scoring_method
    }
    
    base_name = utils.generate_output_filename("llm_scoring_realdata", args.model, config, len(samples))
    output_dir = utils.create_output_directory(__file__)
    
    # Save results
    jsonl_path = output_dir / f"{base_name}.jsonl"
    with open(jsonl_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Generate report
    report_path = output_dir / f"{base_name}.md"
    # Calculate basic metrics for report
    total_tokens = sum(r.get('tokens', {}).get('total', 0) for r in results)
    
    metrics = {
        'success_rate': len(successful) / len(results) if results else 0,
        'successful': len(successful),
        'failed': len(failed),
        'total_samples': len(results),
        'failed_rate': len(failed) / len(results) if results else 0,
        'samples_retried': 0,  # No retry logic tracked separately
        'avg_retries': 0.0,
        'error_types': {},
        'total_tokens': total_tokens,
        'avg_tokens_per_sample': total_tokens / len(results) if results else 0,
        'avg_prompt_tokens': 0,  # Not tracked separately
        'avg_completion_tokens': total_tokens / len(results) if results else 0
    }
    
    if successful:
        all_scores = []
        for r in successful:
            all_scores.extend(r['scores'])
        
        metrics.update({
            'total_completions_scored': len(all_scores),
            'mean_score': np.mean(all_scores),
            'score_std': np.std(all_scores),
            'score_min': np.min(all_scores),
            'score_max': np.max(all_scores)
        })
    
    reporter.generate_report(results, metrics, report_path, config)
    
    print(f"\nReport saved to: {report_path}")
    print(f"Results saved to: {jsonl_path}")


if __name__ == "__main__":
    asyncio.run(main())