#!/usr/bin/env python3
"""
Judge LLM Odd-Even Sort for Real Data

Applies the proven odd-even sort ranking algorithm to real datasets with multiple
conversation branches. Computes rankings for all branches and saves them as a new
annotated dataset.

Usage:
    python judge_llm_oddeven_realdata.py --source-dataset /path/to/dataset --target-path /path/to/output --samples 10
    python judge_llm_oddeven_realdata.py --source-dataset /path/to/dataset --target-path /path/to/output --samples -1  # Full dataset
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
from scipy.stats import spearmanr, kendalltau
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
            
            if len(valid_completions) >= 2:  # Need at least 2 completions to rank
                samples.append({
                    'id': f"{i:04d}",
                    'conversation_id': sample['conversation_id'],
                    'question': sample['initial_prompt']['content'],
                    'completions': valid_completions,
                    'original_sample': sample  # Preserve original for output
                })
        
        return samples


class DatasetRanker:
    """Handles saving ranked datasets with preserved schema."""
    
    def __init__(self, target_path: str, judge_model: str):
        self.target_path = target_path
        self.judge_model = judge_model
        self.timestamp = datetime.now().isoformat()
    
    def save_ranked_dataset(self, samples: List[Dict], ranking_results: List[Dict]):
        """Save dataset with ranking annotations."""
        # Create mapping from sample_id to ranking results
        ranking_map = {r['sample_id']: r for r in ranking_results}
        
        # Process each sample and add rankings
        processed_samples = []
        
        for sample in samples:
            sample_id = sample['id']
            original_sample = copy.deepcopy(sample['original_sample'])
            
            if sample_id in ranking_map and ranking_map[sample_id]['success']:
                ranking_result = ranking_map[sample_id]
                predicted_rankings = ranking_result['predicted']
                
                # Add ranking metadata to each conversation branch
                for comp_idx, completion in enumerate(sample['completions']):
                    branch_idx = completion['branch_idx']
                    
                    # Find the corresponding ranking for this completion
                    if comp_idx < len(predicted_rankings):
                        ranking_score = predicted_rankings[comp_idx]
                        
                        # Add ranking metadata to the original sample's branch
                        if branch_idx < len(original_sample['conversation_branches']):
                            branch = original_sample['conversation_branches'][branch_idx]
                            
                            # Add ranking metadata to the appropriate part/message
                            if branch['messages']:
                                message = branch['messages'][0]
                                
                                if 'parts' in message:
                                    # New format - add to response part
                                    for part in message['parts']:
                                        if part['type'] == 'response':
                                            if 'metadata' not in part:
                                                part['metadata'] = {}
                                            part['metadata']['ranking_score'] = ranking_score
                                            part['metadata']['ranking_metadata'] = {
                                                'judge_model': self.judge_model,
                                                'ranking_method': 'oddeven_sort',
                                                'timestamp': self.timestamp,
                                                'total_branches_ranked': len(sample['completions'])
                                            }
                                            break
                                else:
                                    # Old format - add to message metadata
                                    if 'metadata' not in message:
                                        message['metadata'] = {}
                                    message['metadata']['ranking_score'] = ranking_score
                                    message['metadata']['ranking_metadata'] = {
                                        'judge_model': self.judge_model,
                                        'ranking_method': 'oddeven_sort',
                                        'timestamp': self.timestamp,
                                        'total_branches_ranked': len(sample['completions'])
                                    }
            
            processed_samples.append(original_sample)
        
        # Create new dataset
        new_dataset = Dataset.from_list(processed_samples)
        dataset_dict = DatasetDict({'train': new_dataset})
        
        # Save to disk
        dataset_dict.save_to_disk(self.target_path)
        print(f"Ranked dataset saved to: {self.target_path}")
        print(f"Total samples: {len(processed_samples)}")


class JudgeOddEvenSortRealDataEvaluator:
    """Evaluates and ranks real conversation branches using parallel odd-even sort."""
    
    def __init__(self, model: str = DEFAULT_MODEL, enable_reasoning: bool = False,
                 max_passes: Optional[int] = None, label_type: str = "alphabetic",
                 charter_path: str = "08-judge-evaluation/prompts/charter-generic.txt", 
                 max_retries: int = DEFAULT_MAX_RETRIES, max_concurrent: int = 50,
                 debug_file_path: Optional[Path] = None):
        self.model = model
        self.enable_reasoning = enable_reasoning
        self.max_passes = max_passes  # None = n_completions - 1
        self.label_type = label_type
        self.charter_path = charter_path
        
        # Initialize composable components
        self.llm_client = LLMClient(model=model, max_retries=max_retries, debug_file_path=debug_file_path)
        self.concurrent_evaluator = ConcurrentEvaluator(max_concurrent=max_concurrent)
        
        # Load judge charter
        self.judge_instructions = InstructionsLoader.load_instructions(
            charter_path, None
        )
        
        # Label mappings
        if label_type == "alphabetic":
            self.labels = ["A", "B"]
        else:
            self.labels = ["1", "2"]
    
    def create_pairwise_prompt(self, sample: Dict, idx_a: int, idx_b: int, 
                               first_label: str, second_label: str) -> str:
        """Create prompt for comparing two completions."""
        content_a = sample['completions'][idx_a]['content']
        content_b = sample['completions'][idx_b]['content']
        
        # Handle empty responses
        response_a = content_a if content_a else "[EMPTY RESPONSE]"
        response_b = content_b if content_b else "[EMPTY RESPONSE]"
        
        if self.enable_reasoning:
            prompt = f"""{self.judge_instructions}

Based on the above principles, compare these two responses:

Response {first_label}: {response_a}

Response {second_label}: {response_b}

Compare these two completions and determine which is better.

You must output your response in exactly this format:

REASONING:
1. First, identify the key differences between response {first_label} and response {second_label}
2. Then, evaluate which of these differences make one response better than the other
3. Focus your judgment on the differences rather than evaluating each response in isolation

DECISION: {first_label} or {second_label}

Do not deviate from this format. Output 'REASONING:' as a literal header, followed by your analysis, then 'DECISION:' as a literal header, followed by only the letter."""
        else:
            prompt = f"""{self.judge_instructions}

Based on the above principles, compare these two responses:

Response {first_label}: {response_a}

Response {second_label}: {response_b}

Which response is better, {first_label} or {second_label}?

Don't think or explain. Answer with only one character."""
        
        return prompt
    
    def parse_pairwise_logprobs(self, logprobs: list, label_a: str, label_b: str, debug_mode: bool = False) -> Tuple[float, float]:
        """Extract logprobs for both options from the response."""
        if not logprobs:
            return 0.0, 0.0
        
        in_thinking = False
        after_decision_marker = False
        
        for token_info in logprobs:
            token = token_info.token.strip()
            
            if token == "<think>":
                in_thinking = True
                continue
            elif token == "</think>":
                in_thinking = False
                continue
            
            if self.enable_reasoning and "DECISION" in token.upper():
                after_decision_marker = True
                continue
            
            if not in_thinking and (not self.enable_reasoning or after_decision_marker):
                if token == label_a or token == label_b:
                    choices = {label_a: None, label_b: None}
                    choices[token] = token_info.logprob
                    chosen_token = token
                    
                    if hasattr(token_info, 'top_logprobs') and token_info.top_logprobs:
                        for alt in token_info.top_logprobs:
                            alt_token = alt.token.strip()
                            if alt_token in choices:
                                choices[alt_token] = alt.logprob
                    
                    logprob_a = choices[label_a] if choices[label_a] is not None else -float('inf')
                    logprob_b = choices[label_b] if choices[label_b] is not None else -float('inf')
                    
                    if debug_mode:
                        print(f"Found decision token '{chosen_token}'")
                        for choice, logprob in choices.items():
                            if choice == chosen_token:
                                print(f"{choice}={logprob:.3f} (chosen token)")
                            elif logprob is not None:
                                print(f"{choice}={logprob:.3f} (found in top logprobs)")
                            else:
                                print(f"WARNING: {choice}=missing (not in top logprobs)")
                    
                    return logprob_a, logprob_b
        
        return -10.0, -10.1  # Slight preference for A when parsing fails
    
    async def compare_pair(self, sample: Dict, idx_a: int, idx_b: int, comparison_cache: Dict = None, debug_mode: bool = False, pbar=None) -> Dict:
        """Compare two completions with both orders and aggregate logprobs."""
        if comparison_cache is None:
            comparison_cache = {}
            
        cache_key = frozenset([idx_a, idx_b])
        if cache_key in comparison_cache:
            if debug_mode:
                print(f"Using cached comparison result for completions {idx_a} vs {idx_b}")
            return comparison_cache[cache_key]
        
        content_a = sample['completions'][idx_a]['content']
        content_b = sample['completions'][idx_b]['content']
        
        # Handle empty completions
        if not content_a and content_b:
            result = {
                'winner': idx_b,
                'score_a': -100.0,
                'score_b': 0.0,
                'consistent': True,
                'tokens': 0,
                'error': None
            }
            comparison_cache[cache_key] = result
            return result
        elif content_a and not content_b:
            result = {
                'winner': idx_a,
                'score_a': 0.0,
                'score_b': -100.0,
                'consistent': True,
                'tokens': 0,
                'error': None
            }
            comparison_cache[cache_key] = result
            return result
        elif not content_a and not content_b:
            result = {
                'winner': idx_a,
                'score_a': -100.0,
                'score_b': -100.0,
                'consistent': True,
                'tokens': 0,
                'error': None
            }
            comparison_cache[cache_key] = result
            return result
        
        # Both non-empty, proceed with LLM comparison
        prompt_ab = self.create_pairwise_prompt(sample, idx_a, idx_b, self.labels[0], self.labels[1])
        prompt_ba = self.create_pairwise_prompt(sample, idx_b, idx_a, self.labels[0], self.labels[1])
        
        result_ab, result_ba = await asyncio.gather(
            self.llm_client.make_request(
                [{"role": "user", "content": prompt_ab}],
                temperature=0.0,
                max_tokens=10000,
                openai_kwargs={"logprobs": True, "top_logprobs": 10}
            ),
            self.llm_client.make_request(
                [{"role": "user", "content": prompt_ba}],
                temperature=0.0,
                max_tokens=10000,
                openai_kwargs={"logprobs": True, "top_logprobs": 10}
            )
        )
        
        # Update progress bar for the 2 LLM requests made
        if pbar is not None:
            pbar.update(2)
        
        if not result_ab['success'] or not result_ba['success']:
            return {
                'winner': idx_a,
                'score_a': 0,
                'score_b': 0,
                'consistent': False,
                'tokens': result_ab.get('tokens', {'total': 0})['total'] + result_ba.get('tokens', {'total': 0})['total'],
                'error': result_ab.get('error') or result_ba.get('error', 'Unknown error')
            }
        
        logprob_a_first, logprob_b_first = self.parse_pairwise_logprobs(
            result_ab.get('logprobs', []), self.labels[0], self.labels[1], debug_mode
        )
        
        logprob_label_a_second, logprob_label_b_second = self.parse_pairwise_logprobs(
            result_ba.get('logprobs', []), self.labels[0], self.labels[1], debug_mode
        )
        logprob_b_swapped = logprob_label_a_second
        logprob_a_swapped = logprob_label_b_second
        
        score_a = (logprob_a_first + logprob_a_swapped) / 2
        score_b = (logprob_b_first + logprob_b_swapped) / 2
        
        consistent = (logprob_a_first > logprob_b_first) == (logprob_a_swapped > logprob_b_swapped)
        
        result = {
            'winner': idx_a if score_a > score_b else idx_b,
            'score_a': score_a,
            'score_b': score_b,
            'consistent': consistent,
            'tokens': result_ab['tokens']['total'] + result_ba['tokens']['total'],
            'error': None,
            'detailed_scores': {
                'ab_order': {'a': logprob_a_first, 'b': logprob_b_first},
                'ba_order': {'a': logprob_a_swapped, 'b': logprob_b_swapped},
                'averaged': {'a': score_a, 'b': score_b}
            },
            'llm_responses': {
                'ab_order': result_ab.get('content', ''),
                'ba_order': result_ba.get('content', '')
            }
        }
        
        comparison_cache[cache_key] = result
        return result
    
    async def rank_sample(self, sample: Dict, debug_mode: bool = False, pbar=None) -> Dict:
        """Rank all completions in a sample using odd-even sort."""
        n_completions = len(sample['completions'])
        
        # Add dummy completion if odd number
        has_dummy = n_completions % 2 == 1
        if has_dummy:
            sample_with_dummy = copy.deepcopy(sample)
            sample_with_dummy['completions'].append({
                'content': '',
                'branch_idx': -1,
                'original_metadata': {}
            })
            sample = sample_with_dummy
            n_completions += 1
            if debug_mode:
                print(f"Added dummy completion at index {n_completions-1}")
        
        max_passes = self.max_passes or (n_completions - 1)
        
        # Initialize ranking: start with original order [0, 1, 2, ..., n-1]
        current_order = list(range(n_completions))
        
        if debug_mode:
            print(f"\nSample {sample['id']}: Starting odd-even sort")
            print(f"Initial order: {current_order}")
        
        total_comparisons = 0
        total_tokens = 0
        consistent_comparisons = 0
        comparison_cache = {}
        round_num = 0
        
        # Odd-even sort algorithm
        while round_num < max_passes * 2:
            is_even_round = round_num % 2 == 0
            round_type = "Even" if is_even_round else "Odd"
            
            if debug_mode:
                print(f"\n--- Round {round_num + 1} ({round_type}) ---")
                print(f"Current order: {current_order}")
            
            # Generate comparison pairs
            comparison_pairs = []
            if is_even_round:
                for i in range(0, n_completions - 1, 2):
                    comparison_pairs.append((i, i + 1))
            else:
                for i in range(1, n_completions - 1, 2):
                    comparison_pairs.append((i, i + 1))
            
            if not comparison_pairs:
                round_num += 1
                continue
            
            if debug_mode:
                print(f"Comparisons this round: {comparison_pairs}")
            
            # Prepare comparison tasks
            comparison_tasks = []
            for pos_a, pos_b in comparison_pairs:
                idx_a = current_order[pos_a]
                idx_b = current_order[pos_b]
                comparison_tasks.append(self.compare_pair(sample, idx_a, idx_b, comparison_cache, debug_mode, pbar))
            
            # Run comparisons concurrently
            comparisons = await asyncio.gather(*comparison_tasks)
            
            # Process results and apply swaps
            round_swaps = 0
            for (pos_a, pos_b), comparison in zip(comparison_pairs, comparisons):
                idx_a = current_order[pos_a]
                idx_b = current_order[pos_b]
                
                if debug_mode:
                    print(f"\nPosition {pos_a} vs {pos_b}: completion_{idx_a} vs completion_{idx_b}")
                    if comparison['tokens'] > 0:
                        details = comparison['detailed_scores']
                        print(f"  Final scores: A={details['averaged']['a']:.3f}, B={details['averaged']['b']:.3f}")
                    else:
                        print(f"  Final scores: A={comparison['score_a']:.3f}, B={comparison['score_b']:.3f}")
                    print(f"  Winner: completion_{comparison['winner']}")
                    print(f"  Consistent: {comparison.get('consistent', False)}")
                
                if comparison['tokens'] > 0:
                    total_comparisons += 2
                
                total_tokens += comparison['tokens']
                
                if comparison.get('consistent', False):
                    consistent_comparisons += 1
                
                # Swap if B wins (B should be ranked higher)
                if comparison['winner'] == idx_b:
                    current_order[pos_a], current_order[pos_b] = current_order[pos_b], current_order[pos_a]
                    round_swaps += 1
                    if debug_mode:
                        print(f"  → Swapping! New order at positions {pos_a},{pos_b}: [{current_order[pos_a]}, {current_order[pos_b]}]")
                else:
                    if debug_mode:
                        print(f"  → No swap needed")
            
            if debug_mode:
                print(f"\nEnd of {round_type} round {round_num + 1}: {round_swaps} swaps made")
                print(f"Current order: {current_order}")
            
            # Early termination if no swaps
            if round_swaps == 0:
                if round_num > 0:
                    if debug_mode:
                        print(f"\nNo swaps in this round - checking for convergence")
                    break
            
            round_num += 1
        
        # Remove dummy completion from final order if it was added
        if has_dummy:
            dummy_idx = n_completions - 1  # The dummy completion index
            current_order = [idx for idx in current_order if idx != dummy_idx]
            final_n_completions = n_completions - 1
            if debug_mode:
                print(f"Removed dummy completion from final order")
        else:
            final_n_completions = n_completions
        
        # Convert final order to ranking scores (higher rank = better quality)
        final_ranking = [0] * final_n_completions
        for rank_position, completion_idx in enumerate(current_order):
            if completion_idx < final_n_completions:  # Only rank original completions
                final_ranking[completion_idx] = final_n_completions - rank_position
        
        if debug_mode:
            print(f"\nFINAL RESULTS for Sample {sample['id']}")
            print(f"Final order: {current_order}")
            print(f"Final ranking: {final_ranking}")
            print(f"Total comparisons: {total_comparisons}")
        
        consistency_rate = consistent_comparisons / (total_comparisons // 2) if total_comparisons > 0 else 0
        
        return {
            'sample_id': sample['id'],
            'success': True,
            'predicted': final_ranking,
            'total_comparisons': total_comparisons,
            'rounds_completed': round_num + 1,
            'consistency_rate': consistency_rate,
            'tokens': {
                'prompt': 0,
                'completion': total_tokens,
                'total': total_tokens
            },
            'error': None,
            'raw_response': f"Odd-even sort completed in {round_num + 1} rounds"
        }
    
    async def evaluate_all(self, samples: List[Dict], max_concurrent: Optional[int] = None) -> List[Dict]:
        """Evaluate all samples with progress tracking."""
        # Calculate total LLM requests
        total_llm_requests = 0
        for sample in samples:
            n_completions = len(sample['completions'])
            has_dummy = n_completions % 2 == 1
            n_completions += 1 if has_dummy else 0
            
            unique_pairs = (n_completions * (n_completions - 1)) // 2
            if has_dummy:
                llm_pairs = int(unique_pairs * 0.8 * 0.7)  # Estimate with dummy
            else:
                llm_pairs = int(unique_pairs * 0.6)  # Estimate without dummy
            
            total_llm_requests += llm_pairs * 2
        
        print(f"Total LLM requests to make: {total_llm_requests}")
        
        pbar = tqdm(total=total_llm_requests, desc="LLM Requests", unit="req")
        
        async def evaluate_with_progress(sample):
            debug_mode = len(samples) == 1
            result = await self.rank_sample(sample, debug_mode, pbar)
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
    parser = argparse.ArgumentParser(description="Rank real dataset conversation branches using odd-even sort")
    parser.add_argument("--source-dataset", type=str, required=True,
                       help="Path to source dataset to rank")
    parser.add_argument("--target-path", type=str, required=True,
                       help="Path where ranked dataset will be saved")
    parser.add_argument("--samples", type=int, default=10, 
                       help="Number of samples to process (-1 for all)")
    parser.add_argument("--skip", type=int, default=0,
                       help="Skip the first N samples")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help="Model to use for evaluation")
    parser.add_argument("--enable-reasoning", action="store_true",
                       help="Enable reasoning before pairwise decisions")
    parser.add_argument("--max-passes", type=int, default=None,
                       help="Maximum odd-even sort passes (default: n-1)")
    parser.add_argument("--label-type", type=str, default="alphabetic",
                       choices=["alphabetic", "numeric"],
                       help="Label type for completions")
    parser.add_argument("--concurrent", type=int, default=50,
                       help="Maximum concurrent API requests")
    parser.add_argument("--charter-path", type=str, default="08-judge-evaluation/prompts/charter-generic.txt",
                       help="Path to judge charter file (from repo root)")
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
    ranker = DatasetRanker(args.target_path, args.model)
    
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
    branch_counts = [len(s['completions']) for s in samples]
    print(f"Branches per sample: min={min(branch_counts)}, max={max(branch_counts)}, avg={np.mean(branch_counts):.1f}")
    
    # Create debug file path if debug is enabled
    debug_file_path = None
    if args.debug:
        config = {
            'model': args.model,
            'reasoning': args.enable_reasoning,
            'label_type': args.label_type,
            'charter': args.charter_path
        }
        base_name = utils.generate_output_filename("llm_oddeven_realdata", args.model, config, len(samples))
        debug_file_path = utils.create_debug_file_path(__file__, base_name)
        print(f"Debug logging enabled: {debug_file_path}")
    
    evaluator = JudgeOddEvenSortRealDataEvaluator(
        model=args.model,
        enable_reasoning=args.enable_reasoning,
        max_passes=args.max_passes,
        label_type=args.label_type,
        charter_path=args.charter_path,
        max_retries=args.max_retries,
        max_concurrent=args.concurrent,
        debug_file_path=debug_file_path
    )
    
    # Evaluate and rank
    print("\nStarting ranking evaluation...")
    results = await evaluator.evaluate_all(samples, max_concurrent=args.concurrent)
    
    # Calculate metrics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n=== RANKING RESULTS ===")
    print(f"Success Rate: {len(successful)/len(results):.1%} ({len(successful)}/{len(results)})")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"Mean Rounds to Sort: {np.mean([r['rounds_completed'] for r in successful]):.1f}")
        print(f"Mean Comparisons: {np.mean([r['total_comparisons'] for r in successful]):.1f}")
        print(f"Mean Consistency Rate: {np.mean([r['consistency_rate'] for r in successful]):.1%}")
    
    # Save ranked dataset
    print(f"\nSaving ranked dataset to: {args.target_path}")
    ranker.save_ranked_dataset(samples, results)
    
    # Generate evaluation report
    config = {
        'model': args.model,
        'reasoning': args.enable_reasoning,
        'label_type': args.label_type,
        'charter': args.charter_path,
        'source_dataset': args.source_dataset,
        'target_path': args.target_path
    }
    
    base_name = utils.generate_output_filename("llm_oddeven_realdata", args.model, config, len(samples))
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
    successful_tokens = [r.get('tokens', {}).get('total', 0) for r in successful]
    
    metrics = {
        'success_rate': len(successful) / len(results) if results else 0,
        'successful': len(successful),
        'failed': len(failed),
        'total_samples': len(results),
        'failed_rate': len(failed) / len(results) if results else 0,
        'samples_retried': 0,  # No retry logic in real data ranking
        'avg_retries': 0.0,
        'error_types': {},
        'total_tokens': total_tokens,
        'avg_tokens_per_sample': total_tokens / len(results) if results else 0,
        'avg_prompt_tokens': 0,  # Not tracked separately in this implementation
        'avg_completion_tokens': total_tokens / len(results) if results else 0
    }
    
    if successful:
        metrics.update({
            'mean_rounds': np.mean([r['rounds_completed'] for r in successful]),
            'mean_comparisons': np.mean([r['total_comparisons'] for r in successful]),
            'mean_consistency_rate': np.mean([r['consistency_rate'] for r in successful])
        })
    
    reporter.generate_report(results, metrics, report_path, config)
    
    print(f"\nReport saved to: {report_path}")
    print(f"Results saved to: {jsonl_path}")


if __name__ == "__main__":
    asyncio.run(main())