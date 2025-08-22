#!/usr/bin/env python3
"""
Judge LLM Odd-Even Sort Evaluation

Evaluates a judge model's ability to rank 9 completions using parallel odd-even sort with
pairwise comparisons. Each comparison is done twice (A vs B, B vs A) and uses
logprob aggregation for robust winner determination.

Usage:
    python judge_llm_oddeven.py --samples 3  # Dev mode
    python judge_llm_oddeven.py --samples -1  # Full evaluation
    python judge_llm_oddeven.py --samples 10 --enable-reasoning  # With reasoning
"""

import re
import random
import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, kendalltau
from tqdm.asyncio import tqdm

from lib import (
    SyntheticDatasetLoader, InstructionsLoader, EvaluationAnalyzer,
    ReportGenerator, LLMClient, ConcurrentEvaluator, JudgeEvaluationUtils
)

# Configuration
DATASET_PATH = "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-preference-quality-short-1100-synthetic"
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_MAX_RETRIES = 3


class JudgeOddEvenSortEvaluator:
    """Evaluates judge model's ranking ability using parallel odd-even sort with pairwise comparisons."""
    
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
        # charter_path is relative to repo root
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
            # For reasoning mode, add structured format after the basic prompt
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
            # Simple prompt without reasoning
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
            return 0.0, 0.0  # Return equal scores if no logprobs
        
        in_thinking = False
        after_decision_marker = False
        
        for token_info in logprobs:
            token = token_info.token.strip()
            
            # Handle thinking tokens
            if token == "<think>":
                in_thinking = True
                continue
            elif token == "</think>":
                in_thinking = False
                continue
            
            # For reasoning mode, look after DECISION: marker
            if self.enable_reasoning and "DECISION" in token.upper():
                after_decision_marker = True
                continue
            
            # Look for decision token
            if not in_thinking and (not self.enable_reasoning or after_decision_marker):
                if token == label_a or token == label_b:
                    # Found decision token - extract logprobs for both choices (following minimal_thinking_logprobs.py)
                    choices = {label_a: None, label_b: None}
                    
                    # Store logprob for the chosen token
                    choices[token] = token_info.logprob
                    chosen_token = token
                    
                    # Get logprobs for all choices from top_logprobs
                    if hasattr(token_info, 'top_logprobs') and token_info.top_logprobs:
                        for alt in token_info.top_logprobs:
                            alt_token = alt.token.strip()
                            if alt_token in choices:
                                choices[alt_token] = alt.logprob
                    
                    # Extract final logprobs
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
        
        # Didn't find decision token - return small preference for A to avoid all ties
        return -10.0, -10.1  # Slight preference for A when parsing fails
    
    async def compare_pair(self, sample: Dict, idx_a: int, idx_b: int, comparison_cache: Dict = None, debug_mode: bool = False) -> Dict:
        """Compare two completions with both orders and aggregate logprobs."""
        if comparison_cache is None:
            comparison_cache = {}
            
        # Check cache first
        cache_key = frozenset([idx_a, idx_b])
        if cache_key in comparison_cache:
            if debug_mode:
                print(f"Using cached comparison result for completions {idx_a} vs {idx_b}")
            return comparison_cache[cache_key]
        
        content_a = sample['completions'][idx_a]['content']
        content_b = sample['completions'][idx_b]['content']
        
        # Handle comparisons with empty dummy completion without LLM
        if not content_a and content_b:
            # A is empty, B wins
            result = {
                'winner': idx_b,
                'score_a': -100.0,  # Very low score for empty
                'score_b': 0.0,     # Neutral score for non-empty
                'consistent': True,  # Always consistent
                'tokens': 0,        # No tokens used
                'error': None
            }
            comparison_cache[cache_key] = result
            return result
        elif content_a and not content_b:
            # B is empty, A wins
            result = {
                'winner': idx_a,
                'score_a': 0.0,     # Neutral score for non-empty
                'score_b': -100.0,  # Very low score for empty
                'consistent': True,  # Always consistent
                'tokens': 0,        # No tokens used
                'error': None
            }
            comparison_cache[cache_key] = result
            return result
        elif not content_a and not content_b:
            # Both empty (shouldn't happen), default to A
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
        # Run both orders in parallel: A vs B and B vs A
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
        
        if not result_ab['success']:
            return {
                'winner': idx_a,  # Default to no swap on error
                'score_a': 0,
                'score_b': 0,
                'consistent': False,
                'tokens': result_ab.get('tokens', {'total': 0})['total'] + result_ba.get('tokens', {'total': 0})['total'],
                'error': result_ab.get('error', 'Unknown error')
            }
            
        if not result_ba['success']:
            return {
                'winner': idx_a,  # Default to no swap on error
                'score_a': 0,
                'score_b': 0,
                'consistent': False,
                'tokens': result_ab.get('tokens', {'total': 0})['total'] + result_ba.get('tokens', {'total': 0})['total'],
                'error': result_ba.get('error', 'Unknown error')
            }
        
        logprob_a_first, logprob_b_first = self.parse_pairwise_logprobs(
            result_ab.get('logprobs', []), self.labels[0], self.labels[1], debug_mode
        )
        
        # In the swapped prompt:
        # - label[0] (A) now refers to idx_b content
        # - label[1] (B) now refers to idx_a content
        # So we need to swap the interpretation:
        logprob_label_a_second, logprob_label_b_second = self.parse_pairwise_logprobs(
            result_ba.get('logprobs', []), self.labels[0], self.labels[1], debug_mode
        )
        # Map back to original indices:
        logprob_b_swapped = logprob_label_a_second  # Label A refers to idx_b in swapped
        logprob_a_swapped = logprob_label_b_second  # Label B refers to idx_a in swapped
        
        # Average logprobs (average the log probabilities for each completion across both positions)
        # score_a: average logprob for choosing idx_a content
        # score_b: average logprob for choosing idx_b content
        score_a = (logprob_a_first + logprob_a_swapped) / 2
        score_b = (logprob_b_first + logprob_b_swapped) / 2
        
        # Check consistency (both orders should prefer the same completion)
        # In first order: A wins if logprob_a_first > logprob_b_first
        # In second order: A wins if logprob_a_swapped > logprob_b_swapped  
        consistent = (logprob_a_first > logprob_b_first) == (logprob_a_swapped > logprob_b_swapped)
        
        result = {
            'winner': idx_a if score_a > score_b else idx_b,
            'score_a': score_a,
            'score_b': score_b,
            'consistent': consistent,
            'tokens': result_ab['tokens']['total'] + result_ba['tokens']['total'],
            'error': None,
            # Add detailed scoring breakdown
            'detailed_scores': {
                'ab_order': {'a': logprob_a_first, 'b': logprob_b_first},
                'ba_order': {'a': logprob_a_swapped, 'b': logprob_b_swapped},
                'averaged': {'a': score_a, 'b': score_b}
            },
            # Store LLM responses for debugging (when reasoning is enabled)
            'llm_responses': {
                'ab_order': result_ab.get('content', ''),
                'ba_order': result_ba.get('content', '')
            }
        }
        
        # Cache the result
        comparison_cache[cache_key] = result
        return result
    
    async def evaluate_all(self, samples: List[Dict], max_concurrent: Optional[int] = None) -> List[Dict]:
        """Evaluate all samples with progress tracking by LLM requests."""
        # Calculate total number of LLM requests for odd-even sort (accounting for dummy completion)
        total_llm_requests = 0
        for sample in samples:
            original_n_completions = len(sample['completions'])
            has_dummy = original_n_completions % 2 == 1
            n_completions = original_n_completions + (1 if has_dummy else 0)
            
            # For odd-even sort, estimate total unique comparisons
            # In worst case, we need about n rounds to sort n items
            # Each round has ~n/2 comparisons
            max_rounds = self.max_passes * 2 if self.max_passes else n_completions * 2
            unique_pairs = (n_completions * (n_completions - 1)) // 2  # All possible pairs
            
            if has_dummy:
                # Estimate: about 20% of comparisons involve dummy (no LLM requests)
                # About 30% of remaining comparisons are repeated (cached)
                llm_pairs = int(unique_pairs * 0.8 * 0.7)  # 56% of total pairs need LLM
            else:
                # About 40% of comparisons are repeated (cached)
                llm_pairs = int(unique_pairs * 0.6)  # 60% of total pairs need LLM
            
            total_llm_requests += llm_pairs * 2  # Each comparison = 2 LLM requests (A vs B + B vs A)
        
        print(f"Total LLM requests to make: {total_llm_requests}")
        
        # Create a shared progress bar
        pbar = tqdm(total=total_llm_requests, desc="LLM Requests", unit="req")
        
        async def evaluate_with_progress(sample):
            """Wrapper to update progress bar during evaluation."""
            # Enable detailed debug output only when evaluating a single sample
            debug_mode = len(samples) == 1
            original_n_completions = len(sample['completions'])
            
            # Add dummy completion if odd number to make even
            has_dummy = original_n_completions % 2 == 1
            if has_dummy:
                # Add dummy completion that always loses
                import copy
                sample_with_dummy = copy.deepcopy(sample)
                sample_with_dummy['completions'].append({'content': ''})  # Append to list
                sample_with_dummy['ground_truth'] = sample['ground_truth'] + [0]  # Worst possible rank
                sample = sample_with_dummy
                n_completions = original_n_completions + 1
                if debug_mode:
                    print(f"Added dummy completion at index {original_n_completions} (always loses)")
            else:
                n_completions = original_n_completions
            
            max_passes = self.max_passes or (n_completions - 1)
            
            # Create mapping from degradation_rank to completion_index
            degradation_to_index = {}
            index_to_degradation = {}
            for i in range(original_n_completions):  # Only map original completions
                degradation_rank = sample['completions'][i].get('degradation_rank')
                if degradation_rank is not None:
                    degradation_to_index[degradation_rank] = i
                    index_to_degradation[i] = degradation_rank
            
            # Handle dummy completion (if added)
            if has_dummy:
                dummy_degradation_rank = n_completions - 1  # Use next available rank (e.g., rank 9)
                degradation_to_index[dummy_degradation_rank] = original_n_completions
                index_to_degradation[original_n_completions] = dummy_degradation_rank
                if debug_mode:
                    print(f"Assigned dummy completion degradation rank: {dummy_degradation_rank}")
            
            # Initialize ranking: start with degradation ranks in their initial positions
            # Convert completion indices to their degradation ranks
            current_order = [index_to_degradation[i] for i in range(n_completions)]
            
            # Target order: sorted degradation ranks [0, 1, 2, 3, 4, 5, 6, 7, 8] (best to worst)
            target_order = list(range(n_completions))
            
            # Helper function to convert degradation rank to completion index
            def degradation_to_completion_index(degradation_rank):
                return degradation_to_index[degradation_rank]
            
            if debug_mode:
                print(f"\n{'='*60}")
                print(f"Sample {sample['id']}: Starting odd-even sort")
                print(f"Degradation ranks: {current_order}")
                print(f"Target:            {target_order}")
                print(f"{'='*60}")
            
            total_comparisons = 0
            total_tokens = 0
            consistent_comparisons = 0
            pass_swaps = []
            all_comparison_details = []
            pass_num = 0  # Initialize in case of early termination
            
            # Initialize comparison cache for this sample
            comparison_cache = {}
            round_num = 0
            
            # Odd-even sort: alternate between even and odd position comparisons
            while round_num < max_passes * 2:  # Max rounds = 2 * max_passes
                is_even_round = round_num % 2 == 0
                round_type = "Even" if is_even_round else "Odd"
                
                if debug_mode:
                    print(f"\n--- Round {round_num + 1} ({round_type}) ---")
                    print(f"Degradation ranks: {current_order}")
                
                # Generate comparison pairs for this round
                comparison_pairs = []
                if is_even_round:
                    # Even round: compare (0,1), (2,3), (4,5), (6,7), (8,9)
                    for i in range(0, n_completions - 1, 2):
                        comparison_pairs.append((i, i + 1))
                else:
                    # Odd round: compare (1,2), (3,4), (5,6), (7,8)
                    for i in range(1, n_completions - 1, 2):
                        comparison_pairs.append((i, i + 1))
                
                if not comparison_pairs:
                    if debug_mode:
                        print(f"No comparisons needed in {round_type} round")
                    round_num += 1
                    continue
                
                if debug_mode:
                    print(f"Comparisons this round: {comparison_pairs}")
                
                # Prepare comparison tasks
                comparison_tasks = []
                for pos_a, pos_b in comparison_pairs:
                    degradation_rank_a = current_order[pos_a]
                    degradation_rank_b = current_order[pos_b]
                    # Convert degradation ranks to completion indices for the comparison
                    idx_a = degradation_to_completion_index(degradation_rank_a)
                    idx_b = degradation_to_completion_index(degradation_rank_b)
                    comparison_tasks.append(self.compare_pair(sample, idx_a, idx_b, comparison_cache, debug_mode))
                
                # Run all comparisons in this round concurrently
                comparisons = await asyncio.gather(*comparison_tasks)
                
                # Process results and apply swaps
                round_swaps = 0
                for (pos_a, pos_b), comparison in zip(comparison_pairs, comparisons):
                    degradation_rank_a = current_order[pos_a]
                    degradation_rank_b = current_order[pos_b]
                    # Convert to completion indices for ground truth lookup
                    idx_a = degradation_to_completion_index(degradation_rank_a)
                    idx_b = degradation_to_completion_index(degradation_rank_b)
                    
                    # Determine which completion should be better based on ground truth
                    target_rank_a = sample['ground_truth'][idx_a]
                    target_rank_b = sample['ground_truth'][idx_b]
                    target_winner_idx = idx_a if target_rank_a > target_rank_b else idx_b
                    
                    if debug_mode:
                        print(f"\nPosition {pos_a} vs {pos_b}: degradation_rank_{degradation_rank_a} vs degradation_rank_{degradation_rank_b}")
                        print(f"  Should win: degradation_rank_{index_to_degradation[target_winner_idx]} {'(A)' if target_winner_idx == idx_a else '(B)'}")
                        
                        # Show detailed scoring breakdown if available
                        if 'detailed_scores' in comparison and comparison['tokens'] > 0:
                            details = comparison['detailed_scores']
                            print(f"  A vs B order: A={details['ab_order']['a']:.3f}, B={details['ab_order']['b']:.3f}")
                            print(f"  B vs A order: A={details['ba_order']['a']:.3f}, B={details['ba_order']['b']:.3f}")
                            print(f"  Final scores: A={details['averaged']['a']:.3f}, B={details['averaged']['b']:.3f}")
                        else:
                            print(f"  Final scores: A={comparison['score_a']:.3f}, B={comparison['score_b']:.3f}")
                        
                        print(f"  Winner: degradation_rank_{index_to_degradation[comparison['winner']]} {'(A)' if comparison['winner'] == idx_a else '(B)'}")
                        print(f"  Consistent: {comparison.get('consistent', False)}")
                        
                        # Show if this was determined without LLM or cached
                        if comparison['tokens'] == 0:
                            print(f"  (Determined without LLM - empty completion)")
                    
                    # Check if prediction matches target
                    correct_prediction = comparison['winner'] == target_winner_idx
                    if debug_mode:
                        print(f"  Correct prediction: {correct_prediction} {'✓' if correct_prediction else '✗'}")
                    
                    # Show LLM reasoning when enabled and prediction is wrong
                    if debug_mode and self.enable_reasoning and not correct_prediction and comparison['tokens'] > 0:
                        if 'llm_responses' in comparison:
                            print(f"\n  " + "="*80)
                            print(f"  REASONING (A vs B order):")
                            print(f"  " + "="*80)
                            print(f"  {comparison['llm_responses']['ab_order']}")
                            print(f"  " + "="*80)
                            print(f"  REASONING (B vs A order):")
                            print(f"  " + "="*80)
                            print(f"  {comparison['llm_responses']['ba_order']}")
                            print(f"  " + "="*80 + "\n")
                    
                    # Count actual LLM requests made (dummy/cached comparisons use 0 tokens)
                    if comparison['tokens'] > 0:
                        total_comparisons += 2  # We ask both orders
                        pbar.update(2)  # Update progress bar by 2 (one for each order)
                    
                    total_tokens += comparison['tokens']
                    
                    if comparison.get('consistent', False):
                        consistent_comparisons += 1
                    
                    all_comparison_details.append({
                        'round': round_num,
                        'round_type': round_type,
                        'position_a': pos_a,
                        'position_b': pos_b,
                        'idx_a': idx_a,
                        'idx_b': idx_b,
                        'winner': comparison['winner'],
                        'score_a': comparison['score_a'],
                        'score_b': comparison['score_b'],
                        'consistent': comparison.get('consistent', False)
                    })
                    
                    # Swap if B wins (B should be ranked higher/better)
                    if comparison['winner'] == idx_b:
                        current_order[pos_a], current_order[pos_b] = current_order[pos_b], current_order[pos_a]
                        round_swaps += 1
                        if debug_mode:
                            print(f"  → Swapping! New order at positions {pos_a},{pos_b}: [rank{current_order[pos_a]}, rank{current_order[pos_b]}]")
                    else:
                        if debug_mode:
                            print(f"  → No swap needed")
                
                pass_swaps.append(round_swaps)
                if debug_mode:
                    print(f"\nEnd of {round_type} round {round_num + 1}: {round_swaps} swaps made")
                    print(f"Degradation ranks: {current_order}")
                    
                    # Show how many positions are correct
                    correct_positions = sum(1 for i in range(n_completions) if current_order[i] == target_order[i])
                    print(f"Positions correct: {correct_positions}/{n_completions}")
                
                # Early termination if no swaps were made in this round
                if round_swaps == 0:
                    # Check if we should stop (no swaps in a complete even/odd cycle)
                    if round_num > 0 and pass_swaps[-2:] == [0, 0]:  # Last two rounds had no swaps
                        if debug_mode:
                            print(f"\nNo swaps in last two rounds - sorting complete!")
                        break
                
                round_num += 1
                pass_num = round_num  # Update for final reporting
            
            # Remove dummy completion from final order if it was added
            if has_dummy:
                # Remove the dummy completion (degradation rank n_completions-1) from current_order
                dummy_rank = n_completions - 1
                current_order = [rank for rank in current_order if rank != dummy_rank]
                if debug_mode:
                    print(f"Removed dummy completion from final order")
                final_n_completions = original_n_completions
                final_ground_truth = sample['ground_truth'][:original_n_completions]
            else:
                final_n_completions = n_completions
                final_ground_truth = sample['ground_truth']
            
            # Convert degradation ranks back to ranking (9=best, 1=worst for original 9 completions)
            final_ranking = [0] * final_n_completions
            for rank_position, degradation_rank in enumerate(current_order):
                if degradation_rank < final_n_completions:  # Only rank original completions
                    completion_idx = degradation_to_completion_index(degradation_rank)
                    final_ranking[completion_idx] = final_n_completions - rank_position
            
            # Calculate target order for final comparison (only original completions)
            final_target_order = sorted(range(final_n_completions), key=lambda i: final_ground_truth[i], reverse=True)
            
            if debug_mode:
                print(f"\n{'='*60}")
                print(f"FINAL RESULTS for Sample {sample['id']}")
                print(f"Final degradation ranks: {current_order}")
                print(f"Target degradation ranks: {list(range(final_n_completions))}")
            
            # Calculate metrics
            spearman_corr, _ = spearmanr(final_ground_truth, final_ranking)
            kendall_corr, _ = kendalltau(final_ground_truth, final_ranking)
            
            # Final position accuracy
            final_correct_positions = sum(1 for i in range(final_n_completions) if current_order[i] == final_target_order[i])
            
            if debug_mode:
                print(f"Spearman correlation: {spearman_corr:.3f}")
                print(f"Kendall's tau: {kendall_corr:.3f}")
                print(f"Perfect positions: {final_correct_positions}/{final_n_completions}")
                print(f"Total comparisons: {total_comparisons} ({total_comparisons//2} pairs × 2 orders)")
                print(f"{'='*60}\n")
            
            # Top-k accuracy: count how many of the top 3 actual best completions got high ranks
            if final_n_completions >= 3:
                best_3_indices = sorted(range(final_n_completions), key=lambda i: final_ground_truth[i], reverse=True)[:3]
                top3_threshold = max(1, final_n_completions - 2)  # Top 3 positions
                top3_correct = sum(1 for i in best_3_indices if final_ranking[i] >= top3_threshold)
            else:
                top3_correct = 0
            
            # Calculate consistency rate (pairs that agreed in both orders)
            consistency_rate = consistent_comparisons / (total_comparisons // 2) if total_comparisons > 0 else 0
            
            return {
                'sample_id': sample['id'],
                'success': True,
                'ground_truth': final_ground_truth,
                'predicted': final_ranking,
                'spearman': spearman_corr,
                'kendall': kendall_corr,
                'top3_correct': top3_correct,
                'total_comparisons': total_comparisons,
                'rounds_completed': round_num + 1,
                'swaps_per_pass': pass_swaps,
                'consistency_rate': consistency_rate,
                'comparison_details': all_comparison_details,
                'tokens': {
                    'prompt': 0,  # Will be calculated from total
                    'completion': total_tokens,
                    'total': total_tokens
                },
                'error': None,
                'raw_response': f"Odd-even sort completed in {round_num + 1} rounds"
            }
        
        # Process samples concurrently
        if max_concurrent:
            # Process with limited concurrency
            results = []
            for i in range(0, len(samples), max_concurrent):
                batch = samples[i:i+max_concurrent]
                batch_results = await asyncio.gather(*[evaluate_with_progress(s) for s in batch])
                results.extend(batch_results)
        else:
            # Process all at once
            results = await asyncio.gather(*[evaluate_with_progress(s) for s in samples])
        
        pbar.close()
        return results


async def main():
    parser = argparse.ArgumentParser(description="Evaluate judge model using parallel odd-even sort ranking")
    parser.add_argument("--samples", type=int, default=3, 
                       help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--skip", type=int, default=0,
                       help="Skip the first N samples (useful for testing specific samples)")
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
    loader = SyntheticDatasetLoader(DATASET_PATH)
    analyzer = EvaluationAnalyzer()
    reporter = ReportGenerator()
    utils = JudgeEvaluationUtils()
    
    # Load dataset
    print(f"Loading dataset...")
    if args.skip > 0:
        # Load more samples to account for skipping
        total_needed = args.samples + args.skip if args.samples != -1 else -1
        all_samples = loader.load_dataset(total_needed)
        samples = all_samples[args.skip:args.skip + args.samples] if args.samples != -1 else all_samples[args.skip:]
        print(f"Loaded {len(all_samples)} samples total, skipped first {args.skip}, using {len(samples)} samples")
    else:
        samples = loader.load_dataset(args.samples)
        print(f"Loaded {len(samples)} samples")
    
    # Create debug file path if debug is enabled
    debug_file_path = None
    if args.debug:
        config = {
            'model': args.model,
            'reasoning': args.enable_reasoning,
            'label_type': args.label_type,
            'charter': args.charter_path
        }
        base_name = utils.generate_output_filename("llm_oddeven", args.model, config, len(samples))
        debug_file_path = utils.create_debug_file_path(__file__, base_name)
        print(f"Debug logging enabled: {debug_file_path}")
    
    evaluator = JudgeOddEvenSortEvaluator(
        model=args.model,
        enable_reasoning=args.enable_reasoning,
        max_passes=args.max_passes,
        label_type=args.label_type,
        charter_path=args.charter_path,
        max_retries=args.max_retries,
        max_concurrent=args.concurrent,
        debug_file_path=debug_file_path
    )
    
    # Evaluate
    results = await evaluator.evaluate_all(samples, max_concurrent=args.concurrent)
    
    # Calculate metrics with odd-even sort specific additions
    metrics = analyzer.calculate_overall_metrics(results)
    
    # Add odd-even sort specific metrics
    successful = [r for r in results if r['success']]
    if successful:
        metrics['mean_rounds'] = np.mean([r['rounds_completed'] for r in successful])
        metrics['mean_comparisons'] = np.mean([r['total_comparisons'] for r in successful])
        metrics['mean_consistency_rate'] = np.mean([r['consistency_rate'] for r in successful])
        
        # Analyze swap patterns
        all_swaps = []
        for r in successful:
            all_swaps.extend(r['swaps_per_pass'])
        if all_swaps:
            metrics['mean_swaps_per_round'] = np.mean(all_swaps)
            metrics['total_swaps'] = sum(all_swaps)
    
    # Print final results to terminal
    print("\n=== FINAL RESULTS ===")
    print(f"Success Rate: {metrics['success_rate']:.1%} ({metrics['successful']}/{metrics['total_samples']})")
    print(f"Failed: {metrics['failed_rate']:.1%} ({metrics['failed']}/{metrics['total_samples']})")
    if metrics['successful'] > 0:
        print(f"Mean Spearman: {metrics['mean_spearman']:.3f}")
        print(f"Mean Kendall's Tau: {metrics['mean_kendall']:.3f}")
        print(f"Mean Rounds to Sort: {metrics.get('mean_rounds', 0):.1f}")
        print(f"Mean Comparisons: {metrics.get('mean_comparisons', 0):.1f}")
        print(f"Consistency Rate: {metrics.get('mean_consistency_rate', 0):.1%}")
    
    # Generate output filenames
    config = {
        'model': args.model,
        'reasoning': args.enable_reasoning,
        'label_type': args.label_type,
        'charter': args.charter_pathZ
    }
    base_name = utils.generate_output_filename("llm_oddeven_qwenData_", args.model, config, len(samples))
    output_dir = utils.create_output_directory(__file__)
    
    # Save results
    jsonl_path = output_dir / f"{base_name}.jsonl"
    with open(jsonl_path, 'w') as f:
        for result in results:
            # Remove detailed comparison data for storage efficiency
            result_copy = result.copy()
            if 'comparison_details' in result_copy:
                del result_copy['comparison_details']
            f.write(json.dumps(result_copy) + '\n')
    
    # Generate report
    report_path = output_dir / f"{base_name}.md"
    reporter.generate_report(results, metrics, report_path, config)
    
    print(f"\nReport saved to: {report_path}")
    print(f"Results saved to: {jsonl_path}")


if __name__ == "__main__":
    asyncio.run(main())