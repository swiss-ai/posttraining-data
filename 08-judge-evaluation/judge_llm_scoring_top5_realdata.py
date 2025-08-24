#!/usr/bin/env python3
"""
Judge LLM Top-5 Scoring for Real Data

Optimized ranking algorithm that uses pre-existing scores to select the top 5 branches
and then applies LLM-based odd-even sort only to those top 5, dramatically reducing
the number of comparisons needed (from ~210 to ~10 for 21 branches).

The final ranking combines both methods:
- Top 5: Ranked by LLM odd-even sort (positions 1-5)  
- Remaining: Ranked by pre-existing scores (positions 6-21)

Usage:
    python judge_llm_scoring_top5_realdata.py --source-dataset /path/to/dataset --target-path /path/to/output --samples 10
    python judge_llm_scoring_top5_realdata.py --source-dataset /path/to/dataset --target-path /path/to/output --samples -1  # Full dataset
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
DEFAULT_TOP_K = 5


class RealDatasetLoader:
    """Handles loading and preparation of real datasets with pre-existing scores."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
    
    def extract_branch_score(self, branch: Dict) -> Optional[float]:
        """Extract score from a conversation branch."""
        if not branch.get('messages'):
            return None
            
        message = branch['messages'][0]
        
        # Handle both old and new chat formats
        if 'parts' in message:
            # New format with parts
            for part in message['parts']:
                if part['type'] == 'response':
                    return part.get('metadata', {}).get('score')
        else:
            # Old format with direct content
            return message.get('metadata', {}).get('score')
        
        return None
    
    def load_dataset(self, num_samples: int) -> List[Dict]:
        """Load and prepare dataset samples with score extraction."""
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
            
            # Extract completions with scores from all conversation branches
            completions = []
            
            for branch_idx, branch in enumerate(sample['conversation_branches']):
                if branch['messages']:
                    message = branch['messages'][0]  # Assume single message per branch
                    score = self.extract_branch_score(branch)
                    
                    # Handle both old and new chat formats
                    if 'parts' in message:
                        # New format with parts
                        for part in message['parts']:
                            if part['type'] == 'response':
                                completion_data = {
                                    'content': part['content'],
                                    'branch_idx': branch_idx,
                                    'score': score,
                                    'original_metadata': part.get('metadata', {})
                                }
                                completions.append(completion_data)
                                break
                    else:
                        # Old format with direct content
                        completion_data = {
                            'content': message.get('content', ''),
                            'branch_idx': branch_idx,
                            'score': score,
                            'original_metadata': message.get('metadata', {})
                        }
                        completions.append(completion_data)
            
            # Filter out empty completions and ensure we have valid content and scores
            valid_completions = []
            for comp in completions:
                if (comp['content'] and comp['content'].strip() and 
                    comp['score'] is not None):
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
    
    def __init__(self, target_path: str, judge_model: str, top_k: int):
        self.target_path = target_path
        self.judge_model = judge_model
        self.top_k = top_k
        self.timestamp = datetime.now().isoformat()
    
    def extract_ranking_score(self, branch: Dict) -> Optional[int]:
        """Extract ranking score from a conversation branch."""
        if not branch.get('messages'):
            return None
            
        message = branch['messages'][0]
        
        # Handle both old and new chat formats
        if 'parts' in message:
            # New format with parts
            for part in message['parts']:
                if part['type'] == 'response':
                    return part.get('metadata', {}).get('ranking_score')
        else:
            # Old format with direct content
            return message.get('metadata', {}).get('ranking_score')
        
        return None
    
    def sort_branches_by_ranking(self, sample: Dict) -> Dict:
        """Sort conversation branches by ranking_score (descending = best first)."""
        branches = sample['conversation_branches']
        
        # Create list of (ranking_score, original_index, branch) tuples
        branches_with_scores = []
        for i, branch in enumerate(branches):
            ranking_score = self.extract_ranking_score(branch)
            if ranking_score is not None:
                branches_with_scores.append((ranking_score, i, branch))
            else:
                # Branches without ranking_score go to the end with score 0
                branches_with_scores.append((0, i, branch))
        
        # Sort by ranking_score (descending) - best first
        branches_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Extract sorted branches
        sorted_branches = [branch for _, _, branch in branches_with_scores]
        
        # Create new sample with sorted branches
        sorted_sample = copy.deepcopy(sample)
        sorted_sample['conversation_branches'] = sorted_branches
        
        return sorted_sample
    
    def save_ranked_dataset(self, samples: List[Dict], ranking_results: List[Dict]):
        """Save dataset with hybrid ranking annotations and sorted branches."""
        # Create mapping from sample_id to ranking results
        ranking_map = {r['sample_id']: r for r in ranking_results}
        
        # Process each sample and add rankings
        processed_samples = []
        
        for sample in samples:
            sample_id = sample['id']
            original_sample = copy.deepcopy(sample['original_sample'])
            
            if sample_id in ranking_map and ranking_map[sample_id]['success']:
                ranking_result = ranking_map[sample_id]
                final_rankings = ranking_result['final_rankings']
                
                # Add ranking metadata to each conversation branch
                for completion in sample['completions']:
                    branch_idx = completion['branch_idx']
                    
                    # Find the ranking for this branch
                    if branch_idx < len(final_rankings):
                        ranking_score = final_rankings[branch_idx]
                        
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
                                                'ranking_method': f'hybrid_top{self.top_k}_scoring',
                                                'timestamp': self.timestamp,
                                                'total_branches': len(sample['completions']),
                                                'llm_ranked_top_k': self.top_k,
                                                'score_ranked_remaining': len(sample['completions']) - self.top_k
                                            }
                                            break
                                else:
                                    # Old format - add to message metadata
                                    if 'metadata' not in message:
                                        message['metadata'] = {}
                                    message['metadata']['ranking_score'] = ranking_score
                                    message['metadata']['ranking_metadata'] = {
                                        'judge_model': self.judge_model,
                                        'ranking_method': f'hybrid_top{self.top_k}_scoring',
                                        'timestamp': self.timestamp,
                                        'total_branches': len(sample['completions']),
                                        'llm_ranked_top_k': self.top_k,
                                        'score_ranked_remaining': len(sample['completions']) - self.top_k
                                    }
            
            processed_samples.append(original_sample)
        
        # Sort conversation branches by ranking_score (best first) for each sample
        print("Sorting conversation branches by ranking score (best first)...")
        sorted_samples = []
        for sample in processed_samples:
            sorted_sample = self.sort_branches_by_ranking(sample)
            sorted_samples.append(sorted_sample)
        
        # Create new dataset
        new_dataset = Dataset.from_list(sorted_samples)
        dataset_dict = DatasetDict({'train': new_dataset})
        
        # Save to disk
        dataset_dict.save_to_disk(self.target_path)
        print(f"Ranked dataset saved to: {self.target_path}")
        print(f"Total samples: {len(sorted_samples)}")
        print("Conversation branches are now sorted by ranking_score (best first)")


class JudgeTop5ScoringEvaluator:
    """Evaluates and ranks using hybrid top-5 LLM + score-based approach."""
    
    def __init__(self, model: str = DEFAULT_MODEL, enable_reasoning: bool = False,
                 max_passes: Optional[int] = None, label_type: str = "alphabetic",
                 charter_path: str = "08-judge-evaluation/prompts/charter-generic.txt", 
                 max_retries: int = DEFAULT_MAX_RETRIES, max_concurrent: int = 50,
                 top_k: int = DEFAULT_TOP_K, debug_file_path: Optional[Path] = None):
        self.model = model
        self.enable_reasoning = enable_reasoning
        self.max_passes = max_passes  # None = k-1 for top-k
        self.label_type = label_type
        self.charter_path = charter_path
        self.top_k = top_k
        
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
    
    def select_top_k_by_score(self, completions: List[Dict]) -> Tuple[List[int], List[int]]:
        """Select top-k completions by score and return indices."""
        # Create list of (score, original_index) pairs
        scored_completions = [(comp['score'], idx) for idx, comp in enumerate(completions)]
        
        # Sort by score (descending - higher scores are better)
        scored_completions.sort(key=lambda x: x[0], reverse=True)
        
        # Split into top-k and remaining
        top_k_indices = [idx for _, idx in scored_completions[:self.top_k]]
        remaining_indices = [idx for _, idx in scored_completions[self.top_k:]]
        
        return top_k_indices, remaining_indices
    
    def create_pairwise_prompt(self, completions: List[Dict], idx_a: int, idx_b: int, 
                               first_label: str, second_label: str) -> str:
        """Create prompt for comparing two completions."""
        content_a = completions[idx_a]['content']
        content_b = completions[idx_b]['content']
        
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
    
    async def compare_pair(self, completions: List[Dict], idx_a: int, idx_b: int, 
                          comparison_cache: Dict = None, debug_mode: bool = False, pbar=None) -> Dict:
        """Compare two completions with both orders and aggregate logprobs."""
        if comparison_cache is None:
            comparison_cache = {}
            
        cache_key = frozenset([idx_a, idx_b])
        if cache_key in comparison_cache:
            if debug_mode:
                print(f"Using cached comparison result for completions {idx_a} vs {idx_b}")
            return comparison_cache[cache_key]
        
        content_a = completions[idx_a]['content']
        content_b = completions[idx_b]['content']
        
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
        prompt_ab = self.create_pairwise_prompt(completions, idx_a, idx_b, self.labels[0], self.labels[1])
        prompt_ba = self.create_pairwise_prompt(completions, idx_b, idx_a, self.labels[0], self.labels[1])
        
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
    
    async def rank_top_k_with_oddeven(self, completions: List[Dict], top_k_indices: List[int], 
                                     debug_mode: bool = False, pbar=None) -> List[int]:
        """Rank top-k completions using odd-even sort."""
        if len(top_k_indices) <= 1:
            return top_k_indices
        
        n_completions = len(top_k_indices)
        max_passes = self.max_passes or (n_completions - 1)
        
        # Initialize ranking: start with original order
        current_order = list(range(n_completions))
        
        if debug_mode:
            print(f"\nRanking top-{self.top_k} completions using odd-even sort")
            print(f"Top-k indices: {top_k_indices}")
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
                idx_a = top_k_indices[current_order[pos_a]]
                idx_b = top_k_indices[current_order[pos_b]]
                comparison_tasks.append(self.compare_pair(completions, idx_a, idx_b, comparison_cache, debug_mode, pbar))
            
            # Run comparisons concurrently
            comparisons = await asyncio.gather(*comparison_tasks)
            
            # Process results and apply swaps
            round_swaps = 0
            for (pos_a, pos_b), comparison in zip(comparison_pairs, comparisons):
                idx_a = top_k_indices[current_order[pos_a]]
                idx_b = top_k_indices[current_order[pos_b]]
                
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
        
        # Convert final order back to original indices
        ranked_top_k = [top_k_indices[pos] for pos in current_order]
        
        if debug_mode:
            print(f"\nFINAL TOP-K RANKING")
            print(f"Ranked order: {ranked_top_k}")
            print(f"Total comparisons: {total_comparisons}")
        
        return ranked_top_k, total_tokens, total_comparisons, consistent_comparisons
    
    async def rank_sample(self, sample: Dict, debug_mode: bool = False, pbar=None) -> Dict:
        """Rank all completions using hybrid top-k + score approach."""
        completions = sample['completions']
        n_completions = len(completions)
        
        if debug_mode:
            print(f"\nSample {sample['id']}: Hybrid ranking with top-{self.top_k}")
            print(f"Total completions: {n_completions}")
            for i, comp in enumerate(completions):
                print(f"  Completion {i}: score={comp['score']:.3f}")
        
        # Step 1: Select top-k by score
        top_k_indices, remaining_indices = self.select_top_k_by_score(completions)
        
        if debug_mode:
            print(f"\nTop-{self.top_k} by score: {top_k_indices}")
            print(f"Remaining by score: {remaining_indices}")
            print("Top-k scores:", [completions[i]['score'] for i in top_k_indices])
            print("Remaining scores:", [completions[i]['score'] for i in remaining_indices])
        
        # Step 2: Rank top-k using LLM odd-even sort
        total_tokens = 0
        total_comparisons = 0
        consistent_comparisons = 0
        
        if len(top_k_indices) > 1:
            ranked_top_k, tokens, comparisons, consistent = await self.rank_top_k_with_oddeven(
                completions, top_k_indices, debug_mode, pbar
            )
            total_tokens += tokens
            total_comparisons += comparisons
            consistent_comparisons += consistent
        else:
            ranked_top_k = top_k_indices
        
        # Step 3: Combine rankings
        # Top-k get ranks n, n-1, ..., n-k+1 (highest ranks)
        # Remaining get ranks n-k, n-k-1, ..., 1 (by score order)
        final_rankings = [0] * n_completions
        
        # Assign ranks to top-k (LLM ranked)
        for rank_position, completion_idx in enumerate(ranked_top_k):
            final_rankings[completion_idx] = n_completions - rank_position
        
        # Assign ranks to remaining (score ranked)
        remaining_rank = n_completions - len(ranked_top_k)
        for completion_idx in remaining_indices:
            final_rankings[completion_idx] = remaining_rank
            remaining_rank -= 1
        
        if debug_mode:
            print(f"\nFINAL HYBRID RANKINGS for Sample {sample['id']}")
            print(f"Final rankings: {final_rankings}")
            for i, rank in enumerate(final_rankings):
                method = "LLM" if i in ranked_top_k else "Score"
                print(f"  Completion {i}: rank={rank} (by {method}), score={completions[i]['score']:.3f}")
        
        consistency_rate = consistent_comparisons / (total_comparisons // 2) if total_comparisons > 0 else 1.0
        
        return {
            'sample_id': sample['id'],
            'success': True,
            'final_rankings': final_rankings,
            'top_k_indices': top_k_indices,
            'llm_ranked_order': ranked_top_k,
            'remaining_indices': remaining_indices,
            'total_comparisons': total_comparisons,
            'consistency_rate': consistency_rate,
            'tokens': {
                'prompt': 0,
                'completion': total_tokens,
                'total': total_tokens
            },
            'error': None,
            'raw_response': f"Hybrid top-{self.top_k} ranking completed"
        }
    
    async def evaluate_all(self, samples: List[Dict], max_concurrent: Optional[int] = None) -> List[Dict]:
        """Evaluate all samples with progress tracking."""
        # Calculate total LLM requests (much more efficient than full odd-even)
        total_llm_requests = 0
        for sample in samples:
            n_top_k = min(len(sample['completions']), self.top_k)
            if n_top_k > 1:
                # Number of unique pairs for top-k * 2 (for both orders)
                unique_pairs = (n_top_k * (n_top_k - 1)) // 2
                total_llm_requests += unique_pairs * 2
        
        print(f"Total LLM requests to make: {total_llm_requests}")
        print(f"Efficiency gain: ~{((21*20//2)*2 - total_llm_requests) / ((21*20//2)*2) * 100:.1f}% fewer requests vs full odd-even sort")
        
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
    parser = argparse.ArgumentParser(description="Rank real dataset conversation branches using hybrid top-k scoring")
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
                       help="Maximum odd-even sort passes (default: k-1)")
    parser.add_argument("--label-type", type=str, default="alphabetic",
                       choices=["alphabetic", "numeric"],
                       help="Label type for completions")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                       help="Number of top completions to rank with LLM (default: 5)")
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
    ranker = DatasetRanker(args.target_path, args.model, args.top_k)
    
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
    
    # Show score statistics
    all_scores = []
    for sample in samples:
        all_scores.extend([comp['score'] for comp in sample['completions']])
    print(f"Score range: {min(all_scores):.2f} - {max(all_scores):.2f}, mean: {np.mean(all_scores):.2f}")
    
    # Create debug file path if debug is enabled
    debug_file_path = None
    if args.debug:
        config = {
            'model': args.model,
            'reasoning': args.enable_reasoning,
            'label_type': args.label_type,
            'charter': args.charter_path,
            'top_k': args.top_k
        }
        base_name = utils.generate_output_filename("llm_scoring_top5_realdata", args.model, config, len(samples))
        debug_file_path = utils.create_debug_file_path(__file__, base_name)
        print(f"Debug logging enabled: {debug_file_path}")
    
    evaluator = JudgeTop5ScoringEvaluator(
        model=args.model,
        enable_reasoning=args.enable_reasoning,
        max_passes=args.max_passes,
        label_type=args.label_type,
        charter_path=args.charter_path,
        max_retries=args.max_retries,
        max_concurrent=args.concurrent,
        top_k=args.top_k,
        debug_file_path=debug_file_path
    )
    
    # Evaluate and rank
    print(f"\nStarting hybrid top-{args.top_k} ranking evaluation...")
    results = await evaluator.evaluate_all(samples, max_concurrent=args.concurrent)
    
    # Calculate metrics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n=== HYBRID RANKING RESULTS ===")
    print(f"Success Rate: {len(successful)/len(results):.1%} ({len(successful)}/{len(results)})")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"Mean Comparisons per Sample: {np.mean([r['total_comparisons'] for r in successful]):.1f}")
        print(f"Mean Consistency Rate: {np.mean([r['consistency_rate'] for r in successful]):.1%}")
        print(f"Top-{args.top_k} LLM Ranked, Remaining Score Ranked")
    
    # Save ranked dataset
    print(f"\nSaving ranked dataset to: {args.target_path}")
    ranker.save_ranked_dataset(samples, results)
    
    # Generate evaluation report
    config = {
        'model': args.model,
        'reasoning': args.enable_reasoning,
        'label_type': args.label_type,
        'charter': args.charter_path,
        'top_k': args.top_k,
        'source_dataset': args.source_dataset,
        'target_path': args.target_path
    }
    
    base_name = utils.generate_output_filename("llm_scoring_top5_realdata", args.model, config, len(samples))
    output_dir = utils.create_output_directory(__file__)
    
    # Save results
    jsonl_path = output_dir / f"{base_name}.jsonl"
    with open(jsonl_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Generate report
    report_path = output_dir / f"{base_name}.md"
    total_tokens = sum(r.get('tokens', {}).get('total', 0) for r in results)
    
    metrics = {
        'success_rate': len(successful) / len(results) if results else 0,
        'successful': len(successful),
        'failed': len(failed),
        'total_samples': len(results),
        'failed_rate': len(failed) / len(results) if results else 0,
        'samples_retried': 0,
        'avg_retries': 0.0,
        'error_types': {},
        'total_tokens': total_tokens,
        'avg_tokens_per_sample': total_tokens / len(results) if results else 0,
        'avg_prompt_tokens': 0,
        'avg_completion_tokens': total_tokens / len(results) if results else 0
    }
    
    if successful:
        metrics.update({
            'mean_comparisons': np.mean([r['total_comparisons'] for r in successful]),
            'mean_consistency_rate': np.mean([r['consistency_rate'] for r in successful]),
            'top_k': args.top_k
        })
    
    reporter.generate_report(results, metrics, report_path, config)
    
    print(f"\nReport saved to: {report_path}")
    print(f"Results saved to: {jsonl_path}")


if __name__ == "__main__":
    asyncio.run(main())