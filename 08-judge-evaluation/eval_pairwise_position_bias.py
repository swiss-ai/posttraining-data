#!/usr/bin/env python3
"""
Pairwise Position Bias Evaluation for Judge Models

Tests whether judge models exhibit position bias when comparing pairs of responses.
More closely mirrors actual pairwise comparison workflows by testing preferences
between specific pairs while controlling for position effects.

For each sample with multiple completions:
1. Creates multiple random orderings of the same completions
2. Tests random pairs from each ordering  
3. Tests both question orders ("A vs B" and "B vs A")
4. Records winner and position information for bias analysis

Usage:
    python eval_pairwise_position_bias.py input_dataset --output results.jsonl \
        --orderings 10 --max-samples 100 --pairs-per-ordering 5
"""

import os
import json
import time
import asyncio
import argparse
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque

from datasets import load_from_disk, Dataset
from tqdm import tqdm
import openai


# Configuration constants
MAX_RETRY_ATTEMPTS = 3
DEFAULT_ORDERINGS = 10
DEFAULT_CONCURRENT = 50
API_TIMEOUT = 120

# Adaptive concurrency constants
RAMP_UP_INITIAL = 100
RAMP_UP_INCREMENT = 300
RAMP_UP_INTERVAL = 10  # seconds
STABILITY_ZONE_THRESHOLD = 500  # tasks remaining


@dataclass
class PairwisePositionResult:
    """Result of a pairwise position bias test."""
    sample_id: str
    ordering_num: int
    pair_num: int  # Which pair in this ordering (1-based)
    reasoning_mode: str
    position_mapping: Dict[str, int]  # completion_id -> position (1-based)
    completion_a_id: str  # First completion in internal assignment
    completion_b_id: str  # Second completion in internal assignment
    completion_a_position: int  # Position of completion_a in the list (1-based)
    completion_b_position: int  # Position of completion_b in the list (1-based)
    first_mentioned_id: str  # ID of completion mentioned first in question
    second_mentioned_id: str  # ID of completion mentioned second in question
    first_mentioned_position: int  # Position of first-mentioned completion (1-based)
    second_mentioned_position: int  # Position of second-mentioned completion (1-based)
    question_order: str  # "original_order" or "reversed_order"
    winner: Optional[str]  # "first_mentioned" or "second_mentioned" 
    winner_completion_id: Optional[str]  # completion_id of winner
    position_distance: int  # abs(first_mentioned_position - second_mentioned_position)
    success: bool
    error: Optional[str]
    raw_response: str
    prompt_tokens: int
    response_tokens: int
    timestamp: float
    random_seed: int
    judge_model: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class RequestMetrics:
    """Track rolling window metrics for API requests."""
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.requests = deque()  # (timestamp, success_bool, latency_seconds)
        self.latencies = deque(maxlen=100)
        
    def record_request(self, success: bool, latency_seconds: float = 0.0):
        """Record a completed request."""
        timestamp = time.time()
        self.requests.append((timestamp, success, latency_seconds))
        if latency_seconds > 0:
            self.latencies.append(latency_seconds)
        self._cleanup_old_records()
    
    def _cleanup_old_records(self):
        """Remove records older than window."""
        cutoff = time.time() - self.window_seconds
        while self.requests and self.requests[0][0] < cutoff:
            self.requests.popleft()
    
    def get_metrics(self) -> dict:
        """Get current metrics for the rolling window."""
        self._cleanup_old_records()
        
        if not self.requests:
            return {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 0.0,
                'requests_per_minute': 0.0,
                'avg_latency': 0.0
            }
        
        total = len(self.requests)
        successful = sum(1 for _, success, _ in self.requests if success)
        failed = total - successful
        
        window_duration = min(self.window_seconds, 
                             time.time() - self.requests[0][0] if self.requests else 1)
        
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        
        return {
            'total_requests': total,
            'successful_requests': successful,
            'failed_requests': failed,
            'success_rate': successful / total if total > 0 else 0,
            'requests_per_minute': (total / window_duration) * 60 if window_duration > 0 else 0,
            'avg_latency': avg_latency
        }
    
    def get_adaptive_metrics(self) -> dict:
        """Get enhanced metrics for adaptive concurrency with Little's Law optimization."""
        basic_metrics = self.get_metrics()
        
        # Little's Law: optimal_concurrency = (requests_per_minute * avg_latency_seconds) / 60
        requests_per_minute = basic_metrics['requests_per_minute']
        avg_latency_seconds = basic_metrics['avg_latency']
        
        if requests_per_minute > 0 and avg_latency_seconds > 0:
            optimal_concurrent = max(1, int((requests_per_minute * avg_latency_seconds) / 60))
        else:
            optimal_concurrent = 10  # Fallback minimum
            
        return {
            **basic_metrics,
            'avg_latency_seconds': avg_latency_seconds,
            'optimal_concurrent': optimal_concurrent
        }


class PairwisePositionBiasTester:
    """Tests judge models for pairwise position bias."""
    
    def __init__(self, 
                 api_key: str,
                 model: str = "Qwen/Qwen3-32B",
                 orderings: int = DEFAULT_ORDERINGS,
                 pairs_per_ordering: int = 5,
                 test_both_orders: bool = True,
                 concurrent: int = DEFAULT_CONCURRENT,
                 base_url: Optional[str] = None,
                 disable_adaptive: bool = False,
                 reasoning_mode: str = "no_reasoning"):
        
        self.model = model
        self.orderings = orderings
        self.pairs_per_ordering = pairs_per_ordering
        self.test_both_orders = test_both_orders
        self.current_concurrent = concurrent
        self.initial_concurrent = concurrent
        self.disable_adaptive = disable_adaptive
        self.reasoning_mode = reasoning_mode
        
        # Advanced adaptive concurrency state
        self.min_concurrent = max(10, concurrent // 10)  # Minimum 10 or 10% of initial
        self.last_adaptation = time.time()
        self.stability_zone_entered = False
        self.metrics_display_interval = 5  # seconds
        self.last_metrics_display = time.time()
        
        # Setup OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.swissai.cscs.ch/v1"
        )
        
        # Load prompts
        prompt_dir = Path(__file__).parent / "prompts"
        with open(prompt_dir / "pairwise_position_bias_with_reasoning.txt") as f:
            self.prompt_with_reasoning = f.read()
        with open(prompt_dir / "pairwise_position_bias_no_reasoning.txt") as f:
            self.prompt_no_reasoning = f.read()
        
        # Metrics tracking
        self.metrics = RequestMetrics()
        self.last_adaptation = time.time()
    
    def _extract_completions(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract completions from a sample."""
        completions = []
        
        # Handle new format with conversation_branches
        if "conversation_branches" in sample:
            for idx, branch in enumerate(sample.get("conversation_branches", [])):
                messages = branch.get("messages", [])
                if messages:
                    # Get first assistant message
                    for msg in messages:
                        if msg.get("role") == "assistant":
                            # Handle parts structure
                            if "parts" in msg:
                                response_text = ""
                                for part in msg["parts"]:
                                    if part.get("type") == "response":
                                        response_text = part.get("content", "")
                                        break
                                if response_text:
                                    completions.append({
                                        "id": f"branch_{idx}",
                                        "content": response_text
                                    })
                            elif "content" in msg:
                                completions.append({
                                    "id": f"branch_{idx}",
                                    "content": msg["content"]
                                })
                            break
        
        # Fallback to old format
        elif "completions" in sample:
            for idx, comp in enumerate(sample["completions"]):
                if isinstance(comp, str):
                    completions.append({"id": f"comp_{idx}", "content": comp})
                elif isinstance(comp, dict) and "content" in comp:
                    completions.append({"id": f"comp_{idx}", "content": comp["content"]})
        
        return completions
    
    def _extract_initial_prompt(self, sample: Dict[str, Any]) -> str:
        """Extract the initial user prompt from a sample."""
        # Handle new format
        if "initial_prompt" in sample:
            initial = sample["initial_prompt"]
            if isinstance(initial, dict):
                return initial.get("content", "")
            return str(initial)
        
        # Fallback to old format
        if "prompt" in sample:
            return sample["prompt"]
        
        if "question" in sample:
            return sample["question"]
        
        return ""
    
    def _create_pairwise_test_cases(self, 
                                   sample: Dict[str, Any], 
                                   completions: List[Dict[str, str]], 
                                   ordering_num: int,
                                   reasoning_mode: str) -> List[Dict[str, Any]]:
        """Create pairwise test cases for a single ordering."""
        
        sample_id = sample.get("conversation_id", sample.get("id", str(hash(str(sample)))))
        base_seed = hash(f"{sample_id}_{ordering_num}_{reasoning_mode}")
        random.seed(base_seed)
        
        # Create random ordering
        completion_indices = list(range(len(completions)))
        random.shuffle(completion_indices)
        
        # Create position mappings
        position_mapping = {}  # completion_id -> position (1-based)
        for position, idx in enumerate(completion_indices, 1):
            comp_id = completions[idx]["id"]
            position_mapping[comp_id] = position
        
        # Get ordered completions
        ordered_completions = [completions[idx] for idx in completion_indices]
        
        # Generate random pairs for this ordering
        test_cases = []
        available_pairs = []
        
        # Create all possible pairs
        for i in range(len(ordered_completions)):
            for j in range(i + 1, len(ordered_completions)):
                available_pairs.append((i, j))
        
        # Select random pairs
        random.shuffle(available_pairs)
        selected_pairs = available_pairs[:min(self.pairs_per_ordering, len(available_pairs))]
        
        for pair_num, (idx_a, idx_b) in enumerate(selected_pairs, 1):
            comp_a = ordered_completions[idx_a]
            comp_b = ordered_completions[idx_b]
            
            # Create test cases for both question orders if enabled
            orders_to_test = []
            if self.test_both_orders:
                orders_to_test = [("original_order", comp_a, comp_b), ("reversed_order", comp_b, comp_a)]
            else:
                orders_to_test = [("original_order", comp_a, comp_b)]
            
            for question_order, first_comp, second_comp in orders_to_test:
                test_cases.append({
                    "sample_id": sample_id,
                    "ordering_num": ordering_num,
                    "pair_num": pair_num,
                    "reasoning_mode": reasoning_mode,
                    "position_mapping": position_mapping,
                    "ordered_completions": ordered_completions,
                    "completion_a": first_comp,
                    "completion_b": second_comp,
                    "completion_a_position": position_mapping[first_comp["id"]],
                    "completion_b_position": position_mapping[second_comp["id"]],
                    "question_order": question_order,
                    "initial_prompt": self._extract_initial_prompt(sample),
                    "random_seed": base_seed + pair_num * 1000 + (1 if question_order == "reversed_order" else 0)
                })
        
        return test_cases
    
    def _format_completions_for_pairwise_prompt(self, ordered_completions: List[Dict[str, str]], 
                                              completion_a: Dict[str, str], completion_b: Dict[str, str]) -> Tuple[str, str, str]:
        """Format completions for pairwise comparison prompt."""
        # Display all completions with letter labels
        all_completions_text = []
        for i, comp in enumerate(ordered_completions):
            letter = chr(65 + i)  # A, B, C, ...
            all_completions_text.append(f"**Response {letter}:**\n{comp['content']}")
        
        # Find the labels for the specific pair being compared
        completion_a_label = None
        completion_b_label = None
        
        for i, comp in enumerate(ordered_completions):
            letter = chr(65 + i)
            if comp["id"] == completion_a["id"]:
                completion_a_label = letter
            if comp["id"] == completion_b["id"]:
                completion_b_label = letter
        
        return "\n\n".join(all_completions_text), completion_a_label, completion_b_label
    
    async def _call_judge(self, test_case: Dict[str, Any]) -> PairwisePositionResult:
        """Call the judge API for a single pairwise test case."""
        
        # Select prompt based on reasoning mode
        if test_case["reasoning_mode"] == "with_reasoning":
            prompt_template = self.prompt_with_reasoning
        else:
            prompt_template = self.prompt_no_reasoning
        
        # Format the prompt
        completions_text, completion_a_label, completion_b_label = self._format_completions_for_pairwise_prompt(
            test_case["ordered_completions"], 
            test_case["completion_a"], 
            test_case["completion_b"]
        )
        
        # For reversed order, swap the labels in the comparison question
        if test_case["question_order"] == "reversed_order":
            first_label, second_label = completion_b_label, completion_a_label
        else:
            first_label, second_label = completion_a_label, completion_b_label
        
        prompt = prompt_template.format(
            initial_prompt=test_case["initial_prompt"],
            completions=completions_text,
            completion_a_label=first_label,
            completion_b_label=second_label
        )
        
        # Call API with retries
        last_error = None
        
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                start_time = time.time()
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                    timeout=API_TIMEOUT
                )
                
                latency = time.time() - start_time
                
                # Parse response
                raw_response = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                response_tokens = response.usage.completion_tokens if response.usage else 0
                
                # Extract winner from JSON response
                winner_label = self._parse_pairwise_response(raw_response, test_case["reasoning_mode"])
                
                # Determine presentation order based on question order
                if test_case["question_order"] == "reversed_order":
                    # In reversed: first_mentioned = comp_b, second_mentioned = comp_a
                    first_mentioned_id = test_case["completion_b"]["id"]
                    second_mentioned_id = test_case["completion_a"]["id"]
                    first_mentioned_position = test_case["completion_b_position"]
                    second_mentioned_position = test_case["completion_a_position"]
                else:
                    # In original: first_mentioned = comp_a, second_mentioned = comp_b
                    first_mentioned_id = test_case["completion_a"]["id"]
                    second_mentioned_id = test_case["completion_b"]["id"]
                    first_mentioned_position = test_case["completion_a_position"]
                    second_mentioned_position = test_case["completion_b_position"]
                
                # Map winner to presentation order by checking which completion was actually chosen
                if winner_label and winner_label in [first_label, second_label]:
                    # Find which completion the judge actually picked
                    if winner_label == completion_a_label:
                        winner_completion_id = test_case["completion_a"]["id"]
                    elif winner_label == completion_b_label:
                        winner_completion_id = test_case["completion_b"]["id"]
                    else:
                        winner_completion_id = None
                    
                    # Now determine if the chosen completion was mentioned first or second
                    if winner_completion_id == first_mentioned_id:
                        winner = "first_mentioned"
                    elif winner_completion_id == second_mentioned_id:
                        winner = "second_mentioned"
                    else:
                        winner = None
                    
                    self.metrics.record_request(True, latency)
                    
                    return PairwisePositionResult(
                        sample_id=test_case["sample_id"],
                        ordering_num=test_case["ordering_num"],
                        pair_num=test_case["pair_num"],
                        reasoning_mode=test_case["reasoning_mode"],
                        position_mapping=test_case["position_mapping"],
                        completion_a_id=test_case["completion_a"]["id"],
                        completion_b_id=test_case["completion_b"]["id"],
                        completion_a_position=test_case["completion_a_position"],
                        completion_b_position=test_case["completion_b_position"],
                        first_mentioned_id=first_mentioned_id,
                        second_mentioned_id=second_mentioned_id,
                        first_mentioned_position=first_mentioned_position,
                        second_mentioned_position=second_mentioned_position,
                        question_order=test_case["question_order"],
                        winner=winner,
                        winner_completion_id=winner_completion_id,
                        position_distance=abs(first_mentioned_position - second_mentioned_position),
                        success=True,
                        error=None,
                        raw_response=raw_response,
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
                        timestamp=time.time(),
                        random_seed=test_case["random_seed"],
                        judge_model=self.model
                    )
                
                # Failed to parse
                last_error = f"Failed to parse response: {raw_response[:200]}"
                
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(2 ** attempt)
        
        # All attempts failed - trigger immediate concurrency reduction
        self.metrics.record_request(False)
        self._immediate_failure_response()
        
        # Calculate presentation order for failed case too
        if test_case["question_order"] == "reversed_order":
            first_mentioned_id = test_case["completion_b"]["id"]
            second_mentioned_id = test_case["completion_a"]["id"]
            first_mentioned_position = test_case["completion_b_position"]
            second_mentioned_position = test_case["completion_a_position"]
        else:
            first_mentioned_id = test_case["completion_a"]["id"]
            second_mentioned_id = test_case["completion_b"]["id"]
            first_mentioned_position = test_case["completion_a_position"]
            second_mentioned_position = test_case["completion_b_position"]
            
        return PairwisePositionResult(
            sample_id=test_case["sample_id"],
            ordering_num=test_case["ordering_num"],
            pair_num=test_case["pair_num"],
            reasoning_mode=test_case["reasoning_mode"],
            position_mapping=test_case["position_mapping"],
            completion_a_id=test_case["completion_a"]["id"],
            completion_b_id=test_case["completion_b"]["id"],
            completion_a_position=test_case["completion_a_position"],
            completion_b_position=test_case["completion_b_position"],
            first_mentioned_id=first_mentioned_id,
            second_mentioned_id=second_mentioned_id,
            first_mentioned_position=first_mentioned_position,
            second_mentioned_position=second_mentioned_position,
            question_order=test_case["question_order"],
            winner=None,
            winner_completion_id=None,
            position_distance=abs(first_mentioned_position - second_mentioned_position),
            success=False,
            error=last_error,
            raw_response="",
            prompt_tokens=0,
            response_tokens=0,
            timestamp=time.time(),
            random_seed=test_case["random_seed"],
            judge_model=self.model
        )
    
    def _parse_pairwise_response(self, response: str, reasoning_mode: str) -> Optional[str]:
        """Parse the judge's response to extract winner label."""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                winner = result.get("winner", "").strip().upper()
                
                if winner and len(winner) == 1 and winner.isalpha():
                    return winner
        except:
            pass
        
        return None
    
    async def _process_with_adaptive_concurrency(self, test_cases: List[Dict[str, Any]]) -> List[PairwisePositionResult]:
        """Process test cases with advanced adaptive concurrency, ramp-up, and stability zones."""
        
        # Reset adaptation state for new batch
        self.last_adaptation = time.time()
        self.stability_zone_entered = False
        
        # Set up ramp-up strategy
        ramp_up_target = self.current_concurrent
        is_ramping_up = False
        last_ramp_up_increase = time.time()
        
        if not self.disable_adaptive and ramp_up_target > RAMP_UP_INITIAL:
            self.current_concurrent = RAMP_UP_INITIAL
            is_ramping_up = True
            print(f"🔺 Starting ramp-up: {RAMP_UP_INITIAL} → {ramp_up_target} concurrent requests")
        
        # Processing state
        in_flight_tasks = set()
        work_queue = list(enumerate(test_cases))
        completed_results = [None] * len(test_cases)
        
        pbar = tqdm(total=len(test_cases), desc="Testing position bias")
        
        while work_queue or in_flight_tasks:
            current_time = time.time()
            
            # Ramp-up logic
            if is_ramping_up and current_time - last_ramp_up_increase >= RAMP_UP_INTERVAL:
                old_concurrent = self.current_concurrent
                self.current_concurrent = min(
                    self.current_concurrent + RAMP_UP_INCREMENT,
                    ramp_up_target
                )
                if self.current_concurrent != old_concurrent:
                    print(f"🔺 Ramping up: {old_concurrent} → {self.current_concurrent}")
                last_ramp_up_increase = current_time
                
                if self.current_concurrent >= ramp_up_target:
                    is_ramping_up = False
                    print(f"🔺 Ramp-up complete: {self.current_concurrent} concurrent requests")
                    # Reset adaptation timer after ramp-up completes
                    self.last_adaptation = current_time
            
            # Start new tasks up to current limit
            while len(in_flight_tasks) < self.current_concurrent and work_queue:
                idx, test_case = work_queue.pop(0)
                
                async def process_with_index(idx, tc):
                    result = await self._call_judge(tc)
                    return idx, result
                
                task = asyncio.create_task(process_with_index(idx, test_case))
                in_flight_tasks.add(task)
            
            # Wait for tasks with timeout
            if in_flight_tasks:
                done, pending = await asyncio.wait(
                    in_flight_tasks,
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task in done:
                    in_flight_tasks.remove(task)
                    try:
                        idx, result = await task
                        completed_results[idx] = result
                        pbar.update(1)
                    except Exception as e:
                        print(f"Task error: {e}")
                
                # Periodic adaptation (not during ramp-up)
                if not is_ramping_up:
                    remaining_tasks = len(work_queue) + len(in_flight_tasks)
                    self._adapt_concurrency(remaining_tasks)
                
                # Enhanced progress display
                if current_time - self.last_metrics_display >= self.metrics_display_interval:
                    metrics = self.metrics.get_adaptive_metrics()
                    error_rate = metrics['failed_requests'] / max(metrics['total_requests'], 1)
                    postfix = {
                        'req/min': f"{metrics['requests_per_minute']:.1f}",
                        'success': f"{metrics['success_rate']*100:.0f}%",
                        'error_rate': f"{error_rate*100:.1f}%",
                        'concurrent': self.current_concurrent
                    }
                    pbar.set_postfix(postfix)
                    self.last_metrics_display = current_time
        
        pbar.close()
        
        # Filter out None results
        return [r for r in completed_results if r is not None]
    
    def _immediate_failure_response(self):
        """Immediate concurrency reduction on API failure."""
        if self.disable_adaptive:
            return
            
        old_concurrent = self.current_concurrent
        self.current_concurrent = max(
            self.current_concurrent - 1, 
            self.min_concurrent
        )
        
        if self.current_concurrent < old_concurrent:
            print(f"🔻 Reducing concurrency: {old_concurrent} → {self.current_concurrent} (immediate failure response)")
    
    def _adapt_concurrency(self, remaining_tasks: int = 0):
        """Periodic concurrency adaptation based on performance metrics."""
        if self.disable_adaptive:
            return
            
        now = time.time()
        
        # Only adapt every 60 seconds for periodic optimization
        if now - self.last_adaptation < 60:
            return
            
        # Enter stability zone for final tasks to prevent thrashing
        if remaining_tasks > 0 and remaining_tasks <= STABILITY_ZONE_THRESHOLD and not self.stability_zone_entered:
            self.stability_zone_entered = True
            print(f"🔒 Entering stability zone: preserving concurrency at {self.current_concurrent} for final {remaining_tasks} tasks")
            return
            
        if self.stability_zone_entered:
            return  # No changes in stability zone
            
        metrics = self.metrics.get_adaptive_metrics()
        error_rate = metrics['failed_requests'] / max(metrics['total_requests'], 1)
        old_concurrent = self.current_concurrent
        
        # Periodic optimization based on error rate and Little's Law
        if error_rate <= 0.01 and metrics.get('avg_latency_seconds', 0) < 20.0:  # <1% error rate and low latency
            # Explore higher concurrency (+20)
            self.current_concurrent += 20
            print(f"🔺 Increasing concurrency: {old_concurrent} → {self.current_concurrent} ({error_rate*100:.1f}% error rate, avg {metrics.get('avg_latency_seconds', 0):.1f}s)")
        else:
            # Fall back to calculated optimal
            optimal = max(metrics['optimal_concurrent'], self.min_concurrent)
            self.current_concurrent = min(optimal, self.current_concurrent)  # Don't increase if errors
            if self.current_concurrent < old_concurrent:
                print(f"🔻 Reducing to optimal: {old_concurrent} → {self.current_concurrent} (error rate: {error_rate*100:.1f}%)")
        
        self.last_adaptation = now
    
    async def test_samples(self, samples: List[Dict[str, Any]]) -> List[PairwisePositionResult]:
        """Test multiple samples for pairwise position bias."""
        
        # Generate all test cases
        test_cases = []
        
        for sample in samples:
            completions = self._extract_completions(sample)
            
            if len(completions) < 2:
                continue  # Need at least 2 completions for pairwise comparison
            
            # Limit to 10 completions max for testing
            if len(completions) > 10:
                completions = completions[:10]
            
            # Create pairwise test cases for each ordering
            for ordering_num in range(1, self.orderings + 1):
                if self.reasoning_mode == "both":
                    # Test both reasoning modes
                    test_cases.extend(self._create_pairwise_test_cases(
                        sample, completions, ordering_num, "with_reasoning"
                    ))
                    test_cases.extend(self._create_pairwise_test_cases(
                        sample, completions, ordering_num, "no_reasoning"
                    ))
                else:
                    # Test only the specified reasoning mode
                    test_cases.extend(self._create_pairwise_test_cases(
                        sample, completions, ordering_num, self.reasoning_mode
                    ))
        
        modes_per_sample = 2 if self.reasoning_mode == "both" else 1
        question_orders_per_pair = 2 if self.test_both_orders else 1
        print(f"Generated {len(test_cases)} test cases from {len(samples)} samples")
        print(f"  {self.orderings} orderings × {self.pairs_per_ordering} pairs × {modes_per_sample} reasoning mode(s) × {question_orders_per_pair} question order(s)")
        
        # Process all test cases
        results = await self._process_with_adaptive_concurrency(test_cases)
        
        return results


async def main():
    parser = argparse.ArgumentParser(description="Test judge models for pairwise position bias")
    parser.add_argument("input", help="Input dataset path")
    parser.add_argument("--output", default=None, help="Output JSONL file path (auto-generated if not specified)")
    parser.add_argument("--orderings", type=int, default=DEFAULT_ORDERINGS,
                       help="Number of random orderings per sample")
    parser.add_argument("--pairs-per-ordering", type=int, default=5,
                       help="Number of random pairs to test per ordering (default: 5)")
    parser.add_argument("--no-both-orders", action="store_true",
                       help="Only test 'A vs B', not both 'A vs B' and 'B vs A' (default: test both)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to test")
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT,
                       help="Maximum concurrent API requests")
    parser.add_argument("--disable-adaptive", action="store_true",
                       help="Disable adaptive concurrency")
    parser.add_argument("--model", default="Qwen/Qwen3-32B",
                       help="Model to use for judging")
    parser.add_argument("--reasoning-mode", choices=["with_reasoning", "no_reasoning", "both"], 
                       default="no_reasoning",
                       help="Reasoning mode: with_reasoning, no_reasoning, or both (default: no_reasoning)")
    parser.add_argument("--api-key", default=None,
                       help="API key (defaults to SWISS_AI_API_KEY env var)")
    parser.add_argument("--base-url", default=None,
                       help="API base URL")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("SWISSAI_API_KEY") or os.getenv("SWISS_AI_API_KEY")
    if not api_key:
        print("Error: API key required (--api-key or SWISSAI_API_KEY env var)")
        return
    
    # Load dataset
    print(f"Loading dataset from {args.input}")
    dataset = load_from_disk(args.input)
    
    # Handle DatasetDict
    if hasattr(dataset, 'keys'):
        if 'train' in dataset:
            dataset = dataset['train']
        else:
            dataset = dataset[list(dataset.keys())[0]]
    
    # Sample selection
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    print(f"Testing {len(dataset)} samples")
    
    # Auto-generate output filename if not provided
    if args.output is None:
        # Clean up model name for filename
        model_name = args.model.replace("/", "_").replace("-", "_").lower()
        if "qwen" in model_name:
            model_short = "qwen"
        elif "claude" in model_name:
            model_short = "claude"
        elif "llama" in model_name:
            model_short = "llama"
        else:
            model_short = model_name.split("_")[0]
        
        num_samples = len(dataset)
        orderings = args.orderings
        pairs_per_ordering = args.pairs_per_ordering
        reasoning_suffix = f"_{args.reasoning_mode}" if args.reasoning_mode != "no_reasoning" else ""
        orders_suffix = "_single_order" if args.no_both_orders else ""
        
        filename = f"pairwise_bias_{model_short}_{num_samples}samples_{orderings}orders_{pairs_per_ordering}pairs{reasoning_suffix}{orders_suffix}.jsonl"
        output_path = Path("analysis") / filename
        args.output = str(output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Auto-generated output path: {args.output}")
    
    # Create tester
    tester = PairwisePositionBiasTester(
        api_key=api_key,
        model=args.model,
        orderings=args.orderings,
        pairs_per_ordering=args.pairs_per_ordering,
        test_both_orders=not args.no_both_orders,
        concurrent=args.concurrent,
        base_url=args.base_url,
        disable_adaptive=args.disable_adaptive,
        reasoning_mode=args.reasoning_mode
    )
    
    # Run tests
    samples = [dataset[i] for i in range(len(dataset))]
    results = await tester.test_samples(samples)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + '\n')
    
    print(f"\nResults saved to {output_path}")
    print(f"  Total test cases: {len(results)}")
    print(f"  Successful: {sum(1 for r in results if r.success)}")
    print(f"  Failed: {sum(1 for r in results if not r.success)}")
    
    # Quick pairwise analysis
    winner_counts = {"first_mentioned": 0, "second_mentioned": 0}
    position_distance_wins = {}
    question_order_wins = {"original_order": {"first_mentioned": 0, "second_mentioned": 0}, "reversed_order": {"first_mentioned": 0, "second_mentioned": 0}}
    
    for result in results:
        if result.success and result.winner:
            winner_counts[result.winner] += 1
            
            # Track wins by position distance
            dist = result.position_distance
            if dist not in position_distance_wins:
                position_distance_wins[dist] = {"first_mentioned": 0, "second_mentioned": 0}
            position_distance_wins[dist][result.winner] += 1
            
            # Track wins by question order
            question_order_wins[result.question_order][result.winner] += 1
    
    if sum(winner_counts.values()) > 0:
        print("\nQuick pairwise results summary:")
        total = sum(winner_counts.values())
        print(f"  First-mentioned wins: {winner_counts['first_mentioned']} ({winner_counts['first_mentioned']/total*100:.1f}%)")
        print(f"  Second-mentioned wins: {winner_counts['second_mentioned']} ({winner_counts['second_mentioned']/total*100:.1f}%)")
        
        print("\nBy question order:")
        for order, wins in question_order_wins.items():
            order_total = sum(wins.values())
            if order_total > 0:
                print(f"  {order}: First={wins['first_mentioned']} ({wins['first_mentioned']/order_total*100:.1f}%), Second={wins['second_mentioned']} ({wins['second_mentioned']/order_total*100:.1f}%)")
        
        print("\nBy position distance:")
        for dist in sorted(position_distance_wins.keys()):
            wins = position_distance_wins[dist]
            dist_total = sum(wins.values())
            if dist_total > 0:
                print(f"  Distance {dist}: First={wins['first_mentioned']} ({wins['first_mentioned']/dist_total*100:.1f}%), Second={wins['second_mentioned']} ({wins['second_mentioned']/dist_total*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())