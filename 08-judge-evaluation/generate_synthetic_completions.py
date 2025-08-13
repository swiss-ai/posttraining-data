#!/usr/bin/env python3
"""
Generate synthetic degraded completions for preference learning datasets.

Takes samples from a dataset and generates N progressively degraded completions
using the iterative degradation logic. Creates a new dataset with original +
degraded completions as conversation branches for preference training.

Usage:
    python generate_synthetic_completions.py input_dataset --samples 100 --iterations 5
"""

import argparse
import asyncio
import json
import os
import time
import random
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import deque

from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
import openai


# Configuration constants
MAX_RETRY_ATTEMPTS = 3
DEFAULT_CONCURRENT = 50
API_TIMEOUT = 120

# Adaptive concurrency constants
RAMP_UP_INITIAL = 100
RAMP_UP_INCREMENT = 300
RAMP_UP_INTERVAL = 10  # seconds
STABILITY_ZONE_THRESHOLD = 500  # tasks remaining


@dataclass
class SampleState:
    """State tracking for a single sample through degradation iterations."""
    conversation_id: str
    user_prompt: str
    original_completion: str
    completions_history: List[str]  # [original, iter1, iter2, ...]
    reasonings_history: List[str]   # [iter1_reasoning, iter2_reasoning, ...]
    current_iteration: int
    original_sample: Dict[str, Any]  # Store original sample for schema
    token_usage: List[Dict[str, Any]] = None  # Token usage per iteration
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SampleState':
        """Create SampleState from dictionary."""
        return cls(**data)


class SwissAIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.swissai.cscs.ch/v1"):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()
    
    async def chat_completion(self, messages, model="Qwen/Qwen3-32B", max_tokens=1000):
        """Make a chat completion request."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                stream=False,
                timeout=120.0
            )
            return response.choices[0].message.content, response.usage
        except Exception as e:
            raise Exception(f"API request failed: {e}")


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
    
    def get_adaptive_metrics(self) -> dict:
        """Get enhanced metrics for adaptive concurrency with Little's Law optimization."""
        self._cleanup_old_records()
        
        if not self.requests:
            return {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 0.0,
                'requests_per_minute': 0.0,
                'avg_latency_seconds': 0.0,
                'optimal_concurrent': 10
            }
        
        total = len(self.requests)
        successful = sum(1 for _, success, _ in self.requests if success)
        failed = total - successful
        
        window_duration = min(self.window_seconds, 
                             time.time() - self.requests[0][0] if self.requests else 1)
        
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        requests_per_minute = (total / window_duration) * 60 if window_duration > 0 else 0
        
        # Little's Law: optimal_concurrency = (requests_per_minute * avg_latency_seconds) / 60
        if requests_per_minute > 0 and avg_latency > 0:
            optimal_concurrent = max(1, int((requests_per_minute * avg_latency) / 60))
        else:
            optimal_concurrent = 10  # Fallback minimum
            
        return {
            'total_requests': total,
            'successful_requests': successful,
            'failed_requests': failed,
            'success_rate': successful / total if total > 0 else 0,
            'requests_per_minute': requests_per_minute,
            'avg_latency_seconds': avg_latency,
            'optimal_concurrent': optimal_concurrent
        }


def extract_original_completion(sample, target_model="Qwen/Qwen3-32B"):
    """Extract the original completion from the target model."""
    for branch in sample.get('conversation_branches', []):
        for message in branch.get('messages', []):
            if message.get('role') == 'assistant' and 'parts' in message:
                for part in message.get('parts', []):
                    if isinstance(part, dict) and 'metadata' in part:
                        model = part['metadata'].get('model')
                        if model == target_model and part.get('type') == 'response':
                            return part.get('content', '')
    return None


async def degrade_completion(client, sample_state: SampleState, iteration: int, metrics: RequestMetrics):
    """Degrade a completion by identifying a dimension and making it worse."""
    
    # Build the full history for context (truncate completions to 500 chars, keep full reasonings)
    history_text = ""
    for i, (completion, reasoning) in enumerate(zip(sample_state.completions_history, sample_state.reasonings_history + [None])):
        # Truncate completion to first 500 chars
        truncated_completion = completion[:500] + "..." if len(completion) > 500 else completion
        
        if i == 0:
            history_text += f"ORIGINAL COMPLETION:\n{truncated_completion}\n\n"
        else:
            prev_reasoning = sample_state.reasonings_history[i-1] if i-1 < len(sample_state.reasonings_history) else ""
            history_text += f"ITERATION {i} (Previous Reasoning: {prev_reasoning}):\n{truncated_completion}\n\n"
    
    # Ask Qwen to reason about what to change and then generate the worse completion
    degradation_prompt = f"""You are helping create training data by generating an alternative version of an AI response. 

User Prompt: {sample_state.user_prompt}

Full Degradation History:
{history_text}

Current Latest Completion: {sample_state.completions_history[-1]}

IMPORTANT: Make the completion objectively worse in quality, not just different in content. Focus on degrading the AI's response quality, not changing the narrative content.

Please respond using EXACTLY this format:

REASONING:
Look at the full degradation history above and identify ONE NEW dimension that hasn't been degraded yet to make the response objectively worse in quality. Choose from these possible modifications: lower factual accuracy (add wrong facts, incorrect dates/numbers), reduce logical coherence (make arguments contradictory or illogical), make it incomplete (remove key parts, leave things unfinished), worsen organization/structure (poor flow, confusing order, bad formatting), make it unfocused on the task (add irrelevant information, go off-topic), reduce language quality (introduce typos, grammatical errors, unclear phrasing), use inappropriate certainty levels (be overconfident about uncertain things or uncertain about facts), ignore format instructions (if specific format was requested), skip/ignore parts of the instructions, add faulty reasoning (use incorrect logic, make wrong assumptions, draw invalid conclusions), or provide wrong/no answers (give incorrect final answers, fail to answer the question, or provide no conclusion at all). Select a NEW dimension that hasn't been used in previous iterations. Explain specifically what NEW dimension you will change. IMPORTANT: The degradation should be SIGNIFICANT and HARD TO MISS, not subtle - make sure the quality drop is obvious and noticeable.

COMPLETION:
CRITICAL: You must preserve ALL previous degradations from the latest completion while adding the new degradation. Do not fix, remove, or undo any of the existing problems - keep all previous typos, errors, inconsistencies, missing parts, etc. from the current latest completion. Only ADD the new degradation on top of the existing issues. The new degradation should be SIGNIFICANT and HARD TO MISS - not a subtle change but an obvious quality problem that clearly makes the response worse. Start with the current latest completion and make it noticeably worse in the new dimension while keeping all existing degradations intact. Generate a completely natural response without any brackets, notes, or annotations indicating what was changed. Make the degradation seamless and natural - do not add parenthetical comments or explanatory notes about the modifications. DO NOT warn the user about any errors, problems, or issues in your response - act as if the degraded response is normal and complete."""
    
    messages = [{"role": "user", "content": degradation_prompt}]
    
    # Call API with retries
    last_error = None
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            start_time = time.time()
            response, usage = await client.chat_completion(messages, max_tokens=1500)
            latency = time.time() - start_time
            
            # Parse the delimited response
            try:
                if "REASONING:" in response and "COMPLETION:" in response:
                    parts = response.split("COMPLETION:")
                    reasoning = parts[0].replace("REASONING:", "").strip()
                    worse_completion = parts[1].strip()
                else:
                    # Fallback if format not followed
                    reasoning = "Format not followed correctly"
                    worse_completion = response
            except Exception as e:
                # Fallback if parsing fails
                reasoning = f"Parsing failed: {e}"
                worse_completion = response
            
            metrics.record_request(True, latency)
            return worse_completion, reasoning, usage
            
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRY_ATTEMPTS - 1:
                await asyncio.sleep(2 ** attempt)
    
    # All attempts failed
    metrics.record_request(False)
    raise Exception(f"API request failed after {MAX_RETRY_ATTEMPTS} attempts: {last_error}")


def initialize_sample_states(dataset, num_samples: int, target_model: str = "Qwen/Qwen3-32B") -> List[SampleState]:
    """Initialize sample states from dataset."""
    sample_states = []
    
    # Handle DatasetDict or single Dataset
    if hasattr(dataset, 'keys'):
        data = dataset[list(dataset.keys())[0]]
    else:
        data = dataset
    
    for i in range(min(num_samples, len(data))):
        sample = data[i]
        
        # Extract user prompt and original completion
        user_prompt = sample['initial_prompt']['content']
        original_completion = extract_original_completion(sample, target_model)
        
        if not original_completion:
            print(f"Warning: No original completion found for sample {sample['conversation_id']}, skipping")
            continue
        
        sample_state = SampleState(
            conversation_id=sample['conversation_id'],
            user_prompt=user_prompt,
            original_completion=original_completion,
            completions_history=[original_completion],
            reasonings_history=[],
            current_iteration=0,
            original_sample=sample,
            token_usage=[]
        )
        sample_states.append(sample_state)
    
    return sample_states


class StateManager:
    """Manages state persistence for resume/restart functionality."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.state_dir = output_path.parent / f".{output_path.name}_state"
        self.config_file = self.state_dir / "config.json"
        self.samples_file = self.state_dir / "samples.json"
        self.progress_file = self.state_dir / "progress.json"
    
    def state_exists(self) -> bool:
        """Check if state files exist."""
        return self.state_dir.exists() and self.config_file.exists()
    
    def save_initial_state(self, sample_states: List[SampleState], config: dict):
        """Save initial configuration and sample states."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save sample states
        samples_data = [ss.to_dict() for ss in sample_states]
        with open(self.samples_file, 'w') as f:
            json.dump(samples_data, f, indent=2)
        
        # Save initial progress
        self.save_progress(0, len(sample_states))
    
    def save_progress(self, completed_iteration: int, total_samples: int):
        """Save progress after each iteration."""
        progress_data = {
            "completed_iteration": completed_iteration,
            "total_samples": total_samples,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def save_samples(self, sample_states: List[SampleState]):
        """Save updated sample states."""
        samples_data = [ss.to_dict() for ss in sample_states]
        with open(self.samples_file, 'w') as f:
            json.dump(samples_data, f, indent=2)
    
    def load_state(self) -> tuple[List[SampleState], dict, dict]:
        """Load existing state."""
        # Load configuration
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        # Load sample states
        with open(self.samples_file, 'r') as f:
            samples_data = json.load(f)
        sample_states = [SampleState.from_dict(data) for data in samples_data]
        
        # Load progress
        with open(self.progress_file, 'r') as f:
            progress = json.load(f)
        
        return sample_states, config, progress
    
    def cleanup_state(self):
        """Remove state directory."""
        if self.state_dir.exists():
            shutil.rmtree(self.state_dir)


class SyntheticCompletionsGenerator:
    """Generates synthetic degraded completions with adaptive concurrency."""
    
    def __init__(self, 
                 api_key: str,
                 concurrent: int = DEFAULT_CONCURRENT,
                 disable_adaptive: bool = False,
                 state_manager: Optional[StateManager] = None):
        
        self.current_concurrent = concurrent
        self.initial_concurrent = concurrent
        self.disable_adaptive = disable_adaptive
        self.state_manager = state_manager
        
        # Advanced adaptive concurrency state
        self.min_concurrent = max(10, concurrent // 10)  # Minimum 10 or 10% of initial
        self.last_adaptation = time.time()
        self.stability_zone_entered = False
        self.metrics_display_interval = 5  # seconds
        self.last_metrics_display = time.time()
        
        # Setup client
        self.api_key = api_key
        
        # Metrics tracking
        self.metrics = RequestMetrics()
        self.last_adaptation = time.time()
    
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
    
    async def _process_iteration_batch(self, sample_states: List[SampleState], iteration: int) -> None:
        """Process a single iteration for all samples with adaptive concurrency."""
        
        # Reset adaptation state for new iteration
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
        work_queue = list(enumerate(sample_states))
        
        pbar = tqdm(total=len(sample_states), desc=f"Iteration {iteration}")
        
        async with SwissAIClient(self.api_key) as client:
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
                    idx, sample_state = work_queue.pop(0)
                    
                    async def process_with_index(idx, ss):
                        try:
                            worse_completion, reasoning, usage = await degrade_completion(
                                client, ss, iteration, self.metrics
                            )
                            
                            # Update sample state
                            ss.completions_history.append(worse_completion)
                            ss.reasonings_history.append(reasoning)
                            ss.current_iteration = iteration
                            
                            # Store token usage for this iteration
                            if not hasattr(ss, 'token_usage'):
                                ss.token_usage = []
                            ss.token_usage.append({
                                'iteration': iteration,
                                'prompt_tokens': usage.prompt_tokens,
                                'completion_tokens': usage.completion_tokens,
                                'total_tokens': usage.total_tokens
                            })
                            
                            # Warning for context limits based on actual prompt tokens (32k limit)
                            if usage.prompt_tokens > 24000:  # Warning at 24k tokens (75% of 32k limit)
                                print(f"  ⚠️  WARNING: {usage.prompt_tokens:,} prompt tokens - approaching 32k token limit!")
                            
                            return idx, True, None
                            
                        except Exception as e:
                            self._immediate_failure_response()
                            return idx, False, str(e)
                    
                    task = asyncio.create_task(process_with_index(idx, sample_state))
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
                            idx, success, error = await task
                            if not success:
                                print(f"Error processing sample {idx}: {error}")
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
    
    async def generate_degradations(self, sample_states: List[SampleState], num_iterations: int, start_iteration: int = 1) -> List[SampleState]:
        """Generate degradations for all samples across multiple iterations."""
        if start_iteration == 1:
            print(f"Generating {num_iterations} iterations for {len(sample_states)} samples")
        else:
            print(f"Resuming from iteration {start_iteration}/{num_iterations} for {len(sample_states)} samples")
        
        for iteration in range(start_iteration, num_iterations + 1):
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration}/{num_iterations}")
            print(f"{'='*80}")
            
            await self._process_iteration_batch(sample_states, iteration)
            
            # Save state after each iteration
            if self.state_manager:
                self.state_manager.save_samples(sample_states)
                self.state_manager.save_progress(iteration, len(sample_states))
                print(f"State saved after iteration {iteration}")
            
            # Show progress summary
            print(f"Completed iteration {iteration}")
            successful = sum(1 for ss in sample_states if len(ss.completions_history) >= iteration + 1)
            print(f"  Successful: {successful}/{len(sample_states)} samples")
        
        return sample_states


def create_schema_compliant_part(part_type: str, content: str = "", metadata: Optional[Dict] = None, 
                                 name: str = "", args: str = "") -> Dict[str, Any]:
    """Create a schema-compliant part with all required fields for Arrow compatibility."""
    return {
        "type": part_type,
        "content": content,
        "metadata": metadata or {},
        "name": name,
        "args": args
    }


def convert_to_dataset_format(sample_states: List[SampleState]) -> List[Dict[str, Any]]:
    """Convert sample states to new chat format dataset structure."""
    dataset_samples = []
    
    for sample_state in sample_states:
        original_sample = sample_state.original_sample
        
        # Create conversation branches for original + degraded completions
        branches_with_metadata = []
        
        for completion_idx, completion in enumerate(sample_state.completions_history):
            # Create assistant message with parts structure
            if completion_idx == 0:
                # Original completion - ensure consistent schema with empty values
                metadata = {
                    "model": "Qwen/Qwen3-32B",
                    "degraded": False,
                    "iteration": 0,
                    "degradation_rank": 0,
                    "degradation_reasoning": "",  # Empty string for consistent schema
                    "prompt_tokens": 0,          # 0 for original (no API call)
                    "completion_tokens": 0,      # 0 for original (no API call)
                    "total_tokens": 0            # 0 for original (no API call)
                }
                branch_type = "original"
            else:
                # Degraded completion
                reasoning = sample_state.reasonings_history[completion_idx - 1] if completion_idx - 1 < len(sample_state.reasonings_history) else ""
                
                # Get token usage for this iteration
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if hasattr(sample_state, 'token_usage') and sample_state.token_usage:
                    for usage in sample_state.token_usage:
                        if usage.get('iteration') == completion_idx:
                            prompt_tokens = usage.get('prompt_tokens', 0)
                            completion_tokens = usage.get('completion_tokens', 0)
                            total_tokens = usage.get('total_tokens', 0)
                            break
                
                metadata = {
                    "model": "Qwen/Qwen3-32B",
                    "degraded": True,
                    "iteration": completion_idx,
                    "degradation_rank": completion_idx,
                    "degradation_reasoning": reasoning,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                branch_type = "degraded"
            
            # Create response part
            response_part = create_schema_compliant_part(
                part_type="response",
                content=completion,
                metadata=metadata
            )
            
            # Create assistant message
            assistant_message = {
                "role": "assistant",
                "parts": [response_part]
            }
            
            # Create conversation branch with tracking metadata
            branch = {
                "messages": [assistant_message]
            }
            
            # Store branch with metadata for randomization
            branches_with_metadata.append({
                "branch": branch,
                "original_rank": completion_idx,  # Track original preference order
                "branch_type": branch_type,
                "metadata": metadata
            })
        
        # Randomize the order of branches using sample-specific seed
        # This ensures consistent randomization per sample but different orders across samples
        sample_seed = hash(sample_state.conversation_id) % (2**32)
        rng = random.Random(sample_seed)
        rng.shuffle(branches_with_metadata)
        
        # Extract just the branches in randomized order
        conversation_branches = [item["branch"] for item in branches_with_metadata]
        
        # Create the dataset sample
        dataset_sample = {
            "conversation_id": sample_state.conversation_id,
            "dataset_source": original_sample.get("dataset_source", "synthetic_degradation"),
            "original_metadata": original_sample.get("original_metadata", {}),
            "system_prompt": original_sample.get("system_prompt", {}),
            "initial_prompt": original_sample["initial_prompt"],
            "available_functions": original_sample.get("available_functions", []),
            "conversation_branches": conversation_branches,
            "created_timestamp": datetime.now().isoformat(),
            "synthetic_metadata": {
                "original_conversation_id": sample_state.conversation_id,
                "num_degradations": len(sample_state.completions_history) - 1,
                "generation_timestamp": datetime.now().isoformat(),
                "degradation_model": "Qwen/Qwen3-32B",
                "randomization_seed": sample_seed
            }
        }
        
        dataset_samples.append(dataset_sample)
    
    return dataset_samples


async def main():
    parser = argparse.ArgumentParser(description="Generate synthetic degraded completions for preference learning")
    parser.add_argument("input_dataset", help="Path to input dataset directory")
    parser.add_argument("--samples", "-n", type=int, default=100,
                       help="Number of samples to process (default: 100)")
    parser.add_argument("--iterations", "-i", type=int, default=5,
                       help="Number of degradation iterations (default: 5)")
    parser.add_argument("--output", "-o", default=None,
                       help="Output dataset path (auto-generated if not specified)")
    parser.add_argument("--concurrent", "-c", type=int, default=DEFAULT_CONCURRENT,
                       help="Maximum concurrent API requests (default: 50)")
    parser.add_argument("--disable-adaptive", action="store_true",
                       help="Disable adaptive concurrency")
    parser.add_argument("--target-model", default="Qwen/Qwen3-32B",
                       help="Target model to extract original completions from")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing state (error if no state exists)")
    parser.add_argument("--restart", action="store_true",
                       help="Delete existing state and start fresh")
    
    args = parser.parse_args()
    
    if args.resume and args.restart:
        print("Error: Cannot specify both --resume and --restart")
        return 1
    
    # Get API key from environment
    api_key = os.getenv("SWISSAI_API_KEY")
    if not api_key:
        print("Error: SWISSAI_API_KEY environment variable required")
        return 1
    
    # Determine output path
    if args.output is None:
        input_path = Path(args.input_dataset)
        output_path = input_path.parent / f"{input_path.name}_synthetic_completions"
    else:
        output_path = Path(args.output)
    
    # Initialize state manager
    state_manager = StateManager(output_path)
    
    # Handle existing state
    if state_manager.state_exists():
        if args.restart:
            print("Existing state found. Removing and starting fresh...")
            state_manager.cleanup_state()
            resuming = False
        elif args.resume:
            print("Existing state found. Resuming...")
            resuming = True
        else:
            print(f"Error: Existing state found at {state_manager.state_dir}")
            print("Use --resume to continue or --restart to start fresh")
            return 1
    else:
        if args.resume:
            print("Error: No existing state found to resume from")
            return 1
        resuming = False
    
    if resuming:
        # Load existing state
        print("Loading existing state...")
        sample_states, saved_config, progress = state_manager.load_state()
        
        # Validate configuration compatibility
        current_config = {
            "input_dataset": args.input_dataset,
            "samples": args.samples,
            "iterations": args.iterations,
            "target_model": args.target_model
        }
        
        for key, value in current_config.items():
            if saved_config.get(key) != value:
                print(f"Error: Configuration mismatch for {key}: saved={saved_config.get(key)}, current={value}")
                print("Use --restart to start with new configuration")
                return 1
        
        start_iteration = progress["completed_iteration"] + 1
        if start_iteration > args.iterations:
            print("All iterations already completed!")
            start_iteration = args.iterations + 1  # Will skip generation
        else:
            print(f"Resuming from iteration {start_iteration}/{args.iterations}")
            print(f"Progress: {len(sample_states)} samples, completed {progress['completed_iteration']} iterations")
    
    else:
        # Fresh start
        print(f"Loading dataset from: {args.input_dataset}")
        dataset = load_from_disk(args.input_dataset)
        
        # Initialize sample states
        print(f"Initializing {args.samples} samples...")
        sample_states = initialize_sample_states(dataset, args.samples, args.target_model)
        
        if not sample_states:
            print("Error: No valid samples found in dataset")
            return 1
        
        print(f"Successfully initialized {len(sample_states)} samples")
        
        # Save initial state
        config = {
            "input_dataset": args.input_dataset,
            "samples": args.samples,
            "iterations": args.iterations,
            "target_model": args.target_model,
            "concurrent": args.concurrent,
            "disable_adaptive": args.disable_adaptive,
            "start_time": datetime.now().isoformat()
        }
        state_manager.save_initial_state(sample_states, config)
        print(f"Initial state saved to {state_manager.state_dir}")
        start_iteration = 1
    
    # Generate degradations (or skip if already complete)
    if start_iteration <= args.iterations:
        generator = SyntheticCompletionsGenerator(
            api_key=api_key,
            concurrent=args.concurrent,
            disable_adaptive=args.disable_adaptive,
            state_manager=state_manager
        )
        
        print(f"\nStarting degradation generation...")
        processed_states = await generator.generate_degradations(sample_states, args.iterations, start_iteration)
    else:
        print("All iterations already completed, proceeding to dataset creation...")
        processed_states = sample_states
    
    # Convert to dataset format
    print(f"\nConverting to dataset format...")
    dataset_samples = convert_to_dataset_format(processed_states)
    
    # Create new dataset
    print(f"Creating HuggingFace dataset...")
    synthetic_dataset = Dataset.from_list(dataset_samples)
    synthetic_dataset_dict = DatasetDict({'train': synthetic_dataset})
    
    # Save dataset
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to: {output_path}")
    synthetic_dataset_dict.save_to_disk(output_path)
    
    # Clean up state files after successful completion
    print("Cleaning up state files...")
    state_manager.cleanup_state()
    
    # Summary statistics
    total_completions = sum(len(ss.completions_history) for ss in processed_states)
    successful_samples = len([ss for ss in processed_states if len(ss.completions_history) > 1])
    
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Input samples: {len(sample_states)}")
    print(f"Successful samples: {successful_samples}")
    print(f"Total completions generated: {total_completions}")
    print(f"Iterations per sample: {args.iterations}")
    print(f"Output dataset: {output_path}")
    print(f"Dataset size: {len(dataset_samples)} samples")
    
    # Show degradation statistics
    iteration_counts = {}
    for ss in processed_states:
        num_iterations = len(ss.completions_history) - 1  # -1 for original
        iteration_counts[num_iterations] = iteration_counts.get(num_iterations, 0) + 1
    
    print(f"\nDegradation completion statistics:")
    for iterations, count in sorted(iteration_counts.items()):
        print(f"  {iterations} iterations: {count} samples")
    
    # Token usage statistics
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_requests = 0
    
    for ss in processed_states:
        if hasattr(ss, 'token_usage') and ss.token_usage:
            for usage in ss.token_usage:
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                total_requests += 1
    
    if total_requests > 0:
        print(f"\nToken usage statistics:")
        print(f"  Total API requests: {total_requests}")
        print(f"  Total prompt tokens: {total_prompt_tokens:,}")
        print(f"  Total completion tokens: {total_completion_tokens:,}")
        print(f"  Total tokens: {total_prompt_tokens + total_completion_tokens:,}")
        print(f"  Average prompt tokens per request: {total_prompt_tokens / total_requests:.1f}")
        print(f"  Average completion tokens per request: {total_completion_tokens / total_requests:.1f}")
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())