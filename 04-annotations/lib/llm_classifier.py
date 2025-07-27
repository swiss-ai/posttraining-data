#!/usr/bin/env python3
"""
LLM Classification Library

Provides a simple interface for LLM-based classification using Swiss AI API.
Handles batch processing, error handling, and token tracking.
"""

import os
import json
import time
import asyncio
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

import openai
from tqdm import tqdm

# Configuration constants
MAX_RETRY_ATTEMPTS = 3


@dataclass
class ClassificationResult:
    """Result of a single classification request."""
    classification: str
    reasoning: str
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class RequestMetrics:
    """Track rolling window metrics for API requests."""
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        # Store (timestamp, success_bool, latency_seconds) tuples
        self.requests = deque()
        # Separate deque for latency tracking (for average calculation)
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
                'requests_per_minute': 0.0
            }
        
        total = len(self.requests)
        successful = sum(1 for _, success, _ in self.requests if success)
        failed = total - successful
        
        # Calculate time span of actual data
        oldest_time = self.requests[0][0]
        newest_time = self.requests[-1][0]
        time_span = max(newest_time - oldest_time, 1.0)  # Avoid division by zero
        
        return {
            'total_requests': total,
            'successful_requests': successful,
            'failed_requests': failed,
            'success_rate': successful / total if total > 0 else 0.0,
            'requests_per_minute': (total * 60) / time_span
        }
    
    def get_adaptive_metrics(self) -> dict:
        """Get metrics with adaptive concurrency calculations."""
        metrics = self.get_metrics()
        
        # Calculate average latency
        if self.latencies:
            avg_latency = sum(self.latencies) / len(self.latencies)
            # Little's Law: optimal concurrency = throughput * latency
            optimal_concurrent = max(1, int((metrics['requests_per_minute'] * avg_latency) / 60.0))
        else:
            avg_latency = 0.0
            optimal_concurrent = 50  # Default fallback
            
        return {
            **metrics,
            'avg_latency_seconds': avg_latency,
            'optimal_concurrent': optimal_concurrent
        }


class LLMClassifier:
    """Base class for LLM-based classification using Swiss AI API."""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.3-70B-Instruct", 
                 concurrent: int = 50, adaptive: bool = True):
        """
        Initialize the classifier.
        
        Args:
            api_key: Swiss AI API key
            model: Model name to use for classification
            concurrent: Starting number of concurrent requests
            adaptive: Enable adaptive concurrency adjustment
        """
        self.api_key = api_key
        self.model = model
        
        # Initialize OpenAI client for Swiss AI
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.swissai.cscs.ch/v1"
        )
        
        # Token tracking
        self.total_tokens_used = 0
        
        # Metrics tracking
        self.metrics = RequestMetrics(window_seconds=60)
        self.last_metrics_display = time.time()
        self.metrics_display_interval = 10  # Show metrics every 10 seconds
        
        # Concurrency settings
        self.adaptive_enabled = adaptive
        self.current_concurrent = concurrent
        self.min_concurrent = 1
        self.last_adaptation = time.time()
        self.adaptation_interval = 60  # Adapt every 60 seconds
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    async def get_model_compute_nodes(self) -> int:
        """
        Get the number of compute nodes available for the current model.
        
        Returns:
            Number of compute nodes (duplicate model entries in /models endpoint)
        """
        try:
            # Query the models endpoint to count instances of our model
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Use sync requests in async context (this is a one-time setup call)
            response = requests.get(
                "https://api.swissai.cscs.ch/v1/models",
                headers=headers,
                timeout=10.0
            )
            
            if response.status_code == 200:
                models_data = response.json()
                # Count occurrences of our model ID
                model_count = sum(1 for model in models_data.get('data', []) 
                                if model.get('id') == self.model)
                return max(model_count, 1)  # At least 1 node
            else:
                self.logger.warning(f"Failed to query models endpoint: {response.status_code}")
                return 1  # Default fallback
                
        except Exception as e:
            self.logger.warning(f"Could not determine compute nodes: {e}")
            return 1  # Default fallback
    
    def _clean_json_response(self, response_content: str) -> str:
        """
        Clean up common JSON formatting issues in LLM responses.
        
        Args:
            response_content: Raw response content from LLM
            
        Returns:
            Cleaned JSON string
        """
        import re
        
        # Remove any text before the first '{'
        start_idx = response_content.find('{')
        if start_idx > 0:
            response_content = response_content[start_idx:]
        
        # Remove any text after the last '}'
        end_idx = response_content.rfind('}')
        if end_idx != -1 and end_idx < len(response_content) - 1:
            response_content = response_content[:end_idx + 1]
        
        # Fix common escape sequence issues
        # Replace invalid escape sequences with proper ones
        response_content = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', response_content)
        
        # Fix unescaped quotes in strings (basic heuristic)
        # This is tricky but we'll handle the most common case: quotes in reasoning text
        lines = response_content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # If line contains both a field name and unescaped quotes in the value
            if ':' in line and line.strip().startswith('"'):
                # Split on first colon to separate field from value
                parts = line.split(':', 1)
                if len(parts) == 2:
                    field_part = parts[0]
                    value_part = parts[1].strip()
                    
                    # If value part starts and ends with quotes, fix internal quotes
                    if value_part.startswith('"') and value_part.endswith('"') and len(value_part) > 2:
                        # Remove outer quotes, escape internal quotes, add back outer quotes
                        inner_value = value_part[1:-1]
                        # Escape any unescaped quotes
                        inner_value = re.sub(r'(?<!\\)"', r'\\"', inner_value)
                        value_part = f'"{inner_value}"'
                        line = f"{field_part}: {value_part}"
            
            fixed_lines.append(line)
        
        response_content = '\n'.join(fixed_lines)
        
        # Handle trailing commas before closing braces/brackets
        response_content = re.sub(r',(\s*[}\]])', r'\1', response_content)
        
        return response_content.strip()
    
    def load_prompt_template(self, template_path: Path) -> str:
        """Load prompt template from file."""
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    

    async def classify_single(self, question: str, answer: str, prompt_template: str, 
                            valid_categories: List[str]) -> ClassificationResult:
        """
        Classify a single question-answer pair with retry logic.
        
        Args:
            question: The question/context
            answer: The answer to classify
            prompt_template: Template with {question} and {answer} placeholders
            valid_categories: List of valid classification categories
            
        Returns:
            ClassificationResult with classification and metadata
        """
        last_error = None
        
        for attempt in range(MAX_RETRY_ATTEMPTS + 1):  # 0, 1, 2, 3 (total 4 attempts)
            # Add delay for retries (exponential backoff with jitter, up to ~80s)
            if attempt > 0:
                base_delay = min(2 ** (attempt + 1), 60)  # 4s, 8s, 16s, 32s, 60s, 60s...
                jitter = random.uniform(0, base_delay * 0.25)  # Add up to 25% jitter
                delay = base_delay + jitter
                self.logger.debug(f"Retrying request (attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS + 1}) after {delay:.1f}s delay")
                await asyncio.sleep(delay)
            
            start_time = time.time()
            try:
                # Format the prompt
                formatted_prompt = prompt_template.format(question=question, answer=answer)
                
                # Make API request
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistency
                    response_format={"type": "json_object"}
                )
                
                # Extract response
                response_content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 0
                self.total_tokens_used += tokens_used
                
                # Parse JSON response
                try:
                    # Clean up common JSON formatting issues from LLM responses
                    cleaned_content = self._clean_json_response(response_content)
                    result_data = json.loads(cleaned_content)
                    classification = result_data.get("classification", "").strip()
                    reasoning = result_data.get("reasoning", "").strip()
                    
                    # Validate classification
                    if classification not in valid_categories:
                        # Don't retry for invalid classifications - this is a logic error, not API error
                        last_error = f"Invalid classification: {classification}"
                        continue  # Retry with same request
                    
                    # Success!
                    result = ClassificationResult(
                        classification=classification,
                        reasoning=reasoning,
                        success=True,
                        tokens_used=tokens_used
                    )
                    
                    # Record the request result with latency
                    latency = time.time() - start_time
                    self.metrics.record_request(result.success, latency)
                    return result
                    
                except json.JSONDecodeError as e:
                    # JSON parse errors should be retried
                    last_error = f"JSON parse error: {str(e)}"
                    self.logger.debug(f"JSON parse failed on attempt {attempt + 1}: {e}")
                    continue  # Try again
                    
            except Exception as e:
                # Network/API errors should be retried
                last_error = f"API request failed: {str(e)}"
                self.logger.debug(f"API request failed on attempt {attempt + 1}: {e}")
                continue  # Try again
        
        # All retries exhausted - return failure
        self.logger.error(f"Classification failed after {MAX_RETRY_ATTEMPTS + 1} attempts. Last error: {last_error}")
        result = ClassificationResult(
            classification=valid_categories[-1] if valid_categories else "error",
            reasoning=f"Failed after {MAX_RETRY_ATTEMPTS + 1} attempts: {last_error}",
            success=False,
            error=last_error,
            tokens_used=0
        )
        
        # Record the final failure
        self.metrics.record_request(result.success)
        return result
    
    async def classify_batch(self, items: List[Dict[str, str]], prompt_template: str,
                           valid_categories: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple items concurrently with optional adaptive adjustment.
        
        Args:
            items: List of dicts with 'question' and 'answer' keys
            prompt_template: Template with {question} and {answer} placeholders
            valid_categories: List of valid classification categories
            
        Returns:
            List of ClassificationResult objects
        """
        if self.adaptive_enabled:
            return await self._classify_batch_adaptive(items, prompt_template, valid_categories)
        else:
            return await self._classify_batch_fixed(items, prompt_template, valid_categories)
    
    async def _classify_batch_fixed(self, items: List[Dict[str, str]], prompt_template: str,
                                  valid_categories: List[str]) -> List[ClassificationResult]:
        """Fixed concurrency batch processing."""
        semaphore = asyncio.Semaphore(self.current_concurrent)
        
        async def classify_with_semaphore(item):
            async with semaphore:
                return await self.classify_single(
                    item["question"], 
                    item["answer"], 
                    prompt_template, 
                    valid_categories
                )
        
        # Create tasks for all items
        tasks = [classify_with_semaphore(item) for item in items]
        
        # Execute with progress tracking and metrics display
        results = []
        with tqdm(total=len(tasks), desc="Classifying", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]') as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
                
                # Display metrics periodically
                current_time = time.time()
                if current_time - self.last_metrics_display >= self.metrics_display_interval:
                    metrics = self.metrics.get_adaptive_metrics()
                    error_rate = metrics['failed_requests'] / max(metrics['total_requests'], 1)
                    postfix = {
                        'req/min': f"{metrics['requests_per_minute']:.1f}",
                        'success': f"{metrics['success_rate']*100:.1f}%",
                        'failed': metrics['failed_requests'],
                        'avg_dur': f"{metrics.get('avg_latency_seconds', 0):.1f}s",
                        'error_rate': f"{error_rate*100:.1f}%",
                        'concurrent': self.current_concurrent
                    }
                    pbar.set_postfix(postfix)
                    self.last_metrics_display = current_time
        
        return results
        
    async def _classify_batch_adaptive(self, items: List[Dict[str, str]], prompt_template: str,
                                     valid_categories: List[str]) -> List[ClassificationResult]:
        """Adaptive concurrency batch processing using task set control loop pattern."""
        # Reset adaptation timer for new chunk to allow stabilization
        self.last_adaptation = time.time()
        
        in_flight_tasks = set()
        work_queue = list(enumerate(items))  # Keep track of original indices
        results = [None] * len(items)  # Pre-allocate results list
        
        # Progress tracking
        completed_count = 0
        pbar = tqdm(total=len(items), desc="Classifying",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
        
        while work_queue or in_flight_tasks:
            # Start new tasks up to current limit
            while len(in_flight_tasks) < self.current_concurrent and work_queue:
                idx, item = work_queue.pop(0)
                
                # Create task with index tracking
                task = asyncio.create_task(self._classify_with_index(
                    idx, item["question"], item["answer"], 
                    prompt_template, valid_categories
                ))
                in_flight_tasks.add(task)
            
            # Wait for first completion or timeout
            if in_flight_tasks:
                done, in_flight_tasks = await asyncio.wait(
                    in_flight_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0  # Check every second
                )
                
                # Process completed tasks
                for task in done:
                    idx, result = await task
                    results[idx] = result
                    completed_count += 1
                    pbar.update(1)
                    
                    # Handle failures immediately
                    if not result.success:
                        await self._handle_failure_immediate()
                
                # Display metrics periodically
                current_time = time.time()
                if current_time - self.last_metrics_display >= self.metrics_display_interval:
                    metrics = self.metrics.get_adaptive_metrics()
                    error_rate = metrics['failed_requests'] / max(metrics['total_requests'], 1)
                    postfix = {
                        'req/min': f"{metrics['requests_per_minute']:.1f}",
                        'success': f"{metrics['success_rate']*100:.1f}%",
                        'failed': metrics['failed_requests'],
                        'avg_dur': f"{metrics.get('avg_latency_seconds', 0):.1f}s",
                        'error_rate': f"{error_rate*100:.1f}%",
                        'concurrent': self.current_concurrent
                    }
                    pbar.set_postfix(postfix)
                    self.last_metrics_display = current_time
            
            # Check for periodic adjustment (every 60 seconds)
            self._check_concurrency_adjustment()
        
        pbar.close()
        return results
    
    async def _classify_with_index(self, idx: int, question: str, answer: str, 
                                   prompt_template: str, valid_categories: List[str]) -> Tuple[int, ClassificationResult]:
        """Classify single item and return with its index."""
        result = await self.classify_single(question, answer, prompt_template, valid_categories)
        return idx, result
    
    async def _handle_failure_immediate(self):
        """Immediately reduce concurrency on any failure."""
        old_concurrent = self.current_concurrent
        self.current_concurrent = max(self.current_concurrent - 1, self.min_concurrent)
        
        if self.current_concurrent < old_concurrent:
            print(f"🔻 Reducing concurrency: {old_concurrent} → {self.current_concurrent} (immediate failure response)")
    
    def _check_concurrency_adjustment(self):
        """Check if 60s have passed and adjust concurrency based on metrics."""
        if not self.adaptive_enabled:
            return
            
        now = time.time()
        if now - self.last_adaptation < self.adaptation_interval:
            return  # Too soon to adapt
            
        metrics = self.metrics.get_adaptive_metrics()
        error_rate = metrics['failed_requests'] / max(metrics['total_requests'], 1)
        old_concurrent = self.current_concurrent
        
        # Periodic optimization based on error rate
        if error_rate < 0.01:  # Less than 1% errors
            # Explore higher concurrency (+20)
            self.current_concurrent += 20
            print(f"🔺 Increasing concurrency: {old_concurrent} → {self.current_concurrent} (low error rate)")
        else:
            # Fall back to calculated optimal
            optimal = max(metrics['optimal_concurrent'], self.min_concurrent)
            self.current_concurrent = min(optimal, self.current_concurrent)  # Don't increase if errors
            if self.current_concurrent < old_concurrent:
                print(f"🔻 Reducing to optimal: {old_concurrent} → {self.current_concurrent} (error rate: {error_rate*100:.1f}%)")
        
        self.last_adaptation = now


def extract_message_context(message: Dict[str, Any], conversation_messages: List[Dict[str, Any]]) -> str:
    """
    Extract context for a message from the conversation.
    
    Args:
        message: The message to classify
        conversation_messages: List of all messages in conversation order
        
    Returns:
        Context string (usually the previous user message)
    """
    if message.get("role") == "assistant":
        # For assistant messages, find the most recent user message
        message_idx = None
        for i, msg in enumerate(conversation_messages):
            if msg.get("content") == message.get("content") and msg.get("role") == message.get("role"):
                message_idx = i
                break
        
        if message_idx is not None:
            # Look backwards for user message
            for i in range(message_idx - 1, -1, -1):
                if conversation_messages[i].get("role") == "user":
                    return conversation_messages[i].get("content", "")
    
    return ""


def should_classify_message(message: Dict[str, Any], target_roles: List[str]) -> bool:
    """
    Determine if a message should be classified based on its role.
    
    Args:
        message: Message dictionary with 'role' field
        target_roles: List of roles to classify (e.g., ['assistant'])
        
    Returns:
        True if message should be classified
    """
    return message.get("role", "") in target_roles