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
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque

import openai
from tqdm import tqdm


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
    
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.3-70B-Instruct"):
        """
        Initialize the classifier.
        
        Args:
            api_key: Swiss AI API key
            model: Model name to use for classification
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
        
        # Adaptive concurrency settings
        self.adaptive_enabled = False
        self.current_concurrent = 50  # Will be set by classify_batch
        self.min_concurrent = 1
        self.max_concurrent = 200
        self.last_adaptation = time.time()
        self.adaptation_interval = 60  # Adapt every 60 seconds
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
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
    
    def enable_adaptive_concurrency(self, min_concurrent: int = 1, max_concurrent: int = 200):
        """Enable adaptive concurrency adjustment."""
        self.adaptive_enabled = True
        self.min_concurrent = min_concurrent
        self.max_concurrent = max_concurrent
        
    def _adapt_concurrency(self, current_concurrent: int) -> int:
        """Adapt concurrency based on current performance metrics."""
        if not self.adaptive_enabled:
            return current_concurrent
            
        now = time.time()
        if now - self.last_adaptation < self.adaptation_interval:
            return current_concurrent  # Too soon to adapt
            
        metrics = self.metrics.get_adaptive_metrics()
        error_rate = metrics['failed_requests'] / max(metrics['total_requests'], 1)
        
        # Immediate failure response (reduce by 1 on any recent failure)
        recent_failures = sum(1 for ts, success, _ in self.metrics.requests 
                            if not success and now - ts < 10)  # Last 10 seconds
        if recent_failures > 0:
            new_concurrent = max(current_concurrent - 1, self.min_concurrent)
            if new_concurrent < current_concurrent:
                print(f"🔻 Reducing concurrency: {current_concurrent} → {new_concurrent} (recent failures)")
            self.last_adaptation = now
            return new_concurrent
        
        # Periodic optimization based on error rate
        if error_rate < 0.01:  # Less than 1% errors
            # Explore higher concurrency (+20)
            new_concurrent = min(current_concurrent + 20, self.max_concurrent)
            if new_concurrent > current_concurrent:
                print(f"🔺 Increasing concurrency: {current_concurrent} → {new_concurrent} (low error rate)")
        else:
            # Fall back to calculated optimal
            optimal = max(metrics['optimal_concurrent'], self.min_concurrent)
            new_concurrent = min(optimal, current_concurrent)  # Don't increase if errors
            if new_concurrent < current_concurrent:
                print(f"🔻 Reducing to optimal: {current_concurrent} → {new_concurrent} (error rate: {error_rate*100:.1f}%)")
        
        self.last_adaptation = now
        return new_concurrent

    async def classify_single(self, question: str, answer: str, prompt_template: str, 
                            valid_categories: List[str]) -> ClassificationResult:
        """
        Classify a single question-answer pair.
        
        Args:
            question: The question/context
            answer: The answer to classify
            prompt_template: Template with {question} and {answer} placeholders
            valid_categories: List of valid classification categories
            
        Returns:
            ClassificationResult with classification and metadata
        """
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
                    result = ClassificationResult(
                        classification=valid_categories[-1],  # Default to last category
                        reasoning=f"Invalid classification '{classification}', defaulting to inconclusive",
                        success=False,
                        error=f"Invalid classification: {classification}",
                        tokens_used=tokens_used
                    )
                    
                    # Record the request result
                    self.metrics.record_request(result.success)
                    return result
                
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
                self.logger.error(f"Failed to parse JSON response: {e}")
                self.logger.error(f"Raw response: {response_content}")
                
                result = ClassificationResult(
                    classification=valid_categories[-1],
                    reasoning=f"Failed to parse response: {str(e)}",
                    success=False,
                    error=f"JSON parse error: {str(e)}",
                    tokens_used=tokens_used
                )
                
                # Record the request result with latency
                latency = time.time() - start_time
                self.metrics.record_request(result.success, latency)
                return result
                
        except Exception as e:
            self.logger.error(f"Classification request failed: {e}")
            result = ClassificationResult(
                classification=valid_categories[-1] if valid_categories else "error",
                reasoning=f"API request failed: {str(e)}",
                success=False,
                error=str(e),
                tokens_used=0
            )
            
            # Record the request result
            self.metrics.record_request(result.success)
            return result
    
    async def classify_batch(self, items: List[Dict[str, str]], prompt_template: str,
                           valid_categories: List[str], concurrent: int = 50) -> List[ClassificationResult]:
        """
        Classify multiple items concurrently with optional adaptive adjustment.
        
        Args:
            items: List of dicts with 'question' and 'answer' keys
            prompt_template: Template with {question} and {answer} placeholders
            valid_categories: List of valid classification categories
            concurrent: Starting number of concurrent requests
            
        Returns:
            List of ClassificationResult objects
        """
        # Initialize adaptive settings
        self.current_concurrent = concurrent
        
        if self.adaptive_enabled:
            return await self._classify_batch_adaptive(items, prompt_template, valid_categories, concurrent)
        else:
            return await self._classify_batch_fixed(items, prompt_template, valid_categories, concurrent)
    
    async def _classify_batch_fixed(self, items: List[Dict[str, str]], prompt_template: str,
                                  valid_categories: List[str], concurrent: int) -> List[ClassificationResult]:
        """Fixed concurrency batch processing."""
        semaphore = asyncio.Semaphore(concurrent)
        
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
        with tqdm(total=len(tasks), desc="Classifying") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
                
                # Display metrics periodically
                current_time = time.time()
                if current_time - self.last_metrics_display >= self.metrics_display_interval:
                    metrics = self.metrics.get_adaptive_metrics() if self.adaptive_enabled else self.metrics.get_metrics()
                    postfix = {
                        'req/min': f"{metrics['requests_per_minute']:.1f}",
                        'success': f"{metrics['success_rate']*100:.1f}%",
                        'failed': metrics['failed_requests']
                    }
                    if self.adaptive_enabled and 'optimal_concurrent' in metrics:
                        postfix['concurrent'] = self.current_concurrent
                        postfix['optimal'] = metrics['optimal_concurrent']
                    pbar.set_postfix(postfix)
                    self.last_metrics_display = current_time
        
        return results
        
    async def _classify_batch_adaptive(self, items: List[Dict[str, str]], prompt_template: str,
                                     valid_categories: List[str], initial_concurrent: int) -> List[ClassificationResult]:
        """Adaptive concurrency batch processing with time-based adaptation."""
        # For now, just use fixed processing but with potential for future dynamic semaphore
        # The adaptation logic is based on time (60 seconds) not batch completion
        self.current_concurrent = self._adapt_concurrency(initial_concurrent)
        return await self._classify_batch_fixed(
            items, prompt_template, valid_categories, self.current_concurrent
        )


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