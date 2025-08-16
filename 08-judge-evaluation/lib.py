#!/usr/bin/env python3
"""
Judge Evaluation Library

Composable utilities for judge evaluation scripts. Provides reusable components
for dataset loading, LLM requests, concurrent processing, metrics calculation,
and report generation without forcing inheritance patterns.
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, kendalltau
from datasets import load_from_disk
from tqdm import tqdm
import openai


class SyntheticDatasetLoader:
    """Handles loading and preparation of synthetic preference datasets."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
    
    def load_dataset(self, num_samples: int, iterations_range: Tuple[int, int] = (0, 8)) -> List[Dict]:
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
            
            # Extract completions in natural branch order (preserve original order)
            completions = []
            ground_truth = []
            
            for branch in sample['conversation_branches']:
                if branch['messages'] and 'parts' in branch['messages'][0]:
                    part = branch['messages'][0]['parts'][0]
                    if 'metadata' in part:
                        iteration = part['metadata'].get('iteration', -1)
                        if iterations_range[0] <= iteration <= iterations_range[1]:
                            degradation_rank = part['metadata'].get('degradation_rank', iteration)
                            
                            completion_data = {
                                'content': part['content'],
                                'iteration': iteration,
                                'degradation_rank': degradation_rank
                            }
                            
                            completions.append(completion_data)
                            # Convert 0-based degradation_rank to 1-based quality rank
                            ground_truth.append(degradation_rank + 1)
            
            # Ensure we have all expected iterations with valid content
            expected_count = iterations_range[1] - iterations_range[0] + 1
            if len(completions) == expected_count:
                # Check for None content in any completion (should be filtered in v2 dataset)
                has_none_content = any(comp['content'] is None for comp in completions)
                if not has_none_content:
                    samples.append({
                        'id': f"{i:04d}",
                        'conversation_id': sample['conversation_id'],
                        'question': sample['initial_prompt']['content'],
                        'completions': completions,  # List in natural branch order
                        'ground_truth': ground_truth  # Quality ranks (1-9) matching completion order
                    })
        
        return samples


class InstructionsLoader:
    """Handles loading and validation of judge instructions."""
    
    @staticmethod
    def load_instructions(instructions_path: str, script_dir: str = None) -> str:
        """Load judge instructions from file."""
        try:
            # If path is relative, make it relative to script directory
            if not os.path.isabs(instructions_path) and script_dir:
                instructions_path = os.path.join(script_dir, instructions_path)
                
            with open(instructions_path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Instructions file not found: {instructions_path}")
        except Exception as e:
            raise ValueError(f"Error loading instructions from {instructions_path}: {str(e)}")


class LLMClient:
    """Centralized LLM request handling with retry logic and configuration."""
    
    def __init__(self, model: str = "Qwen/Qwen3-32B", max_retries: int = 2, debug_file_path: Optional[Path] = None):
        self.model = model
        self.max_retries = max_retries
        self.debug_file_path = debug_file_path
        self.request_counter = 0  # For numbering debug entries
        
        # Setup API client
        self.api_key = os.getenv("SWISSAI_API_KEY")
        if not self.api_key:
            raise ValueError("Please set SWISSAI_API_KEY environment variable")
        
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url="http://148.187.108.173:8092/v1"
        )
        
        # Initialize debug file if provided
        if self.debug_file_path:
            # Clear the debug file at start
            with open(self.debug_file_path, 'w') as f:
                f.write(f"=== DEBUG LOG: {model} ===\n")
                f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def _log_debug(self, request_num: int, messages: List[Dict], temperature: float, max_tokens: int, 
                   response_content: str, tokens: Dict, success: bool, retries: int = 0):
        """Log request and response to debug file."""
        if not self.debug_file_path:
            return
            
        try:
            with open(self.debug_file_path, 'a') as f:
                f.write(f"=== LLM REQUEST {request_num} ===\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Temperature: {temperature}\n")
                f.write(f"Max Tokens: {max_tokens}\n\n")
                
                # Log prompt with clear markers
                f.write("=== PROMPT START ===\n")
                for msg in messages:
                    if msg['role'] == 'user':
                        f.write(msg['content'])
                        f.write('\n')
                f.write("=== PROMPT END ===\n\n")
                
                f.write("=== RESPONSE START ===\n")
                f.write(response_content or "[NO RESPONSE]")
                f.write('\n=== RESPONSE END ===\n\n')
                
                f.write(f"TOKENS: prompt={tokens['prompt']}, completion={tokens['completion']}, total={tokens['total']}\n")
                f.write(f"SUCCESS: {success}\n")
                f.write(f"RETRIES: {retries}\n\n")
                
        except Exception as e:
            # Don't let debug logging break the evaluation
            pass
    
    async def make_request(self, messages: List[Dict], temperature: float = 0.0, 
                          max_tokens: int = 10000, openai_kwargs: Optional[Dict] = {}) -> Dict:
        """
        Make an LLM request with retry logic.
        
        Returns:
            Dict with keys: 'success', 'content', 'tokens', 'error', 'retries'
        """
        self.request_counter += 1
        request_num = self.request_counter
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **openai_kwargs,
                )

                result = {
                    'success': True,
                    'content': response.choices[0].message.content,
                    'logprobs': response.choices[0].logprobs.content,
                    'tokens': {
                        'prompt': response.usage.prompt_tokens,
                        'completion': response.usage.completion_tokens,
                        'total': response.usage.total_tokens
                    },
                    'error': None
                }
                
                if attempt > 0:
                    result['retries'] = attempt
                
                # Log debug info for successful request
                self._log_debug(request_num, messages, temperature, max_tokens,
                               result['content'], result['tokens'], True, attempt)
                    
                return result
                
            except Exception as e:
                error_msg = str(e)
                
                if attempt == self.max_retries:
                    result = {
                        'success': False,
                        'content': None,
                        'logprobs': None,
                        'tokens': {'prompt': 0, 'completion': 0, 'total': 0},
                        'error': error_msg,
                        'retries': attempt
                    }
                    
                    # Log debug info for failed request
                    self._log_debug(request_num, messages, temperature, max_tokens,
                                   None, result['tokens'], False, attempt)
                    
                    return result
                else:
                    # Print retry warning
                    print(f"⚠️  API request failed (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                    print(f"   Retrying in {0.1 * (attempt + 1):.1f}s...")
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
        
        # Should never reach here
        result = {
            'success': False,
            'content': None,
            'tokens': {'prompt': 0, 'completion': 0, 'total': 0},
            'error': 'Unknown error',
            'retries': self.max_retries
        }
        
        # Log debug info for unknown error
        self._log_debug(request_num, messages, temperature, max_tokens,
                       None, result['tokens'], False, self.max_retries)
        
        return result


class ConcurrentEvaluator:
    """Handles concurrent evaluation of samples with progress tracking."""
    
    def __init__(self, max_concurrent: int = 50):
        self.max_concurrent = max_concurrent
    
    async def evaluate_all_concurrent(self, samples: List[Dict], 
                                    evaluate_fn: Callable, 
                                    max_concurrent: Optional[int] = None) -> List[Dict]:
        """
        Evaluate all samples concurrently using the provided evaluation function.
        
        Args:
            samples: List of sample dictionaries
            evaluate_fn: Async function that takes a sample and returns a result dict
            max_concurrent: Override default concurrency limit
            
        Returns:
            List of result dictionaries sorted by sample_id
        """
        concurrent_limit = max_concurrent or self.max_concurrent
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def evaluate_with_semaphore(sample):
            async with semaphore:
                return await evaluate_fn(sample)
        
        # Create tasks for all samples
        tasks = [evaluate_with_semaphore(sample) for sample in samples]
        
        # Process with progress bar
        results = []
        with tqdm(total=len(tasks), desc="Processing samples") as pbar:
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                pbar.update(1)
        
        # Sort by sample ID to maintain order
        results.sort(key=lambda x: x['sample_id'])
        return results


class EvaluationAnalyzer:
    """Analyzes evaluation results and calculates metrics."""
    
    def calculate_overall_metrics(self, results: List[Dict]) -> Dict:
        """Calculate overall metrics from results."""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        metrics = {
            "total_samples": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "failed_rate": len(failed) / len(results) if results else 0
        }
        
        # Track retry statistics
        retried_samples = [r for r in results if 'retries' in r and r['retries'] > 0]
        if retried_samples:
            metrics["samples_retried"] = len(retried_samples)
            metrics["total_retries"] = sum(r['retries'] for r in retried_samples)
            metrics["avg_retries"] = metrics["total_retries"] / len(retried_samples)
        else:
            metrics["samples_retried"] = 0
            metrics["total_retries"] = 0
            metrics["avg_retries"] = 0
        
        if successful:
            # Correlation metrics (ranking-specific, but kept for compatibility)
            if 'spearman' in successful[0]:  # Check if ranking metrics exist
                spearmans = [r['spearman'] for r in successful]
                kendalls = [r['kendall'] for r in successful]
                top3_accs = [r['top3_correct'] / 3 for r in successful]
                
                metrics.update({
                    "mean_spearman": np.mean(spearmans),
                    "std_spearman": np.std(spearmans),
                    "median_spearman": np.median(spearmans),
                    "mean_kendall": np.mean(kendalls),
                    "std_kendall": np.std(kendalls),
                    "median_kendall": np.median(kendalls),
                    "mean_top3_accuracy": np.mean(top3_accs),
                    "perfect_rankings": sum(1 for r in successful if r['spearman'] == 1.0)
                })
                
                # Position accuracy (ranking-specific)
                if 'predicted' in successful[0]:
                    position_accuracies = []
                    for pos in range(9):
                        correct = sum(1 for r in successful if r['predicted'][pos] == r['ground_truth'][pos])
                        position_accuracies.append(correct / len(successful))
                    
                    metrics['position_accuracies'] = position_accuracies
                    metrics['top1_accuracy'] = sum(1 for r in successful if r['predicted'][0] == 1) / len(successful)
                    metrics['bottom1_accuracy'] = sum(1 for r in successful if r['predicted'][8] == 9) / len(successful)
            
            # Token usage
            total_tokens = sum(r['tokens']['total'] for r in results)
            metrics.update({
                "total_tokens": total_tokens,
                "avg_tokens_per_sample": total_tokens / len(results) if results else 0,
                "avg_prompt_tokens": np.mean([r['tokens']['prompt'] for r in results]),
                "avg_completion_tokens": np.mean([r['tokens']['completion'] for r in results])
            })
        
        # Error analysis
        if failed:
            error_types = defaultdict(int)
            error_details = defaultdict(list)
            for r in failed:
                # Use error_detail if available, otherwise fall back to error
                detail = r.get('error_detail', r.get('error', 'Unknown'))
                
                # Categorize errors (ranking-specific categories, but extensible)
                if "No RANKING:" in detail:
                    error_types['no_ranking_found'] += 1
                elif "Wrong number of elements" in detail:
                    error_types['wrong_count'] += 1
                elif "Duplicate values" in detail:
                    error_types['duplicate_values'] += 1
                elif "Missing values" in detail:
                    error_types['missing_values'] += 1
                elif "Values outside 1-9 range" in detail:
                    error_types['invalid_range'] += 1
                elif "Non-numeric values" in detail:
                    error_types['non_numeric'] += 1
                elif "Invalid alphabetic labels" in detail:
                    error_types['invalid_labels'] += 1
                elif "API" in r.get('error', ''):
                    error_types['api_error'] += 1
                elif "Correlation undefined" in detail:
                    error_types["correlation_undefined"] += 1
                elif "No valid score" in detail:
                    error_types['no_score_output'] += 1
                else:
                    error_types['other'] += 1
                
                # Store detailed error for report
                error_details[detail].append(r['sample_id'])
            
            metrics['error_types'] = dict(error_types)
            metrics['error_details'] = dict(error_details)
        
        return metrics


class ReportGenerator:
    """Generates comprehensive evaluation reports."""
    
    def generate_report(self, results: List[Dict], metrics: Dict, output_path: str, config: Dict):
        """Generate markdown report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = []
        report.append("# Judge LLM Ranking Evaluation Report\n")
        report.append("## Summary\n")
        report.append(f"Model: {config['model']}")
        
        # Add method-specific config
        if 'label_type' in config:
            report.append(f"Label Type: {config['label_type']}")
        if 'instructions' in config:
            report.append(f"Instructions: {config['instructions']}")
            
        report.append(f"Samples Evaluated: {metrics['total_samples']}")
        report.append(f"Timestamp: {timestamp}\n")
        
        self._add_overall_metrics(report, metrics)
        self._add_detailed_results(report, results, metrics)
        self._add_error_analysis(report, results, metrics)
        
        # Add method-specific analysis if ranking metrics exist
        if metrics.get('successful', 0) > 0 and 'mean_spearman' in metrics:
            self._add_statistical_distributions(report, results, metrics)
        
        # Write report
        with open(output_path, 'w') as f:
            f.write("\n".join(report))
    
    def _add_overall_metrics(self, report: List[str], metrics: Dict):
        """Add overall metrics section."""
        report.append("### Overall Metrics")
        report.append(f"- Success Rate: {metrics['success_rate']:.1%} ({metrics['successful']}/{metrics['total_samples']})")
        report.append(f"- Failed: {metrics['failed_rate']:.1%} ({metrics['failed']}/{metrics['total_samples']})")
        if metrics['samples_retried'] > 0:
            report.append(f"- Samples Retried: {metrics['samples_retried']} (avg {metrics['avg_retries']:.1f} retries each)")
        
        if 'error_types' in metrics and metrics['error_types']:
            report.append("\nError Breakdown:")
            for error_type, count in sorted(metrics['error_types'].items(), key=lambda x: x[1], reverse=True):
                error_name = error_type.replace('_', ' ').title()
                report.append(f"  - {error_name}: {count}")
            report.append("")
        
        # Add ranking-specific metrics if available
        if metrics['successful'] > 0 and 'mean_spearman' in metrics:
            report.append("Spearman measures rank order preservation (>0.7 good, >0.9 excellent), Kendall measures pairwise agreement (>0.5 good, >0.7 excellent).")
            report.append(f"- Mean Spearman Correlation: {metrics['mean_spearman']:.3f} (std: {metrics['std_spearman']:.3f})")
            report.append(f"- Mean Kendall's Tau: {metrics['mean_kendall']:.3f} (std: {metrics['std_kendall']:.3f})")
            report.append(f"- Median Spearman: {metrics['median_spearman']:.3f}")
            report.append(f"- Median Kendall's Tau: {metrics['median_kendall']:.3f}\n")
            
            if 'top1_accuracy' in metrics:
                report.append("### Position Accuracy")
                report.append("Top-1 measures correctly identifying the best response, Top-3 measures having all 3 best responses in the top 3 positions.")
                report.append(f"- Top-1 Accuracy: {metrics['top1_accuracy']:.1%}")
                report.append(f"- Top-3 Accuracy: {metrics['mean_top3_accuracy']:.1%}")
                report.append(f"- Bottom-1 Accuracy: {metrics['bottom1_accuracy']:.1%}")
                report.append(f"- Perfect Rankings: {metrics['perfect_rankings']}/{metrics['successful']} ({metrics['perfect_rankings']/metrics['successful']:.1%})\n")
        
        report.append("### Token Usage")
        report.append(f"- Total Tokens: {metrics['total_tokens']:,}")
        report.append(f"- Average per Sample: {metrics['avg_tokens_per_sample']:.0f} tokens")
        report.append(f"  - Prompt: {metrics['avg_prompt_tokens']:.0f} tokens")
        report.append(f"  - Completion: {metrics['avg_completion_tokens']:.0f} tokens\n")
    
    def _add_detailed_results(self, report: List[str], results: List[Dict], metrics: Dict):
        """Add detailed results tables."""
        report.append("## Detailed Results\n")
        
        # Failed samples table
        failed = [r for r in results if not r['success']]
        if failed:
            report.append(f"### Failed Samples ({len(failed)} total)\n")
            report.append("| Sample ID | Error Type | Detailed Reason | Tokens |")
            report.append("|-----------|------------|-----------------|--------|")
            for r in failed[:20]:  # Show first 20
                error_type = r.get('error', 'Unknown')
                error_detail = r.get('error_detail', 'No details')
                # Truncate long error details
                if len(error_detail) > 60:
                    error_detail = error_detail[:57] + "..."
                report.append(f"| {r['sample_id']} | {error_type} | {error_detail} | {r['tokens']['total']:,} |")
            if len(failed) > 20:
                report.append(f"| ... | ... | ... | ... |")
                report.append(f"| ({len(failed)-20} more failed samples) | | | |")
            report.append("")
        
        # Successful samples table
        successful = [r for r in results if r['success']]
        if successful:
            report.append(f"### Successful Samples ({len(successful)} total)\n")
            
            # Check if this has ranking-specific fields
            if 'spearman' in successful[0]:
                report.append("| Sample   | Spearman | Kendall  | Top-3 Acc | Tokens   |")
                report.append("|----------|----------|----------|-----------|----------|")
                
                for r in successful:
                    # Format numbers with consistent spacing to match header width
                    spearman_str = f"{r['spearman']:+8.3f}"  # "Spearman" is 8 chars
                    kendall_str = f"{r['kendall']:+8.3f}"    # "Kendall" is 7 chars, pad to 8
                    top3_str = f"{r['top3_correct']}/3"      # "Top-3 Acc" is 9 chars
                    tokens_str = f"{r['tokens']['total']:,}"
                    report.append(f"| {r['sample_id']:8s} | {spearman_str} | {kendall_str} | {top3_str:9s} | {tokens_str:8s} |")
            else:
                # Generic success table for non-ranking judges
                report.append("| Sample ID | Success | Tokens |")
                report.append("|-----------|---------|--------|")
                for r in successful:
                    report.append(f"| {r['sample_id']} | ✓ | {r['tokens']['total']:,} |")
            report.append("")
    
    def _add_error_analysis(self, report: List[str], results: List[Dict], metrics: Dict):
        """Add error analysis section."""
        successful = [r for r in results if r['success']]
        if metrics['successful'] > 0 and 'spearman' in successful[0]:  # Ranking-specific analysis
            report.append("## Error Analysis\n")
            
            # Common ranking errors
            report.append("### Common Ranking Errors")
            
            # Analyze adjacent swaps
            adjacent_swaps = defaultdict(int)
            for r in successful:
                for i in range(8):
                    if r['predicted'][i] == r['ground_truth'][i+1] and r['predicted'][i+1] == r['ground_truth'][i]:
                        adjacent_swaps[f"Positions {i+1}-{i+2}"] += 1
            
            if adjacent_swaps:
                report.append("**Adjacent Swaps:**")
                for swap, count in sorted(adjacent_swaps.items(), key=lambda x: x[1], reverse=True)[:5]:
                    report.append(f"- {swap}: {count} occurrences")
                report.append("")
    
    def _add_statistical_distributions(self, report: List[str], results: List[Dict], metrics: Dict):
        """Add statistical distribution visualizations."""
        successful = [r for r in results if r['success']]
        if successful and 'spearman' in successful[0]:
            report.append("## Statistical Distribution\n")
            
            # Spearman correlation distribution
            report.append("### Spearman Correlation Distribution")
            report.append("```")
            self._create_distribution_histogram(
                report, 
                [r['spearman'] for r in successful],
                "Spearman"
            )
            report.append("```")
            report.append("")
            
            # Kendall's tau distribution
            report.append("### Kendall's Tau Distribution")
            report.append("```")
            self._create_distribution_histogram(
                report,
                [r['kendall'] for r in successful], 
                "Kendall"
            )
            report.append("```")
    
    def _create_distribution_histogram(self, report: List[str], values: List[float], name: str):
        """Create text-based histogram for correlation distributions."""
        bins = [(0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2), 
                (-0.2, 0.0), (-0.4, -0.2), (-0.6, -0.4), (-0.8, -0.6), (-1.0, -0.8)]
        max_count = 0
        bin_counts = []
        
        for low, high in bins:
            count = sum(1 for v in values if low <= v < high or (high == 1.0 and v == 1.0) or (low == -1.0 and v == -1.0))
            bin_counts.append((f"{low:+.1f}-{high:+.1f}", count))
            max_count = max(max_count, count)
        
        for range_str, count in bin_counts:
            bar_length = int(15 * count / max_count) if max_count > 0 else 0
            bar = "▓" * bar_length
            report.append(f"{range_str}: {bar} ({count} samples)")


class JudgeEvaluationUtils:
    """Utility functions for judge evaluation."""
    
    @staticmethod
    def generate_output_filename(method: str, model: str, config: Dict, num_samples: int) -> str:
        """Generate standardized output filename."""
        model_short = model.split('/')[-1]  # Keep original format with dashes and capitals
        
        # Build config string from relevant parameters
        config_parts = []
        if 'modal' in config:
            config_parts.append('modal' if config['modal'] else 'mean')
        if 'label_type' in config:
            config_parts.append(config['label_type'])
        if 'instructions' in config:
            instructions_name = Path(config['instructions']).stem
            config_parts.append(instructions_name)
        if 'thinking' in config:
            config_parts.append('thinking' if config['thinking'] else 'no_thinking')
        
        config_str = "_".join(config_parts) if config_parts else "default"
        return f"judge_{method}_{model_short}_{config_str}_{num_samples}samples"
    
    @staticmethod
    def create_output_directory(script_file: str = __file__, base_dir: str = "analysis") -> Path:
        """Create and return output directory path relative to the script location."""
        # Get the directory where the script is located (08-judge-evaluation/)
        script_dir = Path(script_file).parent
        output_dir = script_dir / base_dir
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    @staticmethod
    def create_debug_file_path(script_file: str, base_name: str) -> Path:
        """Create debug file path following the same naming convention."""
        output_dir = JudgeEvaluationUtils.create_output_directory(script_file)
        return output_dir / f"{base_name}_debug.txt"
    
    @staticmethod
    def format_correlation_value(value: float, width: int = 8) -> str:
        """Format correlation value with consistent width and sign."""
        return f"{value:+{width}.3f}"
    
    @staticmethod
    def categorize_error(error_detail: str) -> str:
        """Categorize error based on error detail string."""
        if "No RANKING:" in error_detail:
            return 'no_ranking_found'
        elif "Wrong number of elements" in error_detail:
            return 'wrong_count'
        elif "Duplicate values" in error_detail:
            return 'duplicate_values'
        elif "Missing values" in error_detail:
            return 'missing_values'
        elif "Values outside 1-9 range" in error_detail:
            return 'invalid_range'
        elif "Non-numeric values" in error_detail:
            return 'non_numeric'
        elif "Invalid alphabetic labels" in error_detail:
            return 'invalid_labels'
        else:
            return 'other'