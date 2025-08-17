#!/usr/bin/env python3
"""
Position Bias Test with Dual Requests and Logprob Averaging

Tests position bias by making two LLM requests per sample (original and swapped order)
and averaging logprobs to determine the winner. More robust than single-request approach.

Usage:
    python analyze_simple_position_bias_weighted.py --samples 10
    python analyze_simple_position_bias_weighted.py --samples -1  # Full evaluation
"""

import os
import json
import asyncio
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt
from datasets import load_from_disk
from tqdm import tqdm
import openai
import numpy as np

# Configuration
DATASET_PATH = "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-preference-quality-short-1100-synthetic"
CHARTER_PATH = "08-judge-evaluation/prompts/charter.txt"
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_CONCURRENT = 50
API_TIMEOUT = 120
BASE_URL = "http://148.187.108.173:8092/v1/service/llm/v1/"

# Load charter
def load_charter():
    try:
        with open(CHARTER_PATH, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Charter file not found at {CHARTER_PATH}, using minimal prompt")
        return ""

CHARTER = load_charter()

# Prompt template with charter
PROMPT_TEMPLATE = """{charter}

Based on the above principles, compare these two responses:

Response A: {response_a}

Response B: {response_b}

Which response is better, A or B?

Don't think or explain. Answer with only one character."""


@dataclass
class BiasResult:
    """Result for position bias test with dual requests and logprob averaging."""
    sample_id: str
    winner: Optional[str]  # "rank_0" or "rank_{compare_rank}" based on averaged logprobs
    success: bool
    error: Optional[str] = None
    # Logprob details from dual requests
    chose_a_logprob: Optional[float] = None  # A choice in original order (rank_0=A, compare_rank=B)
    chose_b_logprob: Optional[float] = None  # B choice in original order
    chose_a_swapped_logprob: Optional[float] = None  # A choice in swapped order (compare_rank=A, rank_0=B)
    chose_b_swapped_logprob: Optional[float] = None  # B choice in swapped order
    avg_chose_a_logprob: Optional[float] = None  # Average logprob for choosing rank_0 (better response)
    avg_chose_b_logprob: Optional[float] = None  # Average logprob for choosing compare_rank (worse response)
    # Single-request winners for agreement analysis
    forward_winner: Optional[str] = None  # Winner based on original request only
    backward_winner: Optional[str] = None  # Winner based on swapped request only

    def to_dict(self) -> dict:
        return asdict(self)


class PositionBiasTester:
    """Position bias tester with configurable rank comparison."""
    
    def __init__(self, model: str = DEFAULT_MODEL, max_concurrent: int = DEFAULT_CONCURRENT, compare_rank: int = 1):
        self.model = model
        self.max_concurrent = max_concurrent
        self.compare_rank = compare_rank
        
        api_key = os.getenv("SWISSAI_API_KEY") or os.getenv("SWISS_AI_API_KEY")
        if not api_key:
            raise ValueError("API key required (SWISSAI_API_KEY env var)")
            
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=BASE_URL)
    
    def _extract_choice_logprobs(self, response_obj) -> Dict[str, Optional[float]]:
        """Extract logprobs for both A and B choices."""
        result = {"A": None, "B": None}
        
        if not response_obj.choices or not response_obj.choices[0].logprobs:
            return result
        
        tokens = response_obj.choices[0].logprobs.content
        if not tokens:
            return result
        
        # Find answer token (skip thinking phase if any)
        in_thinking = False
        for token_data in tokens:
            token = token_data.token.strip()
            
            if token == '<think>':
                in_thinking = True
            elif token == '</think>':
                in_thinking = False
            elif not in_thinking and token in ['A', 'B']:
                # Found the chosen answer, now get logprobs for both A and B
                if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                    # Extract logprobs for A and B from top_logprobs
                    for top_token in token_data.top_logprobs:
                        if hasattr(top_token, 'token') and top_token.token.strip() in ['A', 'B']:
                            result[top_token.token.strip()] = top_token.logprob
                    # If the chosen token wasn't in top_logprobs, use its logprob
                    if result[token] is None:
                        result[token] = token_data.logprob
                else:
                    # Fallback: only the chosen token's logprob is available
                    result[token] = token_data.logprob
                break
        
        return result
    
    async def _evaluate_sample(self, sample: Dict[str, Any]) -> BiasResult:
        """Evaluate a single sample for position bias using dual requests and logprob averaging."""
        sample_id = sample.get("conversation_id", str(hash(str(sample))))
        
        # Extract responses with degradation rank 0 and compare_rank
        rank_0_content = None
        compare_rank_content = None
        
        for branch in sample.get('conversation_branches', []):
            for msg in branch.get('messages', []):
                if msg.get('role') == 'assistant' and 'parts' in msg:
                    for part in msg['parts']:
                        if part.get('type') == 'response':
                            rank = part.get('metadata', {}).get('degradation_rank')
                            if rank == 0:
                                rank_0_content = part.get('content', '').strip()
                            elif rank == self.compare_rank:
                                compare_rank_content = part.get('content', '').strip()
        
        if not rank_0_content or not compare_rank_content:
            return BiasResult(
                sample_id=sample_id,
                winner=None,
                success=False,
                error=f"Missing rank 0 or rank {self.compare_rank} responses (found rank 0: {bool(rank_0_content)}, rank {self.compare_rank}: {bool(compare_rank_content)})"
            )
        
        try:
            # Create both prompts: original and swapped
            prompt_original = PROMPT_TEMPLATE.format(charter=CHARTER, response_a=rank_0_content, response_b=compare_rank_content)
            prompt_swapped = PROMPT_TEMPLATE.format(charter=CHARTER, response_a=compare_rank_content, response_b=rank_0_content)
            
            # Make both API calls concurrently
            api_params = {
                "model": self.model,
                "logprobs": True,
                "top_logprobs": 20,
                "temperature": 0.1,
                "max_tokens": 10,
            }
            
            response_original, response_swapped = await asyncio.gather(
                self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_original}],
                    **api_params
                ),
                self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_swapped}],
                    **api_params
                ),
                return_exceptions=True
            )
            
            # Check for API errors
            if isinstance(response_original, Exception):
                return BiasResult(
                    sample_id=sample_id,
                    winner=None,
                    success=False,
                    error=f"Original request failed: {str(response_original)}"
                )
            
            if isinstance(response_swapped, Exception):
                return BiasResult(
                    sample_id=sample_id,
                    winner=None,
                    success=False,
                    error=f"Swapped request failed: {str(response_swapped)}"
                )
            
            # Extract logprobs from both responses
            logprobs_original = self._extract_choice_logprobs(response_original)
            logprobs_swapped = self._extract_choice_logprobs(response_swapped)
            
            # Check if we got valid logprobs
            if logprobs_original["A"] is None and logprobs_original["B"] is None:
                return BiasResult(
                    sample_id=sample_id,
                    winner=None,
                    success=False,
                    error="Could not extract logprobs from original request"
                )
            
            if logprobs_swapped["A"] is None and logprobs_swapped["B"] is None:
                return BiasResult(
                    sample_id=sample_id,
                    winner=None,
                    success=False,
                    error="Could not extract logprobs from swapped request"
                )
            
            # Calculate averaged logprobs
            # In original order: A=rank_0, B=compare_rank
            # In swapped order: A=compare_rank, B=rank_0  
            # So avg_chose_rank_0 = (original_A + swapped_B) / 2
            # And avg_chose_compare_rank = (original_B + swapped_A) / 2
            
            chose_a_logprob = logprobs_original["A"]
            chose_b_logprob = logprobs_original["B"]
            chose_a_swapped_logprob = logprobs_swapped["A"]
            chose_b_swapped_logprob = logprobs_swapped["B"]
            
            # Average logprobs for rank_0 (better response) and compare_rank (worse response)
            avg_chose_rank_0 = None
            avg_chose_compare_rank = None
            
            if chose_a_logprob is not None and chose_b_swapped_logprob is not None:
                avg_chose_rank_0 = (chose_a_logprob + chose_b_swapped_logprob) / 2
            
            if chose_b_logprob is not None and chose_a_swapped_logprob is not None:
                avg_chose_compare_rank = (chose_b_logprob + chose_a_swapped_logprob) / 2
            
            # Determine winner: "rank_0" if better response wins, f"rank_{compare_rank}" if worse response wins
            winner = None
            if avg_chose_rank_0 is not None and avg_chose_compare_rank is not None:
                if avg_chose_rank_0 > avg_chose_compare_rank:
                    winner = "rank_0"  # Better response wins
                else:
                    winner = f"rank_{self.compare_rank}"  # Worse response wins
            
            # Compute forward and backward winners for agreement analysis
            forward_winner = None
            backward_winner = None
            
            # Forward winner: original request (A=rank_0, B=compare_rank)
            if chose_a_logprob is not None and chose_b_logprob is not None:
                if chose_a_logprob > chose_b_logprob:
                    forward_winner = "rank_0"  # A (rank_0) wins
                else:
                    forward_winner = f"rank_{self.compare_rank}"  # B (compare_rank) wins
            
            # Backward winner: swapped request (A=compare_rank, B=rank_0)
            if chose_a_swapped_logprob is not None and chose_b_swapped_logprob is not None:
                if chose_b_swapped_logprob > chose_a_swapped_logprob:
                    backward_winner = "rank_0"  # B (rank_0) wins in swapped order
                else:
                    backward_winner = f"rank_{self.compare_rank}"  # A (compare_rank) wins in swapped order
            
            return BiasResult(
                sample_id=sample_id,
                winner=winner,
                success=True,
                chose_a_logprob=chose_a_logprob,
                chose_b_logprob=chose_b_logprob,
                chose_a_swapped_logprob=chose_a_swapped_logprob,
                chose_b_swapped_logprob=chose_b_swapped_logprob,
                avg_chose_a_logprob=avg_chose_rank_0,
                avg_chose_b_logprob=avg_chose_compare_rank,
                forward_winner=forward_winner,
                backward_winner=backward_winner
            )
                
        except Exception as e:
            return BiasResult(
                sample_id=sample_id,
                winner=None,
                success=False,
                error=str(e)
            )
    
    async def evaluate_samples(self, samples: List[Dict[str, Any]]) -> List[BiasResult]:
        """Evaluate all samples with concurrent processing."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def evaluate_with_semaphore(sample):
            async with semaphore:
                return await self._evaluate_sample(sample)
        
        tasks = [evaluate_with_semaphore(sample) for sample in samples]
        
        results = []
        with tqdm(total=len(tasks), desc="Testing position bias (2 requests per sample)") as pbar:
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                pbar.update(1)
        
        return results


def create_plot(results: List[BiasResult], output_path: Path, compare_rank: int = 1):
    """Create comprehensive 2x2 dashboard showing position bias analysis."""
    successful = [r for r in results if r.success and r.winner and r.forward_winner and r.backward_winner]
    
    if not successful:
        print("No successful results to plot")
        return
    
    total = len(successful)
    
    # Calculate all statistics
    rank_0_wins = sum(1 for r in successful if r.winner == "rank_0")
    rank_0_pct = (rank_0_wins / total) * 100
    compare_rank_pct = 100 - rank_0_pct
    quality_preference = rank_0_pct - 50
    
    # Agreement statistics
    agreement_stats = analyze_agreement(results)
    
    # Method comparison stats
    forward_rank_0 = sum(1 for r in successful if r.forward_winner == "rank_0") / total * 100
    backward_rank_0 = sum(1 for r in successful if r.backward_winner == "rank_0") / total * 100
    
    # Agreement breakdown
    all_agree = sum(1 for r in successful if r.winner == r.forward_winner == r.backward_winner)
    dual_unique = sum(1 for r in successful if r.winner != r.forward_winner or r.winner != r.backward_winner)
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top Left: Quality Preference
    bars1 = ax1.bar(['Better Response\n(Rank 0)', f'Worse Response\n(Rank {compare_rank})'], 
                    [rank_0_pct, compare_rank_pct], 
                    color=['#2E8B57', '#CD5C5C'], alpha=0.8, edgecolor='black')
    
    for bar, pct in zip(bars1, [rank_0_pct, compare_rank_pct]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    ax1.set_ylim(0, max(rank_0_pct, compare_rank_pct) + 10)
    ax1.set_ylabel('Win Percentage', fontweight='bold')
    ax1.set_title(f'Quality Preference\n(Preference: {quality_preference:+.1f}%)', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    # Top Right: Agreement Analysis
    agreement_labels = ['Forward\nvs Dual', 'Backward\nvs Dual', 'Forward vs\nBackward']
    agreement_values = [agreement_stats['forward_dual_agreement'], 
                       agreement_stats['backward_dual_agreement'],
                       agreement_stats['forward_backward_agreement']]
    
    # Color code by agreement level
    colors = ['#228B22' if x >= 90 else '#FFD700' if x >= 80 else '#CD5C5C' for x in agreement_values]
    
    bars2 = ax2.bar(agreement_labels, agreement_values, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars2, agreement_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2.axhline(y=90, color='green', linestyle=':', alpha=0.7, label='High Agreement (90%)')
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Agreement Percentage', fontweight='bold')
    ax2.set_title(f'Method Agreement Analysis\n(Bias Magnitude: {100-agreement_stats["forward_backward_agreement"]:.1f}%)', 
                  fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    # Bottom Left: Position Bias Impact
    impact_labels = ['All Methods\nAgree', 'Dual Averaging\nAdds Value']
    impact_values = [all_agree/total*100, dual_unique/total*100]
    
    bars3 = ax3.bar(impact_labels, impact_values, 
                    color=['#4169E1', '#FF6347'], alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars3, impact_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax3.set_ylim(0, max(impact_values) + 10)
    ax3.set_ylabel('Percentage of Cases', fontweight='bold')
    ax3.set_title('Position Bias Impact', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    
    # Bottom Right: Method Comparison
    method_labels = ['Forward\nOnly', 'Backward\nOnly', 'Dual\nAveraged']
    method_values = [forward_rank_0, backward_rank_0, rank_0_pct]
    
    bars4 = ax4.bar(method_labels, method_values,
                    color=['#9370DB', '#20B2AA', '#2E8B57'], alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars4, method_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    ax4.set_ylim(0, max(method_values) + 10)
    ax4.set_ylabel('Better Response Win Rate', fontweight='bold')
    ax4.set_title('Method Comparison', fontweight='bold', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    ax4.legend()
    
    # Overall title and layout
    fig.suptitle(f'Position Bias Analysis Dashboard: Rank 0 vs Rank {compare_rank} (n={total})', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {output_path}")
    print(f"Dashboard: Quality={quality_preference:+.1f}%, Bias={100-agreement_stats['forward_backward_agreement']:.1f}%, Dual Impact={dual_unique/total*100:.1f}%")


def analyze_agreement(results: List[BiasResult]) -> Dict[str, float]:
    """Analyze agreement rates between forward, backward, and dual-averaged results."""
    # Filter to successful results with all winner types determined
    valid = [r for r in results if r.success and r.winner and r.forward_winner and r.backward_winner]
    
    if not valid:
        return {"forward_dual_agreement": 0.0, "backward_dual_agreement": 0.0, "forward_backward_agreement": 0.0}
    
    total = len(valid)
    
    # Count agreements
    forward_dual_agree = sum(1 for r in valid if r.forward_winner == r.winner)
    backward_dual_agree = sum(1 for r in valid if r.backward_winner == r.winner)
    forward_backward_agree = sum(1 for r in valid if r.forward_winner == r.backward_winner)
    
    return {
        "forward_dual_agreement": forward_dual_agree / total * 100,
        "backward_dual_agreement": backward_dual_agree / total * 100,
        "forward_backward_agreement": forward_backward_agree / total * 100,
        "valid_samples": total
    }


async def main():
    parser = argparse.ArgumentParser(description="Position bias test with configurable rank comparison")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples (-1 for all)")
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT, help="Max concurrent requests")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model for judging")
    parser.add_argument("--output", default=None, help="Output prefix (auto-generated if not specified)")
    parser.add_argument("--compare-rank", type=int, default=1, choices=range(1, 9), 
                       help="Rank to compare against rank 0 (1-8, default: 1)")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
    
    # Get samples
    total_samples = len(dataset['train'])
    if args.samples == -1:
        samples = list(dataset['train'])
    else:
        num_samples = min(args.samples, total_samples)
        samples = [dataset['train'][i] for i in range(num_samples)]
    
    print(f"Testing {len(samples)} samples ({len(samples) * 2} total API requests)")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Output prefix
    if args.output is None:
        model_name = args.model.replace("/", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"dual_request_bias_rank{args.compare_rank}_{model_name}_{len(samples)}samples_{timestamp}"
    else:
        output_prefix = args.output
    
    # Create output directory
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Run evaluation
    tester = PositionBiasTester(model=args.model, max_concurrent=args.concurrent, compare_rank=args.compare_rank)
    results = await tester.evaluate_samples(samples)
    
    # Save results
    results_path = output_dir / f"{output_prefix}.jsonl"
    with open(results_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + '\n')
    
    print(f"\nResults saved to {results_path}")
    
    # Create plot
    plot_path = output_dir / f"{output_prefix}.png"
    create_plot(results, plot_path, compare_rank=args.compare_rank)
    
    # Summary
    successful = [r for r in results if r.success]
    print(f"\nSummary:")
    print(f"  Total tests: {len(results)}")
    print(f"  Successful: {len(successful)}")
    
    if successful:
        rank_0_wins = sum(1 for r in successful if r.winner == "rank_0")
        rank_0_win_rate = rank_0_wins / len(successful) * 100
        print(f"  Better response (rank 0) win rate: {rank_0_win_rate:.1f}%")
        
        # Agreement analysis
        agreement_stats = analyze_agreement(results)
        print(f"\nAgreement Analysis:")
        print(f"  Forward vs Dual: {agreement_stats['forward_dual_agreement']:.1f}%")
        print(f"  Backward vs Dual: {agreement_stats['backward_dual_agreement']:.1f}%") 
        print(f"  Forward vs Backward: {agreement_stats['forward_backward_agreement']:.1f}%")
        print(f"  Position bias magnitude: {100 - agreement_stats['forward_backward_agreement']:.1f}%")
        print(f"  Valid samples for agreement: {agreement_stats['valid_samples']}")


if __name__ == "__main__":
    asyncio.run(main())