#!/usr/bin/env python3
"""
Pairwise Position Bias Analysis Script

Analyzes the results from eval_pairwise_position_bias.py to identify patterns in 
judge behavior when comparing pairs of responses. Focuses on:
- Overall pairwise preferences (A vs B)
- Question order effects ("A vs B" vs "B vs A")
- Position distance effects
- Absolute position effects

Usage:
    python analyze_pairwise_position_bias.py results.jsonl --output analysis/
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(jsonl_path: str) -> pd.DataFrame:
    """Load results from JSONL file into a pandas DataFrame."""
    results = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    return pd.DataFrame(results)


def analyze_pairwise_position_bias(df: pd.DataFrame) -> Dict:
    """Analyze pairwise position bias patterns in the results."""
    
    # Filter successful results only
    successful = df[df['success'] == True].copy()
    
    if len(successful) == 0:
        return {"error": "No successful results to analyze"}
    
    analysis = {}
    
    # Extract metadata from first successful result
    if len(successful) > 0:
        first_result = successful.iloc[0].to_dict()
        judge_model = first_result.get('judge_model', 'unknown')
        
        # Calculate experiment parameters
        num_samples = int(df['sample_id'].nunique()) if 'sample_id' in df.columns else 'unknown'
        orderings_per_sample = int(df['ordering_num'].max()) if 'ordering_num' in df.columns else 'unknown'
        pairs_per_ordering = int(successful['pair_num'].max()) if 'pair_num' in successful.columns else 'unknown'
        
        analysis['test_metadata'] = {
            'judge_model': judge_model,
            'total_test_cases': len(df),
            'successful_test_cases': len(successful),
            'failed_test_cases': len(df) - len(successful),
            'reasoning_modes_tested': sorted(df['reasoning_mode'].unique().tolist()),
            'sample_ids_count': num_samples,
            'orderings_per_sample': orderings_per_sample,
            'unique_pairs_tested': len(successful.groupby(['sample_id', 'ordering_num', 'pair_num'])),
            'pairs_per_ordering': pairs_per_ordering
        }
    
    # Overall winner preferences (A vs B)
    winner_counts = successful['winner'].value_counts()
    total_comparisons = len(successful)
    
    analysis['overall_winner_preferences'] = {
        'counts': winner_counts.to_dict(),
        'percentages': (winner_counts / total_comparisons * 100).round(2).to_dict(),
        'total_comparisons': total_comparisons
    }
    
    # Expected uniform distribution (50-50)
    expected_pct = 50.0
    analysis['expected_uniform_percentage'] = expected_pct
    
    # Chi-square test for uniform distribution
    from scipy.stats import chisquare
    first_count = winner_counts.get('first_mentioned', 0)
    second_count = winner_counts.get('second_mentioned', 0)
    expected_counts = [total_comparisons / 2, total_comparisons / 2]
    actual_counts = [first_count, second_count]
    
    if total_comparisons > 0:
        chi2_stat, chi2_p = chisquare(actual_counts, expected_counts)
        analysis['uniformity_test'] = {
            'chi2_statistic': round(chi2_stat, 4),
            'p_value': round(chi2_p, 6),
            'is_uniform': bool(chi2_p > 0.05)
        }
    
    # Question order effects ("A vs B" vs "B vs A")
    analysis['question_order_effects'] = {}
    for order in successful['question_order'].unique():
        order_data = successful[successful['question_order'] == order]
        order_winners = order_data['winner'].value_counts()
        order_total = len(order_data)
        
        analysis['question_order_effects'][order] = {
            'counts': order_winners.to_dict(),
            'percentages': (order_winners / order_total * 100).round(2).to_dict() if order_total > 0 else {},
            'total_comparisons': order_total
        }
    
    # Position distance effects
    analysis['position_distance_effects'] = {}
    for distance in sorted(successful['position_distance'].unique()):
        distance_data = successful[successful['position_distance'] == distance]
        distance_winners = distance_data['winner'].value_counts()
        distance_total = len(distance_data)
        
        analysis['position_distance_effects'][int(distance)] = {
            'counts': distance_winners.to_dict(),
            'percentages': (distance_winners / distance_total * 100).round(2).to_dict() if distance_total > 0 else {},
            'total_comparisons': distance_total
        }
    
    # Absolute position effects
    analysis['absolute_position_effects'] = {}
    
    # Analyze by first-mentioned completion position
    analysis['absolute_position_effects']['first_mentioned_position'] = {}
    for pos in sorted(successful['first_mentioned_position'].unique()):
        pos_data = successful[successful['first_mentioned_position'] == pos]
        pos_winners = pos_data['winner'].value_counts()
        pos_total = len(pos_data)
        
        analysis['absolute_position_effects']['first_mentioned_position'][int(pos)] = {
            'counts': pos_winners.to_dict(),
            'percentages': (pos_winners / pos_total * 100).round(2).to_dict() if pos_total > 0 else {},
            'total_comparisons': pos_total,
            'first_mentioned_win_rate': pos_winners.get('first_mentioned', 0) / pos_total * 100 if pos_total > 0 else 0
        }
    
    # Analyze by second-mentioned completion position
    analysis['absolute_position_effects']['second_mentioned_position'] = {}
    for pos in sorted(successful['second_mentioned_position'].unique()):
        pos_data = successful[successful['second_mentioned_position'] == pos]
        pos_winners = pos_data['winner'].value_counts()
        pos_total = len(pos_data)
        
        analysis['absolute_position_effects']['second_mentioned_position'][int(pos)] = {
            'counts': pos_winners.to_dict(),
            'percentages': (pos_winners / pos_total * 100).round(2).to_dict() if pos_total > 0 else {},
            'total_comparisons': pos_total,
            'second_mentioned_win_rate': pos_winners.get('second_mentioned', 0) / pos_total * 100 if pos_total > 0 else 0
        }
    
    # Position bias by reasoning mode
    analysis['by_reasoning_mode'] = {}
    for mode in successful['reasoning_mode'].unique():
        mode_data = successful[successful['reasoning_mode'] == mode]
        mode_winners = mode_data['winner'].value_counts()
        mode_total = len(mode_data)
        
        analysis['by_reasoning_mode'][mode] = {
            'counts': mode_winners.to_dict(),
            'percentages': (mode_winners / mode_total * 100).round(2).to_dict() if mode_total > 0 else {},
            'total_comparisons': mode_total
        }
    
    # Consistency analysis per sample
    sample_consistency = {}
    
    for sample_id in successful['sample_id'].unique():
        sample_data = successful[successful['sample_id'] == sample_id]
        
        consistency_by_mode = {}
        
        for mode in sample_data['reasoning_mode'].unique():
            mode_data = sample_data[sample_data['reasoning_mode'] == mode]
            
            # Group by pair to check consistency across orderings
            pair_consistency = []
            
            for ordering_num in mode_data['ordering_num'].unique():
                ordering_data = mode_data[mode_data['ordering_num'] == ordering_num]
                for pair_num in ordering_data['pair_num'].unique():
                    pair_data = ordering_data[ordering_data['pair_num'] == pair_num]
                    if len(pair_data) > 1:  # Multiple question orders for same pair
                        winners = pair_data['winner'].tolist()
                        most_common_winner = Counter(winners).most_common(1)[0]
                        consistency_rate = most_common_winner[1] / len(winners)
                        pair_consistency.append(consistency_rate)
            
            if pair_consistency:
                consistency_by_mode[mode] = {
                    'mean_consistency': round(np.mean(pair_consistency), 3),
                    'total_pairs_analyzed': len(pair_consistency)
                }
        
        sample_consistency[sample_id] = consistency_by_mode
    
    # Overall consistency statistics
    all_consistency_rates = []
    for sample_data in sample_consistency.values():
        for mode_data in sample_data.values():
            if 'mean_consistency' in mode_data:
                all_consistency_rates.append(mode_data['mean_consistency'])
    
    if all_consistency_rates:
        analysis['consistency_stats'] = {
            'mean_consistency': round(np.mean(all_consistency_rates), 3),
            'median_consistency': round(np.median(all_consistency_rates), 3),
            'min_consistency': round(np.min(all_consistency_rates), 3),
            'max_consistency': round(np.max(all_consistency_rates), 3),
            'std_consistency': round(np.std(all_consistency_rates), 3)
        }
    
    # Bias indicators
    analysis['bias_indicators'] = {}
    
    # Overall bias towards first-mentioned or second-mentioned
    first_pct = analysis['overall_winner_preferences']['percentages'].get('first_mentioned', 0)
    second_pct = analysis['overall_winner_preferences']['percentages'].get('second_mentioned', 0)
    analysis['bias_indicators']['winner_bias'] = {
        'first_mentioned_percentage': first_pct,
        'second_mentioned_percentage': second_pct,
        'is_significantly_biased': bool(abs(first_pct - second_pct) > 10),  # More than 10% difference
        'bias_direction': 'first_mentioned' if first_pct > second_pct else 'second_mentioned' if second_pct > first_pct else 'None'
    }
    
    # Question order bias
    if 'original_order' in analysis['question_order_effects'] and 'reversed_order' in analysis['question_order_effects']:
        original_first_pct = analysis['question_order_effects']['original_order']['percentages'].get('first_mentioned', 0)
        reversed_first_pct = analysis['question_order_effects']['reversed_order']['percentages'].get('first_mentioned', 0)
        
        analysis['bias_indicators']['question_order_bias'] = {
            'original_order_chooses_first': original_first_pct,
            'reversed_order_chooses_first': reversed_first_pct,
            'order_effect_magnitude': abs(original_first_pct - reversed_first_pct),
            'has_significant_order_effect': bool(abs(original_first_pct - reversed_first_pct) > 5)
        }
    
    return analysis


def _create_figure_title(base_title: str, analysis: Dict) -> str:
    """Create an informative figure title with metadata."""
    metadata = analysis.get('test_metadata', {})
    model = metadata.get('judge_model', 'unknown')
    num_samples = metadata.get('sample_ids_count', 'unknown')
    orderings = metadata.get('orderings_per_sample', 'unknown')
    pairs = metadata.get('pairs_per_ordering', 'unknown')
    
    # Clean up model name for display
    if isinstance(model, str) and model.startswith('claude-'):
        model_display = model.replace('claude-', '').replace('-20241022', '').replace('-', ' ').title()
        if '3' in model_display and '5' in model_display:
            model_display = model_display.replace('3 5', '3.5')
    else:
        model_display = str(model)
    
    subtitle = f"Model: {model_display} | Samples: {num_samples} | Orderings: {orderings} | Pairs/Ordering: {pairs}"
    return f"{base_title}\n{subtitle}"


def create_visualizations(df: pd.DataFrame, analysis: Dict, output_dir: Path):
    """Create visualizations for pairwise position bias analysis."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter successful results
    successful = df[df['success'] == True]
    
    if len(successful) == 0:
        print("No successful results to visualize")
        return
    
    # 1. Overall winner preferences
    plt.figure(figsize=(10, 6))
    
    winner_data = analysis['overall_winner_preferences']
    winners = list(winner_data['percentages'].keys())
    percentages = list(winner_data['percentages'].values())
    
    bars = plt.bar(winners, percentages, alpha=0.7, color=['lightcoral', 'lightblue'])
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Expected uniform: 50%')
    
    plt.xlabel('Winner')
    plt.ylabel('Percentage of Comparisons')
    plt.title(_create_figure_title('Overall Pairwise Winner Preferences', analysis))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pairwise_winner_preferences.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Question order effects
    if 'question_order_effects' in analysis:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        orders = list(analysis['question_order_effects'].keys())
        x = np.arange(len(orders))
        width = 0.35
        
        # Create clearer labels for the orders
        order_labels = []
        for order in orders:
            if order == 'original_order':
                order_labels.append('Original order')
            elif order == 'reversed_order':
                order_labels.append('Reversed order')
            else:
                order_labels.append(order)
        
        first_percentages = [analysis['question_order_effects'][order]['percentages'].get('first_mentioned', 0) for order in orders]
        second_percentages = [analysis['question_order_effects'][order]['percentages'].get('second_mentioned', 0) for order in orders]
        
        bars1 = ax.bar(x - width/2, first_percentages, width, label='First-mentioned wins', alpha=0.7, color='lightcoral')
        bars2 = ax.bar(x + width/2, second_percentages, width, label='Second-mentioned wins', alpha=0.7, color='lightblue')
        
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Expected uniform: 50%')
        ax.set_xlabel('Question Order')
        ax.set_ylabel('Percentage of Comparisons')
        ax.set_title(_create_figure_title('Winner Preferences by Question Order', analysis))
        ax.set_xticks(x)
        ax.set_xticklabels(order_labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'question_order_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Position distance effects
    if 'position_distance_effects' in analysis:
        plt.figure(figsize=(12, 6))
        
        distances = sorted(analysis['position_distance_effects'].keys())
        first_percentages = [analysis['position_distance_effects'][dist]['percentages'].get('first_mentioned', 0) for dist in distances]
        second_percentages = [analysis['position_distance_effects'][dist]['percentages'].get('second_mentioned', 0) for dist in distances]
        
        x = np.arange(len(distances))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, first_percentages, width, label='First-mentioned wins', alpha=0.7, color='lightcoral')
        bars2 = plt.bar(x + width/2, second_percentages, width, label='Second-mentioned wins', alpha=0.7, color='lightblue')
        
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Expected uniform: 50%')
        plt.xlabel('Position Distance')
        plt.ylabel('Percentage of Comparisons')
        plt.title(_create_figure_title('Winner Preferences by Position Distance', analysis))
        plt.xticks(x, distances)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'position_distance_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Absolute position effects for first-mentioned completion
    if 'absolute_position_effects' in analysis and 'first_mentioned_position' in analysis['absolute_position_effects']:
        plt.figure(figsize=(12, 6))
        
        pos_data = analysis['absolute_position_effects']['first_mentioned_position']
        positions = sorted(pos_data.keys())
        first_win_rates = [pos_data[pos]['first_mentioned_win_rate'] for pos in positions]
        
        bars = plt.bar(positions, first_win_rates, alpha=0.7, color='lightcoral')
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Expected uniform: 50%')
        
        plt.xlabel('First-mentioned Completion Position')
        plt.ylabel('First-mentioned Win Rate (%)')
        plt.title(_create_figure_title('Win Rate for First-mentioned Completion by Its Absolute Position', analysis))
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for bar, rate in zip(bars, first_win_rates):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'first_mentioned_position_effects.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_report(analysis: Dict, output_dir: Path):
    """Generate a comprehensive text report."""
    
    report_path = output_dir / 'pairwise_position_bias_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("PAIRWISE POSITION BIAS ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        if 'error' in analysis:
            f.write(f"Error: {analysis['error']}\n")
            return
        
        # Test metadata section
        if 'test_metadata' in analysis:
            f.write("TEST METADATA\n")
            f.write("-" * 15 + "\n")
            metadata = analysis['test_metadata']
            f.write(f"Judge model: {metadata.get('judge_model', 'unknown')}\n")
            f.write(f"Total test cases: {metadata.get('total_test_cases', 0)}\n")
            f.write(f"Successful test cases: {metadata.get('successful_test_cases', 0)}\n")
            f.write(f"Failed test cases: {metadata.get('failed_test_cases', 0)}\n")
            f.write(f"Reasoning modes tested: {', '.join(metadata.get('reasoning_modes_tested', []))}\n")
            f.write(f"Unique samples tested: {metadata.get('sample_ids_count', 'unknown')}\n")
            f.write(f"Orderings per sample: {metadata.get('orderings_per_sample', 'unknown')}\n")
            f.write(f"Unique pairs tested: {metadata.get('unique_pairs_tested', 'unknown')}\n")
            f.write(f"Pairs per ordering: {metadata.get('pairs_per_ordering', 'unknown')}\n\n")
        
        # Overall winner preferences
        f.write("OVERALL WINNER PREFERENCES\n")
        f.write("-" * 30 + "\n")
        
        winner_prefs = analysis['overall_winner_preferences']
        f.write(f"Total comparisons analyzed: {winner_prefs['total_comparisons']}\n")
        f.write(f"Expected uniform percentage: {analysis['expected_uniform_percentage']:.1f}%\n\n")
        
        f.write("Winner preferences:\n")
        for winner in sorted(winner_prefs['percentages'].keys()):
            pct = winner_prefs['percentages'][winner]
            count = winner_prefs['counts'][winner]
            f.write(f"  Winner {winner}: {count} comparisons ({pct:.2f}%)\n")
        
        # Statistical test
        if 'uniformity_test' in analysis:
            f.write(f"\nUniformity test (Chi-square):\n")
            uniformity = analysis['uniformity_test']
            f.write(f"  Chi-square statistic: {uniformity['chi2_statistic']}\n")
            f.write(f"  P-value: {uniformity['p_value']}\n")
            f.write(f"  Is uniform distribution: {'Yes' if uniformity['is_uniform'] else 'No'}\n")
        
        # Bias indicators
        f.write(f"\nBIAS INDICATORS\n")
        f.write("-" * 15 + "\n")
        
        bias_ind = analysis['bias_indicators']
        
        # Winner bias
        if 'winner_bias' in bias_ind:
            winner_bias = bias_ind['winner_bias']
            f.write(f"Overall winner bias:\n")
            f.write(f"  First-mentioned wins: {winner_bias['first_mentioned_percentage']:.2f}%\n")
            f.write(f"  Second-mentioned wins: {winner_bias['second_mentioned_percentage']:.2f}%\n")
            f.write(f"  Significantly biased: {'Yes' if winner_bias['is_significantly_biased'] else 'No'}\n")
            f.write(f"  Bias direction: {winner_bias['bias_direction']}\n")
        
        # Question order bias
        if 'question_order_bias' in bias_ind:
            order_bias = bias_ind['question_order_bias']
            f.write(f"\nQuestion order bias:\n")
            f.write(f"  Original order chooses first-mentioned: {order_bias['original_order_chooses_first']:.2f}%\n")
            f.write(f"  Reversed order chooses first-mentioned: {order_bias['reversed_order_chooses_first']:.2f}%\n")
            f.write(f"  Order effect magnitude: {order_bias['order_effect_magnitude']:.2f}%\n")
            f.write(f"  Significant order effect: {'Yes' if order_bias['has_significant_order_effect'] else 'No'}\n")
        
        # Question order effects
        if 'question_order_effects' in analysis:
            f.write(f"\nQUESTION ORDER EFFECTS\n")
            f.write("-" * 22 + "\n")
            
            for order, order_data in analysis['question_order_effects'].items():
                f.write(f"\n{order}:\n")
                f.write(f"  Total comparisons: {order_data['total_comparisons']}\n")
                f.write("  Winner preferences:\n")
                for winner in sorted(order_data['percentages'].keys()):
                    pct = order_data['percentages'][winner]
                    count = order_data['counts'][winner]
                    f.write(f"    Winner {winner}: {count} ({pct:.2f}%)\n")
        
        # Position distance effects
        if 'position_distance_effects' in analysis:
            f.write(f"\nPOSITION DISTANCE EFFECTS\n")
            f.write("-" * 25 + "\n")
            
            for distance, distance_data in analysis['position_distance_effects'].items():
                f.write(f"\nDistance {distance}:\n")
                f.write(f"  Total comparisons: {distance_data['total_comparisons']}\n")
                f.write("  Winner preferences:\n")
                for winner in sorted(distance_data['percentages'].keys()):
                    pct = distance_data['percentages'][winner]
                    count = distance_data['counts'][winner]
                    f.write(f"    Winner {winner}: {count} ({pct:.2f}%)\n")
        
        # Consistency statistics
        if 'consistency_stats' in analysis:
            f.write(f"\nCONSISTENCY ANALYSIS\n")
            f.write("-" * 20 + "\n")
            
            consistency = analysis['consistency_stats']
            f.write(f"Mean consistency rate: {consistency['mean_consistency']:.3f}\n")
            f.write(f"Median consistency rate: {consistency['median_consistency']:.3f}\n")
            f.write(f"Range: {consistency['min_consistency']:.3f} - {consistency['max_consistency']:.3f}\n")
            f.write(f"Standard deviation: {consistency['std_consistency']:.3f}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze pairwise position bias test results")
    parser.add_argument("results", help="Path to JSONL results file")
    parser.add_argument("--output", default=None, 
                       help="Output directory for analysis (auto-generated if not specified)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}")
    df = load_results(args.results)
    
    print(f"Loaded {len(df)} test results")
    print(f"Successful results: {sum(df['success'])}")
    print(f"Failed results: {sum(~df['success'])}")
    
    if sum(df['success']) == 0:
        print("No successful results to analyze!")
        return
    
    # Auto-generate output directory if not provided
    if args.output is None:
        # Extract base filename without extension
        input_path = Path(args.results)
        base_name = input_path.stem
        if base_name.endswith('_bias'):
            base_name = base_name[:-5]  # Remove '_bias' suffix if present
        output_dir = input_path.parent / f"{base_name}_analysis"
        print(f"Auto-generated output directory: {output_dir}")
    else:
        output_dir = Path(args.output)
    
    # Perform analysis
    print("Performing pairwise position bias analysis...")
    analysis = analyze_pairwise_position_bias(df)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save analysis as JSON
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Generate report
    generate_report(analysis, output_dir)
    
    # Create visualizations
    if not args.no_plots:
        try:
            create_visualizations(df, analysis, output_dir)
            print("Visualizations created successfully")
        except ImportError:
            print("Warning: matplotlib/seaborn not available, skipping plots")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    
    # Print quick summary
    if 'overall_winner_preferences' in analysis:
        print("\nQuick Summary:")
        winner_prefs = analysis['overall_winner_preferences']['percentages']
        
        first_pct = winner_prefs.get('first_mentioned', 0)
        second_pct = winner_prefs.get('second_mentioned', 0)
        
        print(f"  First-mentioned wins: {first_pct:.1f}%")
        print(f"  Second-mentioned wins: {second_pct:.1f}%")
        print(f"  Expected uniform: 50.0%")
        print(f"  Is uniform: {'Yes' if analysis.get('uniformity_test', {}).get('is_uniform', False) else 'No'}")
        
        if 'bias_indicators' in analysis and 'question_order_bias' in analysis['bias_indicators']:
            order_bias = analysis['bias_indicators']['question_order_bias']
            print(f"  Question order effect: {order_bias['order_effect_magnitude']:.1f}% magnitude")


if __name__ == "__main__":
    main()