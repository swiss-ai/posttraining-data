#!/usr/bin/env python3
"""
Position Bias Analysis Script

Analyzes the results from eval_position_bias.py to identify patterns in judge behavior.
Creates comprehensive statistics and visualizations showing position preferences.

Usage:
    python analyze_position_bias.py results.jsonl --output analysis/
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


def analyze_position_bias(df: pd.DataFrame) -> Dict:
    """Analyze position bias patterns in the results."""
    
    # Filter successful results only
    successful = df[df['success'] == True].copy()
    
    if len(successful) == 0:
        return {"error": "No successful results to analyze"}
    
    analysis = {}
    
    # Extract metadata from first successful result
    if len(successful) > 0:
        # Try to get model name from the test results if available
        first_result = successful.iloc[0].to_dict()
        judge_model = first_result.get('judge_model', 'unknown')
        
        # Calculate experiment parameters
        num_samples = int(df['sample_id'].nunique()) if 'sample_id' in df.columns else 'unknown'
        orderings_per_sample = int(df['ordering_num'].max()) if 'ordering_num' in df.columns else 'unknown'
        
        analysis['test_metadata'] = {
            'judge_model': judge_model,
            'total_test_cases': len(df),
            'successful_test_cases': len(successful),
            'failed_test_cases': len(df) - len(successful),
            'reasoning_modes_tested': sorted(df['reasoning_mode'].unique().tolist()),
            'sample_ids_count': num_samples,
            'orderings_per_sample': orderings_per_sample
        }
    
    # Overall position preferences
    position_counts = successful['chosen_position'].value_counts().sort_index()
    total_choices = len(successful)
    
    analysis['overall_position_preferences'] = {
        'counts': position_counts.to_dict(),
        'percentages': (position_counts / total_choices * 100).round(2).to_dict(),
        'total_choices': total_choices
    }
    
    # Expected uniform distribution
    num_positions = int(position_counts.index.max())
    expected_pct = 100 / num_positions
    analysis['expected_uniform_percentage'] = round(expected_pct, 2)
    
    # Chi-square test for uniform distribution
    from scipy.stats import chisquare
    expected_counts = [total_choices / num_positions] * num_positions
    actual_counts = [position_counts.get(i, 0) for i in range(1, num_positions + 1)]
    chi2_stat, chi2_p = chisquare(actual_counts, expected_counts)
    
    analysis['uniformity_test'] = {
        'chi2_statistic': round(chi2_stat, 4),
        'p_value': round(chi2_p, 6),
        'is_uniform': bool(chi2_p > 0.05)  # Convert to Python bool for JSON
    }
    
    # Position bias by reasoning mode
    analysis['by_reasoning_mode'] = {}
    for mode in successful['reasoning_mode'].unique():
        mode_data = successful[successful['reasoning_mode'] == mode]
        mode_counts = mode_data['chosen_position'].value_counts().sort_index()
        mode_total = len(mode_data)
        
        analysis['by_reasoning_mode'][mode] = {
            'counts': mode_counts.to_dict(),
            'percentages': (mode_counts / mode_total * 100).round(2).to_dict(),
            'total_choices': mode_total
        }
    
    # Consistency analysis per sample
    sample_consistency = {}
    
    for sample_id in successful['sample_id'].unique():
        sample_data = successful[successful['sample_id'] == sample_id]
        
        # Group by reasoning mode
        consistency_by_mode = {}
        
        for mode in sample_data['reasoning_mode'].unique():
            mode_data = sample_data[sample_data['reasoning_mode'] == mode]
            chosen_positions = mode_data['chosen_position'].tolist()
            
            # Calculate consistency (how often same choice is made)
            if len(chosen_positions) > 1:
                most_common_pos = Counter(chosen_positions).most_common(1)[0]
                consistency_rate = most_common_pos[1] / len(chosen_positions)
            else:
                consistency_rate = 1.0
            
            consistency_by_mode[mode] = {
                'total_orderings': len(chosen_positions),
                'chosen_positions': chosen_positions,
                'most_common_position': most_common_pos[0] if len(chosen_positions) > 1 else chosen_positions[0],
                'consistency_rate': round(consistency_rate, 3)
            }
        
        sample_consistency[sample_id] = consistency_by_mode
    
    # Overall consistency statistics
    all_consistency_rates = []
    for sample_data in sample_consistency.values():
        for mode_data in sample_data.values():
            all_consistency_rates.append(mode_data['consistency_rate'])
    
    analysis['consistency_stats'] = {
        'mean_consistency': round(np.mean(all_consistency_rates), 3),
        'median_consistency': round(np.median(all_consistency_rates), 3),
        'min_consistency': round(np.min(all_consistency_rates), 3),
        'max_consistency': round(np.max(all_consistency_rates), 3),
        'std_consistency': round(np.std(all_consistency_rates), 3)
    }
    
    # Strong bias indicators
    analysis['bias_indicators'] = {}
    
    # Primacy bias (preference for first position)
    first_pos_pct = analysis['overall_position_preferences']['percentages'].get(1, 0)
    analysis['bias_indicators']['primacy_bias'] = {
        'first_position_percentage': first_pos_pct,
        'is_significantly_high': bool(first_pos_pct > expected_pct * 1.5)
    }
    
    # Recency bias (preference for last position)
    last_pos_pct = analysis['overall_position_preferences']['percentages'].get(num_positions, 0)
    analysis['bias_indicators']['recency_bias'] = {
        'last_position_percentage': last_pos_pct,
        'is_significantly_high': bool(last_pos_pct > expected_pct * 1.5)
    }
    
    # Middle bias (preference for middle positions)
    if num_positions >= 3:
        middle_positions = list(range(2, num_positions))
        middle_pct = sum(analysis['overall_position_preferences']['percentages'].get(pos, 0) 
                        for pos in middle_positions)
        expected_middle_pct = expected_pct * len(middle_positions)
        
        analysis['bias_indicators']['middle_bias'] = {
            'middle_positions_percentage': round(middle_pct, 2),
            'expected_middle_percentage': round(expected_middle_pct, 2),
            'is_significantly_high': bool(middle_pct > expected_middle_pct * 1.2)
        }
    
    return analysis


def _create_figure_title(base_title: str, analysis: Dict) -> str:
    """Create an informative figure title with metadata."""
    metadata = analysis.get('test_metadata', {})
    model = metadata.get('judge_model', 'unknown')
    num_samples = metadata.get('sample_ids_count', 'unknown')
    orderings = metadata.get('orderings_per_sample', 'unknown')
    
    # Clean up model name for display
    if isinstance(model, str) and model.startswith('claude-'):
        model_display = model.replace('claude-', '').replace('-20241022', '').replace('-', ' ').title()
        if '3' in model_display and '5' in model_display:
            model_display = model_display.replace('3 5', '3.5')
    else:
        model_display = str(model)
    
    subtitle = f"Model: {model_display} | Samples: {num_samples} | Orderings: {orderings}"
    return f"{base_title}\n{subtitle}"


def create_visualizations(df: pd.DataFrame, analysis: Dict, output_dir: Path):
    """Create visualizations for position bias analysis."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter successful results
    successful = df[df['success'] == True]
    
    if len(successful) == 0:
        print("No successful results to visualize")
        return
    
    # 1. Overall position preference bar chart
    plt.figure(figsize=(12, 6))
    
    position_data = analysis['overall_position_preferences']
    positions = list(position_data['percentages'].keys())
    percentages = list(position_data['percentages'].values())
    expected_pct = analysis['expected_uniform_percentage']
    
    bars = plt.bar(positions, percentages, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.axhline(y=expected_pct, color='red', linestyle='--', alpha=0.7, 
                label=f'Expected uniform: {expected_pct:.1f}%')
    
    # Highlight bars that deviate significantly from expected
    for i, (pos, pct) in enumerate(zip(positions, percentages)):
        if abs(pct - expected_pct) > expected_pct * 0.5:  # 50% deviation
            bars[i].set_color('orange')
            bars[i].set_alpha(0.9)
    
    plt.xlabel('Position')
    plt.ylabel('Percentage of Choices')
    plt.title(_create_figure_title('Judge Position Preferences (All Results)', analysis))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'position_preferences_overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Position preferences by reasoning mode
    if 'by_reasoning_mode' in analysis and len(analysis['by_reasoning_mode']) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        modes = list(analysis['by_reasoning_mode'].keys())
        
        # Calculate shared y-axis limits
        all_percentages = []
        for mode in modes:
            mode_data = analysis['by_reasoning_mode'][mode]
            all_percentages.extend(list(mode_data['percentages'].values()))
        
        y_max = max(max(all_percentages), expected_pct * 1.2) * 1.1  # Add 10% padding above highest bar
        y_min = 0
        
        for idx, mode in enumerate(modes):
            mode_data = analysis['by_reasoning_mode'][mode]
            positions = list(mode_data['percentages'].keys())
            percentages = list(mode_data['percentages'].values())
            
            bars = axes[idx].bar(positions, percentages, alpha=0.7, 
                               color='lightcoral' if 'reasoning' in mode else 'lightgreen')
            axes[idx].axhline(y=expected_pct, color='red', linestyle='--', alpha=0.7)
            axes[idx].set_xlabel('Position')
            axes[idx].set_ylabel('Percentage of Choices')
            mode_title = f'Position Preferences: {mode.replace("_", " ").title()}'
            axes[idx].set_title(_create_figure_title(mode_title, analysis))
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Set shared y-axis limits
            axes[idx].set_ylim(y_min, y_max)
            
            # Add percentage labels
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'position_preferences_by_reasoning.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Consistency distribution histogram
    if 'consistency_stats' in analysis:
        plt.figure(figsize=(10, 6))
        
        # Collect all consistency rates
        consistency_rates = []
        for sample_id in successful['sample_id'].unique():
            sample_data = successful[successful['sample_id'] == sample_id]
            for mode in sample_data['reasoning_mode'].unique():
                mode_data = sample_data[sample_data['reasoning_mode'] == mode]
                chosen_positions = mode_data['chosen_position'].tolist()
                
                if len(chosen_positions) > 1:
                    most_common_count = Counter(chosen_positions).most_common(1)[0][1]
                    consistency_rate = most_common_count / len(chosen_positions)
                    consistency_rates.append(consistency_rate)
        
        plt.hist(consistency_rates, bins=20, alpha=0.7, color='lightblue', edgecolor='navy')
        plt.xlabel('Consistency Rate (fraction of orderings with same choice)')
        plt.ylabel('Frequency')
        plt.title(_create_figure_title('Distribution of Judge Consistency Across Samples', analysis))
        plt.axvline(x=np.mean(consistency_rates), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(consistency_rates):.3f}')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'consistency_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Heatmap of position choices across samples
    if len(successful['sample_id'].unique()) <= 50:  # Only for reasonable number of samples
        pivot_data = successful.pivot_table(
            index='sample_id', 
            columns='chosen_position', 
            values='ordering_num', 
            aggfunc='count', 
            fill_value=0
        )
        
        plt.figure(figsize=(12, max(8, len(pivot_data) * 0.3)))
        sns.heatmap(pivot_data, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Number of Choices'})
        plt.xlabel('Chosen Position')
        plt.ylabel('Sample ID')
        plt.title(_create_figure_title('Position Choices Across Samples', analysis))
        plt.tight_layout()
        plt.savefig(output_dir / 'position_choices_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_report(analysis: Dict, output_dir: Path):
    """Generate a comprehensive text report."""
    
    report_path = output_dir / 'position_bias_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("POSITION BIAS ANALYSIS REPORT\n")
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
            f.write(f"Orderings per sample: {metadata.get('orderings_per_sample', 'unknown')}\n\n")
        
        # Overall statistics
        f.write("OVERALL POSITION PREFERENCES\n")
        f.write("-" * 30 + "\n")
        
        pos_prefs = analysis['overall_position_preferences']
        f.write(f"Total choices analyzed: {pos_prefs['total_choices']}\n")
        f.write(f"Expected uniform percentage: {analysis['expected_uniform_percentage']:.2f}%\n\n")
        
        f.write("Position preferences:\n")
        for pos in sorted(pos_prefs['percentages'].keys()):
            pct = pos_prefs['percentages'][pos]
            count = pos_prefs['counts'][pos]
            f.write(f"  Position {pos}: {count} choices ({pct:.2f}%)\n")
        
        # Statistical test
        f.write(f"\nUniformity test (Chi-square):\n")
        uniformity = analysis['uniformity_test']
        f.write(f"  Chi-square statistic: {uniformity['chi2_statistic']}\n")
        f.write(f"  P-value: {uniformity['p_value']}\n")
        f.write(f"  Is uniform distribution: {'Yes' if uniformity['is_uniform'] else 'No'}\n")
        
        # Bias indicators
        f.write(f"\nBIAS INDICATORS\n")
        f.write("-" * 15 + "\n")
        
        bias_ind = analysis['bias_indicators']
        
        # Primacy bias
        primacy = bias_ind['primacy_bias']
        f.write(f"Primacy bias (first position preference):\n")
        f.write(f"  First position percentage: {primacy['first_position_percentage']:.2f}%\n")
        f.write(f"  Significantly high: {'Yes' if primacy['is_significantly_high'] else 'No'}\n")
        
        # Recency bias
        recency = bias_ind['recency_bias']
        f.write(f"\nRecency bias (last position preference):\n")
        f.write(f"  Last position percentage: {recency['last_position_percentage']:.2f}%\n")
        f.write(f"  Significantly high: {'Yes' if recency['is_significantly_high'] else 'No'}\n")
        
        # Middle bias
        if 'middle_bias' in bias_ind:
            middle = bias_ind['middle_bias']
            f.write(f"\nMiddle bias (middle positions preference):\n")
            f.write(f"  Middle positions percentage: {middle['middle_positions_percentage']:.2f}%\n")
            f.write(f"  Expected middle percentage: {middle['expected_middle_percentage']:.2f}%\n")
            f.write(f"  Significantly high: {'Yes' if middle['is_significantly_high'] else 'No'}\n")
        
        # Consistency statistics
        if 'consistency_stats' in analysis:
            f.write(f"\nCONSISTENCY ANALYSIS\n")
            f.write("-" * 20 + "\n")
            
            consistency = analysis['consistency_stats']
            f.write(f"Mean consistency rate: {consistency['mean_consistency']:.3f}\n")
            f.write(f"Median consistency rate: {consistency['median_consistency']:.3f}\n")
            f.write(f"Range: {consistency['min_consistency']:.3f} - {consistency['max_consistency']:.3f}\n")
            f.write(f"Standard deviation: {consistency['std_consistency']:.3f}\n")
        
        # Reasoning mode comparison
        if 'by_reasoning_mode' in analysis:
            f.write(f"\nBY REASONING MODE\n")
            f.write("-" * 17 + "\n")
            
            for mode, mode_data in analysis['by_reasoning_mode'].items():
                f.write(f"\n{mode.replace('_', ' ').title()}:\n")
                f.write(f"  Total choices: {mode_data['total_choices']}\n")
                f.write("  Position preferences:\n")
                for pos in sorted(mode_data['percentages'].keys()):
                    pct = mode_data['percentages'][pos]
                    count = mode_data['counts'][pos]
                    f.write(f"    Position {pos}: {count} ({pct:.2f}%)\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze position bias test results")
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
    print("Performing position bias analysis...")
    analysis = analyze_position_bias(df)
    
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
    if 'overall_position_preferences' in analysis:
        print("\nQuick Summary:")
        pos_prefs = analysis['overall_position_preferences']['percentages']
        expected = analysis['expected_uniform_percentage']
        
        max_pos = max(pos_prefs.keys(), key=lambda k: pos_prefs[k])
        max_pct = pos_prefs[max_pos]
        
        print(f"  Most preferred position: {max_pos} ({max_pct:.1f}%)")
        print(f"  Expected uniform: {expected:.1f}%")
        print(f"  Is uniform: {'Yes' if analysis['uniformity_test']['is_uniform'] else 'No'}")


if __name__ == "__main__":
    main()