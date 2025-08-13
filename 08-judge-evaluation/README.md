# Judge Evaluation Framework

This folder contains tools for evaluating judge model behavior and bias patterns, particularly focusing on position bias in response ranking and pairwise comparisons.

## Overview

Judge models are LLMs used to evaluate and rank responses from other AI systems. However, these judges can exhibit systematic biases that affect their reliability. This framework provides comprehensive tools to identify and analyze such biases.

## Position Bias Types

**Position bias** occurs when a judge's preference is influenced by where a response appears in a list, rather than its actual quality:

- **Primacy bias**: Preference for responses at the beginning of a list
- **Recency bias**: Preference for responses at the end of a list  
- **Middle bias**: Preference for responses in middle positions
- **Distance effects**: How the relative distance between compared items affects preferences

## Tools

### 1. Synthetic Completions Generation

**`generate_synthetic_completions.py`** - Generates degraded completions for preference learning datasets

**Features:**
- Iterative degradation using comprehensive quality dimensions
- Exact degradation logic from `iterative_degradation.py`
- Batch processing with adaptive concurrency
- Token usage tracking for each generation
- Randomized conversation branch ordering
- Full dataset schema compliance

**Usage:**
```bash
# Generate synthetic preference dataset
venv/bin/python generate_synthetic_completions.py data/olmo-2-preference-quality-500 \
    --samples 100 \
    --iterations 5 \
    --output data/preference_synthetic_completions

# Quick test with small sample
venv/bin/python generate_synthetic_completions.py data/input \
    --samples 10 \
    --iterations 3 \
    --concurrent 20

# Large-scale generation with auto-output naming
venv/bin/python generate_synthetic_completions.py data/high_quality_dataset \
    --samples 1000 \
    --iterations 7 \
    --concurrent 100 \
    --disable-adaptive
```

**Output Dataset Structure:**
- Each sample contains original + N degraded completions as conversation branches
- Branches are randomized to prevent position bias during training
- Full metadata tracking:
  - `degraded`: true/false flag
  - `iteration`: 0 (original) to N (degraded)
  - `degradation_rank`: Ground truth preference rank (0=best, N=worst)
  - `degradation_reasoning`: Explanation of quality degradation
  - Token usage: `prompt_tokens`, `completion_tokens`, `total_tokens`
- Auto-generated output path: `{input_name}_synthetic_completions`

**Degradation Dimensions:**
The script systematically degrades completions across these quality dimensions:
- Factual accuracy (wrong facts, incorrect numbers/dates)
- Logical coherence (contradictory arguments, illogical flow)
- Completeness (missing key parts, unfinished responses)
- Organization/structure (poor flow, confusing order)
- Task focus (irrelevant information, off-topic content)
- Language quality (typos, grammatical errors, unclear phrasing)
- Certainty levels (overconfident about uncertain things)
- Format compliance (ignoring specific format instructions)
- Reasoning quality (faulty logic, wrong assumptions)
- Answer correctness (wrong final answers, no conclusions)

### 2. Overall Position Bias Evaluation

**`eval_position_bias.py`** - Tests position bias when judge selects "best" from multiple options

**Features:**
- Multiple random orderings of the same completions
- Configurable reasoning modes (with/without reasoning)
- Advanced adaptive concurrency with ramp-up
- Statistical analysis of position preferences

**Usage:**
```bash
# Basic evaluation (auto-generated filename)
venv/bin/python eval_position_bias.py data/02-standardised/tulu-3-sft-mixture \
    --max-samples 100 \
    --orderings 20

# With reasoning mode (auto-generated filename)
venv/bin/python eval_position_bias.py data/input \
    --reasoning-mode with_reasoning \
    --max-samples 50 \
    --orderings 10

# Custom output filename
venv/bin/python eval_position_bias.py data/input \
    --output results/custom_bias_test.jsonl \
    --max-samples 100
```

### 2. Pairwise Position Bias Evaluation

**`eval_pairwise_position_bias.py`** - Tests position bias in pairwise comparisons

**Features:**
- More realistic evaluation mirroring actual judge workflows
- Tests specific pairs from larger lists
- Analyzes both absolute positions and relative distance effects
- Question order effects ("A vs B" vs "B vs A")
- Configurable pairs per ordering

**Usage:**
```bash
# Pairwise evaluation (auto-generated filename)
venv/bin/python eval_pairwise_position_bias.py data/input \
    --max-samples 100 \
    --orderings 10 \
    --pairs-per-ordering 5

# Skip reverse question order for faster evaluation (auto-generated filename)
venv/bin/python eval_pairwise_position_bias.py data/input \
    --no-both-orders \
    --pairs-per-ordering 3

# Custom output filename
venv/bin/python eval_pairwise_position_bias.py data/input \
    --output results/custom_pairwise.jsonl \
    --max-samples 100
```

### 3. Analysis Scripts

**`analyze_position_bias.py`** - Comprehensive analysis for overall position bias

**Features:**
- Position preference statistics and visualizations
- Chi-square tests for uniformity
- Bias indicators (primacy, recency, middle bias)
- Consistency analysis across samples
- Reasoning mode comparisons

**`analyze_pairwise_position_bias.py`** - Analysis for pairwise comparisons

**Features:**
- Winner preference analysis (A vs B bias)
- Question order effect analysis
- Position distance effect analysis  
- Absolute position effect analysis
- Consistency analysis for pairs

**Usage:**
```bash
# Analyze overall position bias (auto-generated output directory)
venv/bin/python analyze_position_bias.py results/position_bias_claude_100samples_20orders.jsonl

# Analyze pairwise position bias (auto-generated output directory)
venv/bin/python analyze_pairwise_position_bias.py results/pairwise_bias_qwen_50samples_10orders_5pairs.jsonl

# Custom output directory
venv/bin/python analyze_position_bias.py results/bias_test.jsonl \
    --output analysis/custom_analysis/
```

## Auto-Generated File and Directory Names

Both evaluation and analysis scripts automatically generate meaningful filenames and directories if not specified:

### Evaluation Output Files
**Position Bias:**
- Format: `position_bias_{model}_{samples}samples_{orderings}orders[_{reasoning_mode}].jsonl`
- Examples:
  - `position_bias_qwen_100samples_20orders.jsonl`
  - `position_bias_claude_50samples_10orders_both.jsonl`

**Pairwise Position Bias:**
- Format: `pairwise_bias_{model}_{samples}samples_{orderings}orders_{pairs}pairs[_{reasoning_mode}][_single_order].jsonl`
- Examples:
  - `pairwise_bias_qwen_100samples_10orders_5pairs.jsonl`
  - `pairwise_bias_claude_50samples_20orders_3pairs_both_single_order.jsonl`

### Analysis Output Directories
- Auto-generated from input filename by adding `_analysis` suffix
- Examples:
  - `position_bias_qwen_100samples_20orders.jsonl` → `position_bias_qwen_100samples_20orders_analysis/`
  - `pairwise_bias_claude_50samples_10orders_5pairs.jsonl` → `pairwise_bias_claude_50samples_10orders_5pairs_analysis/`

All files are created in the `judge-analysis/` directory by default.

## Configuration

### Common Parameters

- `--max-samples N`: Limit evaluation to N samples
- `--orderings N`: Number of random orderings per sample (default: 10)
- `--concurrent N`: Maximum concurrent API requests (default: 50)
- `--disable-adaptive`: Disable adaptive concurrency management
- `--model MODEL`: Judge model to use (default: claude-3-5-sonnet-20241022)
- `--reasoning-mode MODE`: `with_reasoning`, `no_reasoning`, or `both`

### Pairwise-Specific Parameters

- `--pairs-per-ordering N`: Number of random pairs to test per ordering (default: 5)
- `--no-both-orders`: Only test "A vs B", skip "B vs A" for faster evaluation

### Environment Variables

Set one of these API keys:
- `SWISSAI_API_KEY`: Swiss AI API key
- `SWISS_AI_API_KEY`: Alternative Swiss AI API key name

## Output Formats

### Position Bias Results (JSONL)
```json
{
  "sample_id": "unique_id",
  "ordering_num": 1,
  "reasoning_mode": "no_reasoning",
  "position_mapping": {"comp_0": 1, "comp_1": 2},
  "judge_choice": "comp_1",
  "chosen_position": 2,
  "success": true,
  "judge_model": "claude-3-5-sonnet-20241022"
}
```

### Pairwise Results (JSONL)
```json
{
  "sample_id": "unique_id",
  "pair_num": 1,
  "completion_a_position": 2,
  "completion_b_position": 5,
  "question_order": "A_vs_B",
  "winner": "B",
  "position_distance": 3,
  "success": true,
  "judge_model": "claude-3-5-sonnet-20241022"
}
```

## Analysis Outputs

Both analysis scripts generate:
- **JSON summary** (`analysis_results.json`): Complete statistical analysis
- **Text report** (`*_bias_report.txt`): Human-readable summary
- **Visualizations** (PNG files): Charts showing bias patterns

### Key Visualizations

**Position Bias:**
- `position_preferences_overall.png`: Overall position preference chart
- `position_preferences_by_reasoning.png`: Preferences by reasoning mode
- `consistency_distribution.png`: Judge consistency across samples

**Pairwise Bias:**
- `pairwise_winner_preferences.png`: A vs B preference rates
- `question_order_effects.png`: Effects of question order
- `position_distance_effects.png`: How distance affects preferences
- `completion_a_position_effects.png`: Absolute position effects

## Understanding Pairwise Position Bias Visualizations

The pairwise analysis generates three key figures that reveal different types of bias:

### 📊 Figure 1: Overall Pairwise Winner Preferences
**What it shows:** Across all comparisons, how often does the judge choose option "A" vs option "B"?

**What to look for:**
- **No bias:** Both A and B around 50%
- **Label bias:** Strong preference for A (or B) regardless of content
- **Example:** A wins 70%, B wins 30% → Judge systematically favors the "A" option

### 🔄 Figure 2: Winner Preferences by Question Order  
**What it shows:** For the same completions, does changing the order they're mentioned in the comparison question affect the outcome?

**The test:** 
- **"Original order":** "Compare completion X and completion Y" 
- **"Reversed order":** "Compare completion Y and completion X"

**What to look for:**
- **No mention order effect:** Both groups show similar patterns
- **Mention order effect:** Different win rates between original and reversed question orders
- **Example:** Original order shows first-mentioned wins 70%, reversed order shows first-mentioned wins 60%

**Key insight:** This tests whether mentioning a completion first in the question ("Compare X and Y") gives it an advantage over mentioning it second ("Compare Y and X").

### 📏 Figure 3: Winner Preferences by Position Distance
**What it shows:** Does the gap between compared items in the original list affect which one wins?

**Distance examples:**
- Distance 1: Comparing adjacent items (positions 3 vs 4)
- Distance 3: Comparing items with 2 positions between (positions 3 vs 6) 
- Distance 7: Comparing items far apart (positions 2 vs 9)

**What to look for:**
- **No distance effect:** Similar win rates across all distances
- **Adjacent bias:** Different behavior for nearby vs distant items
- **Extreme distance bias:** Very different patterns for large gaps
- **Example:** Distance 1-2 show 50/50 split, Distance 6+ show 80% A preference

**Key insight:** Tests whether judges are influenced by the spatial relationship between items, independent of their labels or absolute positions.

## Judge Prompts

The `judge_prompts/` directory contains evaluation prompts:

- `position_bias_with_reasoning.txt`: Multi-choice with detailed reasoning
- `position_bias_no_reasoning.txt`: Multi-choice with simple winner selection
- `pairwise_position_bias_with_reasoning.txt`: Pairwise with detailed reasoning
- `pairwise_position_bias_no_reasoning.txt`: Pairwise with simple winner selection

## Advanced Features

### Adaptive Concurrency

Both evaluation scripts include advanced adaptive concurrency management:

- **Ramp-up strategy**: Gradual increase to target concurrency
- **Little's Law optimization**: Automatic optimal concurrency calculation
- **Immediate failure response**: Quick reduction on API errors
- **Stability zones**: Prevent thrashing during final tasks
- **Real-time metrics**: requests/min, success rate, error rate tracking

### Statistical Analysis

- **Chi-square tests**: Test for uniform distribution
- **Bias indicators**: Quantify primacy, recency, and middle bias
- **Consistency metrics**: Judge reliability across orderings
- **Effect size calculations**: Magnitude of bias effects

## Example Workflows

### Quick Position Bias Check
```bash
# Evaluate 50 samples with 10 orderings each
venv/bin/python eval_position_bias.py data/input \
    --output results/quick_check.jsonl \
    --max-samples 50 --orderings 10

# Analyze results
venv/bin/python analyze_position_bias.py results/quick_check.jsonl
```

### Comprehensive Pairwise Analysis
```bash
# Full pairwise evaluation
venv/bin/python eval_pairwise_position_bias.py data/input \
    --output results/pairwise_full.jsonl \
    --max-samples 200 \
    --orderings 20 \
    --pairs-per-ordering 5 \
    --reasoning-mode both

# Generate detailed analysis
venv/bin/python analyze_pairwise_position_bias.py results/pairwise_full.jsonl \
    --output analysis/pairwise_comprehensive/
```

### Large-Scale Production Evaluation
```bash
# High-volume evaluation with adaptive concurrency
venv/bin/python eval_position_bias.py data/large_dataset \
    --output results/production_bias.jsonl \
    --max-samples 1000 \
    --orderings 50 \
    --concurrent 100 \
    --reasoning-mode no_reasoning
```

## Troubleshooting

### Common Issues

1. **API timeouts**: Reduce `--concurrent` parameter
2. **Memory issues**: Reduce `--max-samples` or process in batches
3. **No successful results**: Check API key and endpoint configuration
4. **Plotting errors**: Install matplotlib and seaborn: `pip install matplotlib seaborn scipy`

### Performance Optimization

- Use `--no-both-orders` for faster pairwise evaluation
- Set `--reasoning-mode no_reasoning` for simpler/faster evaluation
- Enable adaptive concurrency for optimal throughput
- Process large datasets in chunks if memory constrained

## Future Extensions

This framework can be extended to evaluate:
- Content bias (topic, style, length preferences)
- Demographic bias in judge preferences  
- Cross-model judge consistency
- Temporal bias (evaluation order effects)
- Multi-turn conversation judging bias