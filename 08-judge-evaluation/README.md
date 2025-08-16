# Judge Evaluation Framework

This framework provides tools for evaluating LLM judge models on synthetic preference datasets. It includes a complete evaluation pipeline with dataset loading, metrics calculation, retry mechanisms, and comprehensive reporting.

## Quick Start

### Basic Usage
```bash
# Test with 3 samples (development)
venv/bin/python 08-judge-evaluation/judge_llm_ranking.py --samples 3

# Full evaluation on all samples
venv/bin/python 08-judge-evaluation/judge_llm_ranking.py --samples -1

# Use custom judge instructions
venv/bin/python 08-judge-evaluation/judge_llm_ranking.py --samples 10 \
    --instructions judge_instructions/charter.txt
```

### Advanced Options
```bash
# Alphabetic labels instead of numeric
venv/bin/python 08-judge-evaluation/judge_llm_ranking.py --samples 100 \
    --label-type alphabetic --concurrent 20 --max-retries 3
```

## Architecture

The framework uses a modular architecture with reusable components:

### Core Library (`lib.py`)
- **`SyntheticDatasetLoader`** - Dataset loading and preparation
- **`InstructionsLoader`** - Judge instructions management  
- **`EvaluationAnalyzer`** - Metrics calculation and error analysis
- **`ReportGenerator`** - Comprehensive markdown report generation
- **`BaseJudgeEvaluator`** - Common evaluation framework with retry logic
- **`JudgeEvaluationUtils`** - File naming and utility functions

### Judge-Specific Scripts
- **`judge_llm_ranking.py`** - Ranking evaluation (current implementation)
- **`judge_llm_ranking_standalone.py`** - Backup of original monolithic version

## Implementing New Judge Methods

Creating a new judge evaluation script is straightforward using the provided framework. Follow this step-by-step guide:

### Step 1: Create the Script File

Create a new file following the naming convention: `judge_<method_name>.py`

```bash
touch 08-judge-evaluation/judge_pairwise.py
```

### Step 2: Basic Script Template

```python
#!/usr/bin/env python3
"""
Judge Pairwise Comparison Evaluation

Evaluates a judge model's ability to compare pairs of completions.
"""

import sys
import os
import re
import random
import argparse
import asyncio
from typing import Dict, Any, List, Optional, Tuple

# Add lib directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib import (
    SyntheticDatasetLoader, EvaluationAnalyzer,
    ReportGenerator, BaseJudgeEvaluator, JudgeEvaluationUtils
)

# Configuration
DATASET_PATH = "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-preference-quality-short-1100-synthetic"
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_MAX_RETRIES = 2

class JudgePairwiseEvaluator(BaseJudgeEvaluator):
    """Evaluates judge model's pairwise comparison ability."""
    
    def __init__(self, model: str = DEFAULT_MODEL, 
                 instructions_path: str = "judge_instructions/default.txt", 
                 max_retries: int = DEFAULT_MAX_RETRIES):
        super().__init__(model, instructions_path, max_retries)
        # Add any method-specific initialization here
    
    def create_prompt(self, sample: Dict, pair_data: Dict) -> str:
        """Create pairwise comparison prompt."""
        # YOUR PROMPT LOGIC HERE
        # Use self.judge_instructions for loaded instructions
        # Return formatted prompt string
        pass
    
    def parse_response(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse response from model. Returns (result, error_detail)."""
        # YOUR PARSING LOGIC HERE
        # Return (parsed_result, None) on success
        # Return (None, error_description) on failure
        pass
    
    async def _evaluate_sample_once(self, sample: Dict) -> Dict:
        """Evaluate a single sample (single attempt)."""
        # YOUR EVALUATION LOGIC HERE
        # This is where you implement the specific judging method
        
        # 1. Prepare the specific evaluation format (pairs, scoring, etc.)
        # 2. Create prompt using self.create_prompt()
        # 3. Make API call using self.client
        # 4. Parse response using self.parse_response()
        # 5. Calculate metrics specific to your method
        # 6. Return standardized result dictionary
        
        try:
            # Your implementation here
            # Must return a dict with keys:
            # - sample_id, success, error, error_detail (if failed)
            # - Plus your method-specific results (if successful)
            pass
        except Exception as e:
            return {
                "sample_id": sample['id'],
                "success": False,
                "error": "API Error",
                "error_detail": str(e),
                "raw_response": None,
                "tokens": {"prompt": 0, "completion": 0, "total": 0}
            }

async def main():
    parser = argparse.ArgumentParser(description="Evaluate judge pairwise comparison ability")
    parser.add_argument("--samples", type=int, default=3, 
                       help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help="Model to use for evaluation")
    parser.add_argument("--concurrent", type=int, default=50,
                       help="Maximum concurrent API requests")
    parser.add_argument("--instructions", type=str, default="judge_instructions/default.txt",
                       help="Path to judge instructions file")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                       help="Maximum number of retries for failed samples")
    # Add method-specific arguments here
    
    args = parser.parse_args()
    
    # Initialize components
    loader = SyntheticDatasetLoader(DATASET_PATH)
    analyzer = EvaluationAnalyzer()  # You may need to extend this for method-specific metrics
    reporter = ReportGenerator()     # You may need to extend this for method-specific reports
    utils = JudgeEvaluationUtils()
    
    evaluator = JudgePairwiseEvaluator(
        model=args.model, 
        instructions_path=args.instructions, 
        max_retries=args.max_retries
    )
    
    # Load dataset and evaluate
    print(f"Loading dataset...")
    samples = loader.load_dataset(args.samples)
    print(f"Loaded {len(samples)} samples")
    
    results = await evaluator.evaluate_all(samples, max_concurrent=args.concurrent)
    metrics = analyzer.calculate_overall_metrics(results)
    
    # Print final results
    print("\n=== FINAL RESULTS ===")
    print(f"Success Rate: {metrics['success_rate']:.1%} ({metrics['successful']}/{metrics['total_samples']})")
    print(f"Failed: {metrics['failed_rate']:.1%} ({metrics['failed']}/{metrics['total_samples']})")
    
    # Generate output files
    base_name = f"judge_pairwise_{args.model.split('/')[-1]}_{utils.generate_output_filename(args.model, 'pairwise', args.instructions, len(samples))}"
    output_dir = utils.create_output_directory()
    
    # Save results and generate report
    # ... (same pattern as ranking script)

if __name__ == "__main__":
    import json
    asyncio.run(main())
```

### Step 3: Implement Core Methods

#### 3.1 Implement `create_prompt()`
This method should:
- Use `self.judge_instructions` for loaded instructions
- Format your specific evaluation task (pairwise, scoring, etc.)
- Include clear formatting for model responses
- Return a complete prompt string

#### 3.2 Implement `parse_response()`
This method should:
- Parse the model's response format
- Validate the response structure
- Return `(parsed_result, None)` on success
- Return `(None, error_description)` on failure

#### 3.3 Implement `_evaluate_sample_once()`
This method should:
- Prepare sample-specific data (pairs, presentation order, etc.)
- Call `create_prompt()` to generate the prompt
- Make API call using `self.client.chat.completions.create()`
- Parse response using `parse_response()`
- Calculate method-specific metrics
- Return standardized result dictionary

### Step 4: Extend Analysis (Optional)

If your method needs custom metrics, extend the analyzer:

```python
class PairwiseEvaluationAnalyzer(EvaluationAnalyzer):
    def calculate_pairwise_metrics(self, results: List[Dict]) -> Dict:
        """Calculate pairwise-specific metrics."""
        # Your custom metric calculations
        pass
```

### Step 5: Extend Reporting (Optional)

If your method needs custom reports, extend the reporter:

```python
class PairwiseReportGenerator(ReportGenerator):
    def _add_pairwise_analysis(self, report: List[str], results: List[Dict], metrics: Dict):
        """Add pairwise-specific analysis to report."""
        # Your custom report sections
        pass
```

## Common Patterns

### Dataset Structure
All samples have this structure:
```python
{
    'id': '0001',
    'conversation_id': 'original_id',
    'question': 'The user question',
    'completions': {
        0: {'content': '...', 'iteration': 0, 'degradation_rank': 0},  # Best quality
        1: {'content': '...', 'iteration': 1, 'degradation_rank': 1},  # Slightly degraded
        # ...
        8: {'content': '...', 'iteration': 8, 'degradation_rank': 8}   # Most degraded
    }
}
```

### Standard Result Format
All evaluation methods should return results with these keys:
```python
{
    "sample_id": str,
    "success": bool,
    "error": Optional[str],           # If failed
    "error_detail": Optional[str],    # If failed
    "raw_response": Optional[str],
    "tokens": {"prompt": int, "completion": int, "total": int},
    # Plus method-specific results...
}
```

### Error Handling
- Use the retry mechanism (inherited from `BaseJudgeEvaluator`)
- Provide detailed error messages in `error_detail`
- Categorize errors consistently
- Always return a valid result dictionary

### Progress Tracking
- Progress bars are handled automatically by `BaseJudgeEvaluator`
- Focus on implementing the single-sample evaluation logic

## Examples of Judge Methods

### 1. Ranking (Implemented)
- **Input**: 9 completions to rank
- **Output**: Complete ranking from 1-9
- **Metrics**: Spearman correlation, Kendall's tau, position accuracy

### 2. Pairwise Comparison (Template above)
- **Input**: Multiple pairs of completions
- **Output**: Winner for each pair (A or B)
- **Metrics**: Pairwise accuracy, transitivity consistency

### 3. Scoring
- **Input**: Individual completions
- **Output**: Quality score (1-9 scale)
- **Metrics**: Score correlation with ground truth, score distribution

### 4. Single Token Prediction
- **Input**: Prompt asking for best completion
- **Output**: Single token (1-9)
- **Metrics**: Selection accuracy, top-k accuracy

## Judge Instructions

Create modular instruction files in `judge_instructions/`:

- **`default.txt`** - General helpful/honest/harmless principles
- **`charter.txt`** - Constitutional AI charter-based evaluation
- **`concise.txt`** - Brevity-focused evaluation
- **Custom instructions** - Domain-specific evaluation criteria

Additional prompts for bias testing are available in `prompts/`.

## Best Practices

1. **Reuse the library** - Don't reimplement common functionality
2. **Consistent naming** - Follow `judge_<method>_<model>_<config>_<samples>samples` pattern  
3. **Clear error messages** - Help debug parsing and API issues
4. **Document metrics** - Explain what your custom metrics mean
5. **Test thoroughly** - Use small sample sizes during development
6. **Handle edge cases** - Empty responses, malformed JSON, etc.

## Testing Your Implementation

```bash
# Start with 1 sample for basic functionality
venv/bin/python 08-judge-evaluation/judge_yourmethod.py --samples 1

# Test error handling with various configurations
venv/bin/python 08-judge-evaluation/judge_yourmethod.py --samples 3 --max-retries 0

# Scale up gradually
venv/bin/python 08-judge-evaluation/judge_yourmethod.py --samples 10
venv/bin/python 08-judge-evaluation/judge_yourmethod.py --samples 100
```

## Output Files

All judge scripts generate output in the `analysis/` directory:
- **`judge_<method>_<config>.jsonl`** - Raw results for further analysis
- **`judge_<method>_<config>.md`** - Comprehensive evaluation report

Reports include:
- Success rates and error analysis  
- Method-specific metrics and distributions
- Token usage statistics
- Sample-by-sample breakdown
- Statistical visualizations

## Getting Help

- Check `judge_llm_ranking.py` for a complete working example
- Review `lib.py` for available utilities
- Use `judge_llm_ranking_standalone.py` to see the original monolithic version
- Test incrementally with small sample sizes during development