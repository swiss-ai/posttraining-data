# 04-Annotations Implementation Guide

This guide provides comprehensive instructions for implementing and extending the annotation system in the 04-annotations folder.

## Overview

The 04-annotations system provides tools for adding various classifications and annotations to chat format datasets using LLMs and local models. The system follows a consistent pattern of loading datasets, applying annotations, and saving to new directories with updated metadata.

## Architecture

### Core Components

1. **lib/llm_classifier.py** - Shared library for LLM-based classification
   - Adaptive concurrency system using task set control loop pattern
   - Retry logic with exponential backoff
   - Token tracking and metrics collection
   - Swiss AI API integration

2. **Classification Scripts**
   - `classify_refusal.py` - Identifies refusal responses in assistant messages
   - `classify_ideology.py` - Scores ideological sensitivity of initial prompts (0-3)
   - `language_annotate.py` - FastText-based language detection (no API needed)

3. **Prompt Templates** in `prompts/`
   - Each classification type has its own prompt template
   - Templates use placeholders that get filled during processing

## Metadata Structure

### Refusal Classification (per message in conversation branches)
```json
{
  "role": "assistant",
  "content": "...",
  "metadata": {
    "refusal_classification": {
      "classification": "refusal",
      "reasoning": "...",
      "success": true,
      "timestamp": 1234567890.0,
      "model": "meta-llama/Llama-3.3-70B-Instruct"
    }
  }
}
```

### Ideological Classification (on initial_prompt)
```json
{
  "initial_prompt": {
    "role": "user",
    "content": "...",
    "metadata": {
      "ideological_classification": {
        "meta-llama/Llama-3.3-70B-Instruct": {
          "classification": 2,
          "reasoning": "..."
        }
      }
    }
  }
}
```

### Language Classification (on any message)
```json
{
  "metadata": {
    "language_classification": {
      "primary_language": "en",
      "primary_confidence": 0.99,
      "top_languages": [
        {"language": "en", "confidence": 0.99},
        {"language": "de", "confidence": 0.01}
      ],
      "timestamp": "2025-01-27T...",
      "model": "fasttext-lid.176",
      "top_k": 3
    }
  }
}
```

## Implementing a New Classifier

### Step 1: Create the Prompt Template

Create a new file in `prompts/your_classification.txt`:

```text
You are analyzing [what you're analyzing] to [purpose].

TASK: [Clear description of the classification task]

[CLASSIFICATION CRITERIA AND GUIDELINES]

OUTPUT FORMAT: Respond with ONLY valid JSON in this exact format:
{{
  "reasoning": "Explanation for the classification (MUST be in English)",
  "classification": "category1|category2|..."
}}

Do not include any text before or after the JSON.

[PLACEHOLDERS FOR INPUT DATA]
```

### Step 2: Create the Classification Script

Base your script on `classify_refusal.py` or `classify_ideology.py` depending on whether you're:
- Classifying multiple messages per sample → use refusal pattern
- Classifying once per sample → use ideology pattern

Key functions to implement:

```python
def collect_classification_tasks(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract items to classify from a sample."""
    # Return list of tasks with required data
    
def apply_classification_results(tasks, results, model) -> List[Dict[str, Any]]:
    """Apply classification results to samples."""
    # Update samples with classification metadata
```

### Step 3: Handle Metadata Structure

For message-level classification (like refusal):
```python
# Simple structure under the classification name
message["metadata"]["your_classification"] = {
    "classification": result.classification,
    "reasoning": result.reasoning,
    "success": result.success,
    "timestamp": result.timestamp,
    "model": model
}
```

For sample-level classification (like ideology):
```python
# Nested structure: classification_name -> model -> data
if "your_classification" not in sample["initial_prompt"]["metadata"]:
    sample["initial_prompt"]["metadata"]["your_classification"] = {}

sample["initial_prompt"]["metadata"]["your_classification"][model] = {
    "classification": classification_value,
    "reasoning": result.reasoning
}
```

## Using the Adaptive Concurrency System

The LLMClassifier provides adaptive concurrency by default:

```python
# Initialize with adaptive enabled (default)
classifier = LLMClassifier(api_key, model, concurrent=50, adaptive=True)

# Or with fixed concurrency
classifier = LLMClassifier(api_key, model, concurrent=50, adaptive=False)
```

### How Adaptive Concurrency Works

1. **Task Set Control Loop**: Uses asyncio task sets instead of semaphores
2. **Dynamic Adjustment**: 
   - Immediate: Reduces by 1 on any failure
   - Periodic (60s): Increases by 20 if 0% error rate
3. **Little's Law**: Calculates optimal concurrency = (requests/min × avg_latency) / 60
4. **Chunk Stabilization**: Resets adaptation timer between chunks

### Monitoring Performance

The system displays real-time metrics:
- `req/min`: Request throughput
- `success`: Success rate percentage
- `failed`: Number of failed requests
- `avg_dur`: Average request duration
- `error_rate`: Current error rate
- `concurrent`: Current concurrency level

## Command Line Usage

### Basic Classification
```bash
# Refusal classification
venv/bin/python 04-annotations/classify_refusal.py \
  data/02-standardised/dataset-name \
  --output data/04-annotations/dataset-name-refusal

# Ideological classification  
venv/bin/python 04-annotations/classify_ideology.py \
  data/02-standardised/dataset-name \
  --output data/04-annotations/dataset-name-ideology
```

### Advanced Options
```bash
# High concurrency with large chunks
--concurrent 600 --chunk-size 50000

# Resume interrupted processing
--resume

# Restart from beginning
--restart

# Disable adaptive concurrency
--disable-adaptive

# Use different model
--model "Qwen/Qwen3-32B"
```

## Progress Persistence

The system saves progress after each chunk:
- Progress files: `.progress_model_name.json`
- Allows resuming interrupted runs with `--resume`
- Clear progress with `--restart`

## Error Handling

### API Errors
- Exponential backoff: 4s, 8s, 16s, 32s, 60s...
- Up to 4 total attempts (initial + 3 retries)
- 25% jitter added to prevent thundering herd

### Classification Failures
Failed classifications are marked in metadata:
```json
{
  "classification": "inconclusive",
  "reasoning": "Failed after 4 attempts: error details",
  "success": false,
  "error": "specific error message"
}
```

## Processing Pipeline Integration

Each classifier:
1. Loads dataset from input path
2. Processes in configurable chunks
3. Saves to new output directory
4. Updates `dataset_metadata.json` with processing log:

```json
{
  "processing_log": [{
    "operation": "your_classification",
    "script": "classify_your.py",
    "timestamp": "ISO datetime",
    "input_path": "...",
    "output_path": "...",
    "model": "...",
    "samples_processed": 1000,
    "classification_success": 990,
    "classification_failed": 10,
    "total_tokens_used": 123456
  }]
}
```

## Best Practices

1. **Chunk Size**: Default 10,000 is good for most cases. Increase for simple classifications, decrease for complex ones.

2. **Concurrency**: Start with 50-100. The system will adapt, but starting closer to optimal reduces convergence time.

3. **Model Selection**: 
   - Llama-3.3-70B-Instruct: Good default, balanced performance
   - Qwen3-32B: Faster, good for simpler classifications

4. **Prompt Design**:
   - Be explicit about output format
   - Include examples in the prompt
   - Specify "MUST be in English" for reasoning
   - Use clear category names

5. **Testing**: Always test on a small dataset first:
   ```bash
   --chunk-size 100 --concurrent 5
   ```

## Troubleshooting

### Common Issues

1. **"SWISSAI_API_KEY environment variable is required"**
   ```bash
   export SWISSAI_API_KEY="your_key"
   ```

2. **High error rates**
   - Check API quota/limits
   - Reduce initial concurrency
   - Verify prompt template syntax

3. **Memory issues with large datasets**
   - Reduce chunk size
   - The system processes chunks independently

4. **Slow processing**
   - Increase concurrency if error rate is 0%
   - Check compute node warnings at startup

## Future Extensions

When adding new classifiers:

1. **Assistant Classification** (`classify_assistant.py`)
   - Target: Assistant messages
   - Categories: ai_assistant_related, non_ai_assistant

2. **Quality Classification** (`classify_quality.py`)
   - Target: User/system messages  
   - Dimensions: well_formedness, clarity, scope, completeness
   - Scores: 1-3 per dimension

3. **Custom Classifications**
   - Follow the established patterns
   - Consider metadata structure carefully
   - Test thoroughly before large-scale runs

## Environment Requirements

- Python 3.8+
- Dependencies in requirements.txt
- SWISSAI_API_KEY environment variable
- For language detection: FastText model (auto-downloads)

## Performance Considerations

- Each classification typically uses 200-500 tokens per item
- Processing speed depends on model and concurrency
- Expect 50-200 classifications per minute with default settings
- Adaptive system typically converges to optimal within 2-3 minutes