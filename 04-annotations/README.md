# Dataset Annotations

Adds annotations to chat format datasets including language detection and LLM-based classifications.

## Environment Setup
```bash
export SWISSAI_API_KEY="your_api_key"  # For LLM classification
```

## Refusal Classification

LLM-based classification to identify assistant refusal responses with **adaptive concurrency** and retry logic.

```bash
# Adaptive concurrency (default) - automatically adjusts based on API performance
venv/bin/python 05-annotations/classify_refusal.py \
  data/02-standardised/dataset-name \
  --output data/05-annotations/dataset-name-refusal

# High throughput processing
venv/bin/python 05-annotations/classify_refusal.py \
  data/02-standardised/EuroBlocks-SFT-Synthetic-1124 \
  --output data/05-annotations/EuroBlocks-SFT-Synthetic-1124-refusal \
  --concurrent 100 --chunk-size 50000

# Fixed concurrency mode (disable adaptive)
venv/bin/python 05-annotations/classify_refusal.py \
  data/02-standardised/dataset-name \
  --output data/05-annotations/dataset-name-refusal \
  --concurrent 50 --disable-adaptive
```

**Features**: 
- Auto-adjusts concurrency every 60s (increases +20 if error rate <1%)
- 3 retries with exponential backoff for failed requests
- Real-time metrics: req/min, error rate, avg duration
- Progress persistence with `--resume`

## Language Detection

FastText-based language detection for message content (no API required).

```bash
# Detect languages in all messages
venv/bin/python 05-annotations/annotate_language.py \
  data/02-standardised/dataset-name \
  --output data/05-annotations/dataset-name-lang

# Detect only in assistant messages
venv/bin/python 05-annotations/annotate_language.py \
  data/02-standardised/dataset-name \
  --output data/05-annotations/dataset-name-lang \
  --target-roles assistant

# Process with custom chunk size
venv/bin/python 05-annotations/annotate_language.py \
  data/02-standardised/dataset-name \
  --output data/05-annotations/dataset-name-lang \
  --chunk-size 20000
```

## Output Format

Classifications are added to message metadata:
```json
{
  "role": "assistant",
  "content": "I can't help with that request...",
  "metadata": {
    "refusal_classification": {
      "classification": "refusal",
      "reasoning": "Assistant explicitly declines the request...",
      "success": true,
      "timestamp": 1672531200.0,
      "model": "meta-llama/Llama-3.3-70B-Instruct"
    }
  }
}
```