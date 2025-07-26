# Dataset Annotations

Adds LLM-based classifications and annotations to chat format datasets. Currently supports refusal detection for assistant messages.

## Usage

### Environment Setup
```bash
# Ensure SWISSAI_API_KEY is set
export SWISSAI_API_KEY="your_api_key"
```

### Refusal Classification
Classifies assistant messages to identify refusal responses where the assistant declines to provide information due to safety, ethical, or capability constraints.

```bash
# Basic refusal classification (run from project root)
venv/bin/python 04-annotations/classify_refusal.py data/02-standardised/tulu-3-sft-mixture \
  --output data/04-annotations/tulu-3-sft-mixture-refusal

# With custom settings
venv/bin/python 04-annotations/classify_refusal.py data/02-standardised/smoltalk \
  --output data/04-annotations/smoltalk-refusal \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --concurrent 100 \
  --chunk-size 5000

# Resume interrupted processing
venv/bin/python 04-annotations/classify_refusal.py data/02-standardised/smoltalk \
  --output data/04-annotations/smoltalk-refusal \
  --resume

# Restart from beginning
venv/bin/python 04-annotations/classify_refusal.py data/02-standardised/smoltalk \
  --output data/04-annotations/smoltalk-refusal \
  --restart
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