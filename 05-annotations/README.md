# 04-annotations: Dataset Annotation Tools

LLM-based classification tools for annotating chat format datasets with refusal detection, ideological sensitivity, AI assistant identification, and question quality assessment.

## Environment Setup
```bash
export SWISSAI_API_KEY="your_swiss_ai_api_key"
```

Always use the project virtual environment:
```bash
venv/bin/python 04-annotations/classify_refusal.py --help
```

## Available Classifications

### 1. Refusal Classification
Identifies assistant messages that decline requests due to safety/ethical constraints.

```bash
venv/bin/python 04-annotations/classify_refusal.py \
  data/02-standardised/dataset-name \
  --output data/04-annotations/dataset-name-refusal
```

### 2. Ideological Classification  
Scores ideological sensitivity of initial prompts (0-3 scale).

```bash
venv/bin/python 04-annotations/classify_ideology.py \
  data/02-standardised/dataset-name \
  --output data/04-annotations/dataset-name-ideology
```

### 3. Assistant Classification
Identifies content involving AI assistants or language models.

```bash
venv/bin/python 04-annotations/classify_assistant.py \
  data/02-standardised/dataset-name \
  --output data/04-annotations/dataset-name-assistant
```

### 4. Quality Classification
Evaluates question quality across 4 dimensions (1-3 scoring each).

```bash
venv/bin/python 04-annotations/classify_quality.py \
  data/02-standardised/dataset-name \
  --output data/04-annotations/dataset-name-quality
```

### 5. Language Detection
FastText-based language detection (176 languages, offline processing).

```bash
venv/bin/python 05-annotations/language_annotate.py \
  data/02-standardised/dataset-name \
  data/05-annotations/dataset-name-lang
```

### 6. Boxed Answer Extraction
Extracts \\boxed{answer} patterns from assistant messages, commonly found in mathematical reasoning datasets. Removes the \\boxed{} wrapper and stores extracted answers in metadata.

```bash
# Auto-generate output name with -boxedRemoved suffix
venv/bin/python 05-annotations/extract_boxed_answers.py \
  data/04-decontaminated/math-dataset \
  --output data/05-annotations/

# Custom output name
venv/bin/python 05-annotations/extract_boxed_answers.py \
  data/04-decontaminated/math-dataset \
  --output data/05-annotations/math-dataset-extracted \
  --batch-size 50000 \
  --num-proc 8
```

### 7. System Prompt Fix (smoltalk2 datasets)
Moves system prompts from `original_metadata.chat_template_kwargs.custom_instructions` to the top-level `system_prompt` field as required by the standardized chat format schema. This is specifically needed for smoltalk2 datasets where system prompts were not properly extracted during conversion.

```bash
# Auto-generate output name with -systemPromptFix suffix
venv/bin/python 05-annotations/fix_system_prompts.py \
  data/04-decontaminated/smoltalk2-dataset \
  --output data/05-annotations/

# Custom output name
venv/bin/python 05-annotations/fix_system_prompts.py \
  data/04-decontaminated/smoltalk2-dataset \
  --output data/05-annotations/dataset-fixed \
  --batch-size 50000 \
  --num-proc 8
```

### 8. Thinking Tag Extraction (smoltalk2 datasets)
Converts smoltalk2 think datasets to the new augmented format with parts structure. Extracts `<think>...</think>` tags from assistant messages and converts them to separate "thought" parts, while preserving the cleaned response text. Also handles tool use, function calls, and verifiable responses if present.

```bash
# Auto-generate output name with -thinkPromoted suffix
venv/bin/python 05-annotations/convert_smoltalk2_think.py \
  data/04-decontaminated/smoltalk2-think-dataset \
  --output data/05-annotations/

# Custom output name
venv/bin/python 05-annotations/convert_smoltalk2_think.py \
  data/04-decontaminated/smoltalk2-think-dataset \
  --output data/05-annotations/dataset-think-converted \
  --batch-size 10000 \
  --num-proc 8
```

## Chained Processing
Classifications can be chained by using output as input:

```bash
venv/bin/python 04-annotations/classify_refusal.py \
  data/02-standardised/dataset-name \
  --output data/04-annotations/dataset-name-refusal

venv/bin/python 04-annotations/classify_quality.py \
  data/04-annotations/dataset-name-refusal \
  --output data/04-annotations/dataset-name-refusal-quality
```

## Metadata Structure

### Simple Classifications (Refusal, Ideology, Assistant)
```json
{
  "metadata": {
    "refusal_classification": {
      "meta-llama/Llama-3.3-70B-Instruct": {
        "classification": "no_refusal",
        "reasoning": "Assistant provides helpful information."
      }
    }
  }
}
```

### Quality Classification (4 Dimensions)
```json
{
  "metadata": {
    "quality_classification": {
      "meta-llama/Llama-3.3-70B-Instruct": {
        "well_formedness": {"reasoning": "...", "score": 3},
        "clarity_of_intent": {"reasoning": "...", "score": 3},
        "answerable_scope": {"reasoning": "...", "score": 3},
        "completeness": {"reasoning": "...", "score": 3}
      }
    }
  }
}
```

### Boxed Answer Extraction
```json
{
  "metadata": {
    "verifiable_answer": ["x^2 + 1", "42"]
  }
}
```

## BaseClassifier Framework

All tools use a common framework providing:
- **Incremental saving**: Results saved after each chunk
- **Adaptive concurrency**: Automatic optimization based on API performance  
- **Progress tracking**: Resume interrupted processing with `--resume`
- **Consistent metadata**: Nested structure supporting multiple models
- **Error handling**: 3 retries with exponential backoff

### Creating New Classifiers
```python
from base_classifier import BaseClassifier

class MyClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(
            classifier_name="my_classification",  # Metadata field name
            template_filename="my_template.txt",   # Prompt template in prompts/
            valid_categories=["cat1", "cat2"],     # Expected LLM responses ([] for structured)
            description="My classifier"            # Help text description
        )
    
    def collect_tasks(self, sample):
        """Extract classification tasks from a chat sample.
        
        Return list of dicts with template placeholders:
        [{"sample": sample, "content": "...", "context": "..."}]
        """
        # Implementation here (~10 lines)
        pass
        
    def apply_results(self, tasks, results, model):
        """Apply LLM results back to sample metadata.
        
        Create nested structure: classification_name.model.{classification, reasoning}
        """
        # Implementation here (~10 lines)
        pass

# Usage
def main():
    classifier = MyClassifier()
    return classifier.run_classification()
```