# Judge LLM Ranking Evaluation Report

## Summary

Model: Qwen/Qwen3-32B
Instructions: prompts/principles.txt
Samples Evaluated: 1
Timestamp: 2025-08-16 22:27:54

### Overall Metrics
- Success Rate: 100.0% (1/1)
- Failed: 0.0% (0/1)
Spearman measures rank order preservation (>0.7 good, >0.9 excellent), Kendall measures pairwise agreement (>0.5 good, >0.7 excellent).
- Mean Spearman Correlation: 0.917 (std: 0.000)
- Mean Kendall's Tau: 0.778 (std: 0.000)
- Median Spearman: 0.917
- Median Kendall's Tau: 0.778

### Position Accuracy
Top-1 measures correctly identifying the best response (giving it rank 9), Top-3 measures having all 3 best responses in ranks 7-9.
- Top-1 Accuracy: 0.0%
- Top-3 Accuracy: 66.7%
- Bottom-1 Accuracy: 0.0%
- Perfect Rankings: 0/1 (0.0%)

### Token Usage
- Total Tokens: 15,569
- Average per Sample: 15569 tokens
  - Prompt: 15551 tokens
  - Completion: 18 tokens

## Detailed Results

### Successful Samples (1 total)

| Sample   | Spearman | Kendall  | Top-3 Acc | Tokens   |
|----------|----------|----------|-----------|----------|
| 0000     |   +0.917 |   +0.778 | 2/3       | 15,569   |

## Error Analysis

### Common Ranking Errors
## Statistical Distribution

### Spearman Correlation Distribution
```
+0.8-+1.0: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (1 samples)
+0.6-+0.8:  (0 samples)
+0.4-+0.6:  (0 samples)
+0.2-+0.4:  (0 samples)
+0.0-+0.2:  (0 samples)
-0.2-+0.0:  (0 samples)
-0.4--0.2:  (0 samples)
-0.6--0.4:  (0 samples)
-0.8--0.6:  (0 samples)
-1.0--0.8:  (0 samples)
```

### Kendall's Tau Distribution
```
+0.8-+1.0:  (0 samples)
+0.6-+0.8: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (1 samples)
+0.4-+0.6:  (0 samples)
+0.2-+0.4:  (0 samples)
+0.0-+0.2:  (0 samples)
-0.2-+0.0:  (0 samples)
-0.4--0.2:  (0 samples)
-0.6--0.4:  (0 samples)
-0.8--0.6:  (0 samples)
-1.0--0.8:  (0 samples)
```