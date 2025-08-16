# Judge LLM Ranking Evaluation Report

## Summary

Model: Qwen/Qwen3-32B
Label Type: numeric
Instructions: judge_instructions/default.txt
Samples Evaluated: 5
Timestamp: 2025-08-14 23:59:18

### Overall Metrics
- Success Rate: 100.0% (5/5)
- Failed: 0.0% (0/5)
Spearman measures rank order preservation (>0.7 good, >0.9 excellent), Kendall measures pairwise agreement (>0.5 good, >0.7 excellent).
- Mean Spearman Correlation: -0.127 (std: 0.249)
- Mean Kendall's Tau: -0.056 (std: 0.189)
- Median Spearman: -0.017
- Median Kendall's Tau: 0.000

### Position Accuracy
Top-1 measures correctly identifying the best response, Top-3 measures having all 3 best responses in the top 3 positions.
- Top-1 Accuracy: 40.0%
- Top-3 Accuracy: 26.7%
- Bottom-1 Accuracy: 60.0%
- Perfect Rankings: 0/5 (0.0%)

### Token Usage
- Total Tokens: 670,065
- Average per Sample: 134013 tokens
  - Prompt: 119213 tokens
  - Completion: 14800 tokens

## Detailed Results

### Successful Samples (5 total)

| Sample   | Spearman | Kendall  | Top-3 Acc | Tokens   |
|----------|----------|----------|-----------|----------|
| 0000     |   -0.567 |   -0.389 | 0/3       | 162,328  |
| 0001     |   +0.167 |   +0.167 | 2/3       | 104,468  |
| 0002     |   -0.200 |   -0.111 | 1/3       | 51,367   |
| 0003     |   -0.017 |   +0.056 | 1/3       | 209,271  |
| 0004     |   -0.017 |   +0.000 | 0/3       | 142,631  |

## Error Analysis

### Common Ranking Errors
**Adjacent Swaps:**
- Positions 4-5: 1 occurrences

## Statistical Distribution

### Spearman Correlation Distribution
```
+0.8-+1.0:  (0 samples)
+0.6-+0.8:  (0 samples)
+0.4-+0.6:  (0 samples)
+0.2-+0.4:  (0 samples)
+0.0-+0.2: ▓▓▓▓▓ (1 samples)
-0.2-+0.0: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (3 samples)
-0.4--0.2:  (0 samples)
-0.6--0.4: ▓▓▓▓▓ (1 samples)
-0.8--0.6:  (0 samples)
-1.0--0.8:  (0 samples)
```

### Kendall's Tau Distribution
```
+0.8-+1.0:  (0 samples)
+0.6-+0.8:  (0 samples)
+0.4-+0.6:  (0 samples)
+0.2-+0.4:  (0 samples)
+0.0-+0.2: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (3 samples)
-0.2-+0.0: ▓▓▓▓▓ (1 samples)
-0.4--0.2: ▓▓▓▓▓ (1 samples)
-0.6--0.4:  (0 samples)
-0.8--0.6:  (0 samples)
-1.0--0.8:  (0 samples)
```