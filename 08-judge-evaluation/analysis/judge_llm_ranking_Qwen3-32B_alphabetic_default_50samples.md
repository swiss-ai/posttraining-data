# Judge LLM Ranking Evaluation Report

## Summary

Model: Qwen/Qwen3-32B
Label Type: alphabetic
Instructions: judge_instructions/default.txt
Samples Evaluated: 50
Timestamp: 2025-08-14 13:26:29

### Overall Metrics
- Success Rate: 94.0% (47/50)
- Failed: 6.0% (3/50)

Error Breakdown:
  - No Ranking Found: 3

Spearman measures rank order preservation (>0.7 good, >0.9 excellent), Kendall measures pairwise agreement (>0.5 good, >0.7 excellent).
- Mean Spearman Correlation: 0.048 (std: 0.341)
- Mean Kendall's Tau: 0.034 (std: 0.258)
- Median Spearman: 0.083
- Median Kendall's Tau: 0.056

### Position Accuracy
Top-1 measures correctly identifying the best response, Top-3 measures having all 3 best responses in the top 3 positions.
- Top-1 Accuracy: 4.3%
- Top-3 Accuracy: 34.0%
- Bottom-1 Accuracy: 17.0%
- Perfect Rankings: 0/47 (0.0%)

### Token Usage
- Total Tokens: 484,974
- Average per Sample: 9699 tokens
  - Prompt: 8169 tokens
  - Completion: 1531 tokens

## Detailed Results

### Failed Samples (3 total)

| Sample ID | Error Type | Detailed Reason | Tokens |
|-----------|------------|-----------------|--------|
| 0020 | Failed to parse valid ranking | No RANKING: [...] found in response | 9,073 |
| 0021 | Failed to parse valid ranking | No RANKING: [...] found in response | 7,006 |
| 0040 | Failed to parse valid ranking | No RANKING: [...] found in response | 6,195 |

### Successful Samples (47 total)

| Sample   | Spearman | Kendall  | Top-3 Acc | Tokens   |
|----------|----------|----------|-----------|----------|
| 0000     |   -0.367 |   -0.278 | 0/3       | 13,424   |
| 0001     |   +0.150 |   +0.111 | 1/3       | 8,675    |
| 0002     |   +0.400 |   +0.389 | 2/3       | 5,019    |
| 0003     |   -0.267 |   -0.167 | 1/3       | 17,462   |
| 0004     |   +0.817 |   +0.667 | 2/3       | 11,591   |
| 0005     |   -0.483 |   -0.333 | 1/3       | 9,389    |
| 0006     |   +0.333 |   +0.222 | 2/3       | 9,609    |
| 0007     |   -0.250 |   -0.167 | 0/3       | 10,259   |
| 0008     |   -0.200 |   -0.111 | 1/3       | 6,567    |
| 0009     |   +0.533 |   +0.389 | 2/3       | 5,708    |
| 0010     |   -0.083 |   -0.111 | 1/3       | 9,785    |
| 0011     |   +0.333 |   +0.278 | 2/3       | 11,660   |
| 0012     |   +0.200 |   +0.056 | 1/3       | 14,105   |
| 0013     |   +0.400 |   +0.278 | 1/3       | 10,291   |
| 0014     |   +0.483 |   +0.333 | 2/3       | 8,566    |
| 0015     |   -0.567 |   -0.444 | 0/3       | 8,102    |
| 0016     |   +0.267 |   +0.222 | 1/3       | 19,710   |
| 0017     |   +0.083 |   +0.111 | 1/3       | 5,490    |
| 0018     |   +0.067 |   +0.056 | 2/3       | 11,188   |
| 0019     |   -0.317 |   -0.278 | 0/3       | 5,677    |
| 0022     |   -0.550 |   -0.333 | 0/3       | 13,711   |
| 0023     |   -0.083 |   -0.111 | 0/3       | 8,304    |
| 0024     |   -0.083 |   -0.056 | 0/3       | 8,784    |
| 0025     |   +0.483 |   +0.333 | 1/3       | 8,407    |
| 0026     |   +0.233 |   +0.167 | 2/3       | 9,037    |
| 0027     |   +0.567 |   +0.444 | 2/3       | 6,091    |
| 0028     |   +0.133 |   +0.111 | 1/3       | 6,686    |
| 0029     |   -0.350 |   -0.333 | 1/3       | 14,913   |
| 0030     |   +0.250 |   +0.222 | 1/3       | 8,429    |
| 0031     |   +0.400 |   +0.167 | 1/3       | 10,894   |
| 0032     |   +0.283 |   +0.222 | 2/3       | 6,937    |
| 0033     |   -0.083 |   -0.056 | 1/3       | 13,488   |
| 0034     |   +0.200 |   +0.167 | 1/3       | 15,110   |
| 0035     |   +0.483 |   +0.333 | 1/3       | 4,041    |
| 0036     |   +0.317 |   +0.222 | 2/3       | 15,807   |
| 0037     |   -0.267 |   -0.278 | 1/3       | 12,508   |
| 0038     |   -0.050 |   -0.056 | 0/3       | 6,625    |
| 0039     |   +0.050 |   +0.056 | 1/3       | 9,853    |
| 0041     |   -0.017 |   -0.056 | 2/3       | 10,583   |
| 0042     |   -0.183 |   -0.111 | 0/3       | 7,463    |
| 0043     |   -0.417 |   -0.278 | 1/3       | 7,683    |
| 0044     |   -0.400 |   -0.278 | 0/3       | 5,457    |
| 0045     |   -0.333 |   -0.333 | 0/3       | 11,273   |
| 0046     |   +0.100 |   +0.056 | 1/3       | 7,394    |
| 0047     |   -0.517 |   -0.333 | 0/3       | 7,435    |
| 0048     |   +0.300 |   +0.222 | 1/3       | 12,492   |
| 0049     |   +0.250 |   +0.278 | 2/3       | 11,018   |

## Error Analysis

### Common Ranking Errors
**Adjacent Swaps:**
- Positions 8-9: 2 occurrences
- Positions 7-8: 2 occurrences
- Positions 3-4: 1 occurrences
- Positions 5-6: 1 occurrences
- Positions 4-5: 1 occurrences

## Statistical Distribution

### Spearman Correlation Distribution
```
+0.8-+1.0: ▓ (1 samples)
+0.6-+0.8:  (0 samples)
+0.4-+0.6: ▓▓▓▓▓▓ (5 samples)
+0.2-+0.4: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (12 samples)
+0.0-+0.2: ▓▓▓▓▓▓▓▓▓▓ (8 samples)
-0.2-+0.0: ▓▓▓▓▓▓▓▓▓▓ (8 samples)
-0.4--0.2: ▓▓▓▓▓▓▓▓▓▓ (8 samples)
-0.6--0.4: ▓▓▓▓▓▓ (5 samples)
-0.8--0.6:  (0 samples)
-1.0--0.8:  (0 samples)
```

### Kendall's Tau Distribution
```
+0.8-+1.0:  (0 samples)
+0.6-+0.8: ▓ (1 samples)
+0.4-+0.6: ▓ (1 samples)
+0.2-+0.4: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (14 samples)
+0.0-+0.2: ▓▓▓▓▓▓▓▓▓▓ (10 samples)
-0.2-+0.0: ▓▓▓▓▓▓▓▓▓▓ (10 samples)
-0.4--0.2: ▓▓▓▓▓▓▓▓▓▓ (10 samples)
-0.6--0.4: ▓ (1 samples)
-0.8--0.6:  (0 samples)
-1.0--0.8:  (0 samples)
```