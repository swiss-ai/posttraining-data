# Judge LLM Ranking Evaluation Report

## Summary

Model: Qwen/Qwen3-32B
Instructions: prompts/principles.txt
Samples Evaluated: 50
Timestamp: 2025-08-19 15:14:47

### Overall Metrics
- Success Rate: 100.0% (50/50)
- Failed: 0.0% (0/50)
Spearman measures rank order preservation (>0.7 good, >0.9 excellent), Kendall measures pairwise agreement (>0.5 good, >0.7 excellent).
- Mean Spearman Correlation: 0.805 (std: 0.324)
- Mean Kendall's Tau: 0.736 (std: 0.298)
- Median Spearman: 0.925
- Median Kendall's Tau: 0.833

### Position Accuracy
Top-1 measures correctly identifying the best response (giving it rank 9), Top-3 measures having all 3 best responses in ranks 7-9.
- Top-1 Accuracy: 76.0%
- Top-3 Accuracy: 82.7%
- Bottom-1 Accuracy: 56.0%
- Perfect Rankings: 6/50 (12.0%)

### Token Usage
- Total Tokens: 696,884
- Average per Sample: 13938 tokens
  - Prompt: 13916 tokens
  - Completion: 22 tokens

## Detailed Results

### Successful Samples (50 total)

| Sample   | Spearman | Kendall  | Top-3 Acc | Tokens   |
|----------|----------|----------|-----------|----------|
| 0000     |   +0.917 |   +0.778 | 2/3       | 15,569   |
| 0001     |   +0.950 |   +0.889 | 2/3       | 10,090   |
| 0002     |   +1.000 |   +1.000 | 3/3       | 7,460    |
| 0003     |   +0.800 |   +0.722 | 2/3       | 20,484   |
| 0004     |   +1.000 |   +1.000 | 3/3       | 13,867   |
| 0005     |   +0.983 |   +0.944 | 3/3       | 10,240   |
| 0006     |   +0.933 |   +0.833 | 2/3       | 15,047   |
| 0007     |   +0.933 |   +0.833 | 3/3       | 13,080   |
| 0008     |   +0.833 |   +0.667 | 2/3       | 8,623    |
| 0009     |   +0.983 |   +0.944 | 2/3       | 7,556    |
| 0010     |   +0.933 |   +0.833 | 3/3       | 13,138   |
| 0011     |   +0.917 |   +0.778 | 3/3       | 18,240   |
| 0012     |   +1.000 |   +1.000 | 3/3       | 15,927   |
| 0013     |   -0.233 |   -0.222 | 1/3       | 13,976   |
| 0014     |   +0.333 |   +0.389 | 2/3       | 19,522   |
| 0015     |   +0.983 |   +0.944 | 3/3       | 11,251   |
| 0016     |   +0.933 |   +0.833 | 3/3       | 45,282   |
| 0017     |   +0.617 |   +0.444 | 3/3       | 9,381    |
| 0018     |   +0.917 |   +0.778 | 3/3       | 13,220   |
| 0019     |   +0.833 |   +0.778 | 3/3       | 7,446    |
| 0020     |   +0.683 |   +0.556 | 2/3       | 22,957   |
| 0021     |   +0.967 |   +0.889 | 3/3       | 9,197    |
| 0022     |   +0.817 |   +0.611 | 3/3       | 16,261   |
| 0023     |   +0.883 |   +0.778 | 2/3       | 13,944   |
| 0024     |   +1.000 |   +1.000 | 3/3       | 12,329   |
| 0025     |   +0.867 |   +0.722 | 2/3       | 20,783   |
| 0026     |   +0.950 |   +0.889 | 3/3       | 14,510   |
| 0027     |   +0.983 |   +0.944 | 3/3       | 9,799    |
| 0028     |   +0.867 |   +0.778 | 2/3       | 9,221    |
| 0029     |   +0.933 |   +0.833 | 3/3       | 16,994   |
| 0030     |   +0.933 |   +0.833 | 3/3       | 10,525   |
| 0031     |   -0.700 |   -0.500 | 0/3       | 13,918   |
| 0032     |   +0.983 |   +0.944 | 3/3       | 8,526    |
| 0033     |   +0.383 |   +0.333 | 1/3       | 29,628   |
| 0034     |   +1.000 |   +1.000 | 3/3       | 17,368   |
| 0035     |   +0.700 |   +0.500 | 2/3       | 10,024   |
| 0036     |   +0.150 |   +0.056 | 1/3       | 19,507   |
| 0037     |   +0.967 |   +0.889 | 2/3       | 12,895   |
| 0038     |   +0.983 |   +0.944 | 3/3       | 8,424    |
| 0039     |   +0.717 |   +0.611 | 3/3       | 12,574   |
| 0040     |   +0.917 |   +0.778 | 2/3       | 9,009    |
| 0041     |   +0.967 |   +0.889 | 3/3       | 12,672   |
| 0042     |   +0.917 |   +0.778 | 2/3       | 10,291   |
| 0043     |   +0.900 |   +0.833 | 3/3       | 13,677   |
| 0044     |   +0.983 |   +0.944 | 2/3       | 7,910    |
| 0045     |   +0.917 |   +0.778 | 3/3       | 11,592   |
| 0046     |   +1.000 |   +1.000 | 3/3       | 14,788   |
| 0047     |   +0.633 |   +0.556 | 3/3       | 10,214   |
| 0048     |   +0.383 |   +0.500 | 2/3       | 14,617   |
| 0049     |   +0.983 |   +0.944 | 3/3       | 13,331   |

## Error Analysis

### Common Ranking Errors
**Adjacent Swaps:**
- Positions 7-8: 2 occurrences
- Positions 3-4: 2 occurrences
- Positions 4-5: 2 occurrences
- Positions 6-7: 1 occurrences
- Positions 5-6: 1 occurrences

## Statistical Distribution

### Spearman Correlation Distribution
```
+0.8-+1.0: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (38 samples)
+0.6-+0.8: ▓▓ (6 samples)
+0.4-+0.6:  (0 samples)
+0.2-+0.4: ▓ (3 samples)
+0.0-+0.2:  (1 samples)
-0.2-+0.0:  (0 samples)
-0.4--0.2:  (1 samples)
-0.6--0.4:  (0 samples)
-0.8--0.6:  (1 samples)
-1.0--0.8:  (0 samples)
```

### Kendall's Tau Distribution
```
+0.8-+1.0: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (26 samples)
+0.6-+0.8: ▓▓▓▓▓▓▓▓ (14 samples)
+0.4-+0.6: ▓▓ (5 samples)
+0.2-+0.4: ▓ (2 samples)
+0.0-+0.2:  (1 samples)
-0.2-+0.0:  (0 samples)
-0.4--0.2:  (1 samples)
-0.6--0.4:  (1 samples)
-0.8--0.6:  (0 samples)
-1.0--0.8:  (0 samples)
```