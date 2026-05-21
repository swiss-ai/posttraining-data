### RewardBench-2 — accuracy (higher is better; bracketed % = none rate)

|   Judge # | Description                                                          | Factuality               | Focus     | Math                     | Precise IF                   | Safety                   | Ties      | overall                  |
|----------:|:---------------------------------------------------------------------|:-------------------------|:----------|:-------------------------|:-----------------------------|:-------------------------|:----------|:-------------------------|
|        01 | Helpfulness                                                          | 80.42                    | **87.47** | 74.86                    | 43.12                        | 94.67                    | 76.03     | 76.10                    |
|        02 | Instruction Following                                                | 79.79                    | 86.87     | 76.50                    | 40.62                        | 94.67                    | 76.09     | 75.76                    |
|        03 | Honesty                                                              | **82.95**                | 79.19     | 72.13                    | 46.25                        | **97.33**                | 78.14     | 76.00                    |
|        04 | Truthfulness                                                         | 81.89                    | 76.16     | 72.13                    | 42.50                        | 96.00                    | 86.47     | 75.86                    |
|        21 | 01, but with Qwen3.6-27B                                             | 77.89                    | 81.41     | 75.41                    | 45.62                        | 91.78                    | **87.06** | 76.53                    |
|        22 | 01, but with Qwen3.5-35B-A3B-FP8                                     | 68.00                    | 76.97     | 71.04                    | 41.25                        | 91.56                    | 44.02     | 65.47                    |
|        23 | 12, but with Qwen3.6-27B                                             | 68.69 (5% random scored) | 72.42     | 85.56 (2% random scored) | **51.77 (8% random scored)** | 88.70 (0% random scored) | 77.16     | 74.05 (2% random scored) |
|        24 | 23, but precomputes own answer and reuses, and with activeuf scoring | 74.74                    | 82.22     | **85.79**                | 51.25                        | 87.33                    | 80.77     | **77.02**                |

