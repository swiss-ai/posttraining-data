### RM-Bench — domain average accuracy

|   Judge # | Description                      | chat     | math     | code     | safety   | total_avg_acc   |
|----------:|:---------------------------------|:---------|:---------|:---------|:---------|:----------------|
|        00 | Mean of scores from judges 01-04 | 0.80     | 0.78     | 0.75     | 0.93     | 0.82            |
|        01 | Helpfulness                      | 0.73     | 0.77     | 0.74     | 0.93     | 0.79            |
|        02 | Instruction Following            | 0.76     | 0.77     | 0.75     | 0.93     | 0.80            |
|        03 | Honesty                          | 0.81     | 0.75     | 0.74     | 0.92     | 0.81            |
|        04 | Truthfulness                     | **0.83** | 0.76     | 0.76     | 0.91     | 0.82            |
|        05 | Swiss AI Charter compliance      | 0.72     | 0.73     | 0.69     | 0.90     | 0.76            |
|        11 | General quality                  | 0.74     | 0.78     | 0.74     | 0.93     | 0.80            |
|        12 | ArenaHard Judge                  | 0.60     | **0.92** | 0.67     | 0.88     | 0.77            |
|        21 | 01, but with Qwen3.6-27B         | 0.77     | 0.84     | **0.79** | **0.94** | **0.83**        |
|        22 | 01, but with Qwen3.5-35B-A3B-FP8 | 0.68     | 0.79     | 0.73     | 0.92     | 0.78            |


### RM-Bench — style matrix (hard / normal / easy)

|   Judge # | Description                      | hard_acc   | normal_acc   | easy_acc   |
|----------:|:---------------------------------|:-----------|:-------------|:-----------|
|        00 | Mean of scores from judges 01-04 | 0.74       | 0.83         | 0.87       |
|        01 | Helpfulness                      | 0.68       | 0.82         | **0.88**   |
|        02 | Instruction Following            | 0.72       | 0.82         | **0.88**   |
|        03 | Honesty                          | 0.75       | 0.82         | 0.85       |
|        04 | Truthfulness                     | 0.80       | 0.82         | 0.82       |
|        05 | Swiss AI Charter compliance      | 0.61       | 0.79         | **0.88**   |
|        11 | General quality                  | 0.69       | 0.82         | **0.88**   |
|        12 | ArenaHard Judge                  | 0.72       | 0.77         | 0.82       |
|        21 | 01, but with Qwen3.6-27B         | **0.81**   | **0.84**     | 0.85       |
|        22 | 01, but with Qwen3.5-35B-A3B-FP8 | 0.69       | 0.79         | 0.86       |

