# HuggingFace Dataset Downloader

Downloads datasets from HuggingFace Hub with compatibility checks and metadata tracking.

## Description

This script downloads HuggingFace datasets and stores them locally with metadata including version information, commit hash, and dataset details. It includes a compatibility check for the `datasets` library version (requires 3.3.2 for compatibility with current docker images).

## Arguments

- `dataset_name`: Name of the dataset to download (e.g., 'allenai/tulu-3-sft-mixture')
- `--download-folder`: Folder to download the dataset to (required)
- `--subset`: Dataset subset/configuration name (optional)
- `--split`: Dataset split to download (e.g., 'train', 'test', 'validation') (optional)

## Examples

```bash
python 01-hf-download/hf-download.py allenai/tulu-3-sft-mixture --download-folder data/01-hf-data
python 01-hf-download/hf-download.py HuggingFaceTB/smoltalk --download-folder data/01-hf-data --subset all
python 01-hf-download/hf-download.py HuggingFaceTB/smoltalk2 --download-folder data/01-hf-data --subset SFT
python 01-hf-download/hf-download.py arcee-ai/The-Tome --download-folder data/01-hf-data
python 01-hf-download/hf-download.py utter-project/EuroBlocks-SFT-Synthetic-1124 --download-folder data/01-hf-data
python 01-hf-download/hf-download.py nvidia/Llama-Nemotron-Post-Training-Dataset --download-folder data/01-hf-data
```

## Known SFT & Alignment Datasets

```
AI-MO/NuminaMath-TIR
AI4Chem/ChemData700K
BAAI/Infinity-Instruct
BAAI/Infinity-Preference
CohereLabs/aya_dataset
FreedomIntelligence/medical-o1-reasoning-SFT
HuggingFaceH4/no_robots
HuggingFaceTB/smoltalk
HuggingFaceTB/smoltalk2
HumanLLMs/Human-Like-DPO-Dataset
LipengCS/Table-GPT
NousResearch/hermes-function-calling-v1
OpenCoder-LLM/opc-sft-stage2
PKU-Alignment/Align-Anything-Instruction-100K
RyokoAI/ShareGPT52K
Salesforce/xlam-function-calling-60k
ServiceNow-AI/M2Lingual
SirNeural/flan_v2
Skywork/Skywork-Reward-Preference-80K-v0.2
THUDM/AgentInstruct
TIGER-Lab/MathInstruct
Team-ACE/ToolACE
Vezora/Code-Preference-Pairs
Vezora/Open-Critic-GPT
Vezora/Tested-143k-Python-Alpaca
allenai/SciRIFF
allenai/WildChat
allenai/WildChat-1M
allenai/tulu-3-sft-mixture
arcee-ai/The-Tome
argilla/databricks-dolly-15k-curated-en
argilla/ifeval-like-data
argilla/ultrafeedback-binarized-preferences-cleaned
b-mc2/sql-create-context
bigcode/self-oss-instruct-sc2-exec-filter-50k
camel-ai/ai_society
camel-ai/amc_aime_distilled
camel-ai/backtranslated-tir
camel-ai/physics
chargoddard/WebInstructSub-prometheus
databricks/databricks-dolly-15k
garage-bAInd/Open-Platypus
glaiveai/glaive-code-assistant
glaiveai/glaive-function-calling-v2
gorilla-llm/APIBench
gretelai/synthetic_text_to_sql
ibm-research/finqa
internlm/Agent-FLAN
jondurbin/airoboros-3.2
lmsys/chatbot_arena_conversations
lmsys/lmsys-chat-1m
m-a-p/Code-Feedback
m-a-p/CodeFeedback-Filtered-Instruction
meta-math/MetaMathQA
microsoft/orca-agentinstruct-1M-v1
microsoft/orca-math-word-problems-200k
mlabonne/open-perfectblend
mlabonne/orca-agentinstruct-1M-v1-cleaned
mlabonne/orpo-dpo-mix-40k
mteb/summeval
nvidia/AceReason-1.1-SFT
nvidia/CantTalkAboutThis-Topic-Control-Dataset-NC
nvidia/ChatQA-Training-Data
nvidia/HelpSteer2
nvidia/HelpSteer3
nvidia/Llama-Nemotron-Post-Training-Dataset
openbmb/UltraFeedback
openbmb/UltraSafety
stanfordnlp/wikitablequestions
stingning/ultrachat
tatsu-lab/alpaca
theblackcat102/evol-codealpaca-v1
utter-project/EuroBlocks-SFT-Synthetic-1124
yizhongw/self_instruct
```