#!/bin/bash
# For every dataset that has a cluster table, submit the decontam pipeline
# (prepare-decontam-input -> decontaminate.py) as a single sbatch.
set -euo pipefail

SFT="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft"
SFTCOT="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft_cot"
DEDUP="/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/dedup"
WRAP="$HOME/apertus1p5-decontam/vision-sft-decontam/dedup"

declare -A DS_SCRIPT DS_GLOB DS_FOLDER DS_EXTRA
register() {
    DS_SCRIPT[$1]="$2"
    DS_GLOB[$1]="$3"
    DS_FOLDER[$1]="$4"
    DS_EXTRA[$1]="${5:-}"
}

# Same registry as launch-all.sh
register innovator         compute-prompt-hashes.py     "$SFT/hf___InnovatorLab___Innovator-VL-Instruct-46M/data*/SFT_*.parquet"                                  hf___InnovatorLab___Innovator-VL-Instruct-46M
register chartverse        compute-chartverse.py        "$SFT/hf___opendatalab___ChartVerse-SFT-1.8M/*.parquet"                                                   hf___opendatalab___ChartVerse-SFT-1.8M
register vdr-cooking       compute-vdr-cooking.py       "$SFT/hf___racineai___VDR_Cooking_Recipes/*.parquet"                                                      hf___racineai___VDR_Cooking_Recipes
register path-vqa          compute-path-vqa.py          "$SFT/hf___flaviagiammarino___path-vqa/*.parquet"                                                         hf___flaviagiammarino___path-vqa
register pixmo-ask         compute-pixmo-ask.py         "$SFT/hf___allenai___pixmo-ask-model-anything/data/*.parquet"                                             hf___allenai___pixmo-ask-model-anything
register pixmo-cap-qa      compute-pixmo-cap-qa.py      "$SFT/hf___allenai___pixmo-cap-qa/data/*.parquet"                                                         hf___allenai___pixmo-cap-qa
register pixmo-point       compute-pixmo-point.py       "$SFT/hf___allenai___pixmo-point-explanations/data/*.parquet"                                             hf___allenai___pixmo-point-explanations
register common-o          compute-common-o.py          "$SFT/hf___facebook___Common-O/data/*.parquet"                                                            hf___facebook___Common-O
register mathnet           compute-mathnet.py           "$SFT/hf___ShadenA___MathNet/data/*/*.parquet"                                                            hf___ShadenA___MathNet
register radimagenet       compute-radimagenet.py       "$SFT/hf___raidium___RadImageNet-VQA/**/*.parquet"                                                        hf___raidium___RadImageNet-VQA
register omnimodal-agent   compute-omnimodal-agent.py   "$SFT/hf___RUC-NLPIR___Omnimodal-Agent-SFT-2K/data/*.parquet"                                             hf___RUC-NLPIR___Omnimodal-Agent-SFT-2K
register mmfinereason      compute-mmfinereason.py      "$SFTCOT/hf___OpenDataArena___MMFineReason-1.8M-Qwen3-VL-235B-Thinking/data/*.parquet"                    hf___OpenDataArena___MMFineReason-1.8M-Qwen3-VL-235B-Thinking
register llava-cot         compute-llava-cot.py         "$SFTCOT/hf___mvp-lab___LLaVA-OneVision-1.5-Instruct-Data___llava_cot_100k/0.0.0/*/*.arrow"               hf___mvp-lab___LLaVA-OneVision-1.5-Instruct-Data___llava_cot_100k
register bigdata-ksu       compute-bigdata-ksu.py       "$SFT/hf___BigData-KSU___RS-instructions-dataset/*.json"                                                  hf___BigData-KSU___RS-instructions-dataset
register culturalground    compute-culturalground.py    "$SFT/hf___neulab___CulturalGround/*.jsonl"                                                               hf___neulab___CulturalGround
register personavlm        compute-personavlm.py        "$SFT/hf___ClareNie___PersonaVLM-Dataset/sft/sft.json $SFT/hf___ClareNie___PersonaVLM-Dataset/rl/rl.jsonl" hf___ClareNie___PersonaVLM-Dataset
register tcm-shizhen       compute-tcm-shizhen.py       "$SFT/hf___FreedomIntelligence___TCM-Instruction-Tuning-ShizhenGPT/TCM_speech_instruction.json $SFT/hf___FreedomIntelligence___TCM-Instruction-Tuning-ShizhenGPT/TCM_text_instruction.json $SFT/hf___FreedomIntelligence___TCM-Instruction-Tuning-ShizhenGPT/TCM_vision_instruction.json $SFT/hf___FreedomIntelligence___TCM-Instruction-Tuning-ShizhenGPT/TCM_vision_instruction.jsonl" hf___FreedomIntelligence___TCM-Instruction-Tuning-ShizhenGPT
register drim              compute-drim.py              "$SFT/hf___xiuhuywh___DRIM-VisualReasonHard/train.json"                                                   hf___xiuhuywh___DRIM-VisualReasonHard
register onethinker        compute-onethinker.py        "$SFT/hf___OneThink___OneThinker-train-data/onethinker_sft_image.json $SFT/hf___OneThink___OneThinker-train-data/onethinker_sft_video.json $SFT/hf___OneThink___OneThinker-train-data/onethinker_rl_train.json $SFT/hf___OneThink___OneThinker-train-data/onethinker_rl_train_unsampled.json" hf___OneThink___OneThinker-train-data
register spiqa             compute-spiqa.py             "$SFT/hf___scimdr___SPIQA_50K_Re/spiqa_50k.json $SFT/hf___scimdr___SPIQA_50K_Re/spiqa_50k_reannotate.json" hf___scimdr___SPIQA_50K_Re
register molmo2            compute-molmo2.py            "$SFT/hf___allenai___Molmo2-MultiImageQA/data/*.parquet"                                                  hf___allenai___Molmo2-MultiImageQA
register rsrcc             compute-rsrcc.py             "$SFT/hf___google___RSRCC/*/metadata.csv"                                                                 hf___google___RSRCC
register bigearthnet       compute-bigearthnet.py       "$SFT/hf___BIFOLD-BigEarthNetv2-0___BigEarthNet/text/BigEarthNet.txt.parquet"                             hf___BIFOLD-BigEarthNetv2-0___BigEarthNet
register eo-data           compute-eo-data.py           "$SFT/hf___IPEC-COMMUNITY___EO-Data1.5M/*/*.parquet"                                                      hf___IPEC-COMMUNITY___EO-Data1.5M
register pangea-master     compute-pangea-master.py     "$SFT/hf___neulab___PangeaInstruct/PangeaIns.json"                                                        hf___neulab___PangeaInstruct
register nemotron-archive  compute-nemotron-archive.py  "$SFT/nemotron_image_training_v3/archive/*.parquet"                                                       nemotron_image_training_v3
register nemotron-nvidia   compute-nemotron-nvidia.py   "$SFT/nemotron_image_training_v3/hf___nvidia___Nemotron-Image-Training-v3/*/*.jsonl"                      nemotron_image_training_v3_hf_nvidia
register nemotron-swissai  compute-nemotron-archive.py  "$SFT/nemotron_image_training_v3/swissai___Nemotron-Image-Training-v3/*/*.parquet"                        nemotron_image_training_v3_swissai
register llava-onevision   compute-llava-cot.py         "$SFT/hf___mvp-lab___LLaVA-OneVision-1.5-Instruct-Data/*/0.0.0/*/*.arrow"                                 hf___mvp-lab___LLaVA-OneVision-1.5-Instruct-Data

submit_one() {
    local name="$1"
    local folder="${DS_FOLDER[$name]}"
    local script="${DS_SCRIPT[$name]}"
    local glob="${DS_GLOB[$name]}"
    local ds_dir="$DEDUP/$folder"
    if [ ! -f "$ds_dir/clusters_prompt_prefix8.parquet" ]; then
        echo "[$name] SKIP — no clusters_prompt_prefix8.parquet"
        return
    fi
    if [ -f "$ds_dir/decontam-reports-13gram07/summary.json" ]; then
        echo "[$name] SKIP — decontam already complete"
        return
    fi
    DS_FOLDER="$folder" SOURCE_GLOB="$glob" COMPUTE_SCRIPT="$script" \
        sbatch --export=ALL --job-name="dec-$name" \
        "$WRAP/submit-decontam-pipeline.sbatch" \
        | awk -v name="$name" '{print "[" name "] " $0}'
}

if [ "$#" -ge 1 ]; then
    for n in "$@"; do submit_one "$n"; done
else
    for n in "${!DS_FOLDER[@]}"; do submit_one "$n"; done
fi
