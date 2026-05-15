"""Convert one Innovator-VL text-only smoke shard to a standardised HF
DatasetDict suitable for input to posttraining-data/04-decontamination."""

import os
import re
from datetime import datetime

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict

SRC = "/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/smoke/SFT_000001.txt.parquet"
DST = "/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/smoke/dataset"

tbl = pq.read_table(SRC)
img_pat = re.compile(r"<image>\n?")
ts = datetime.now().isoformat()

records = []
empty = 0
for row in tbl.to_pylist():
    convs = row["conversations"]
    if not convs:
        empty += 1
        continue
    init = convs[0]
    # already stripped at smoke time, but be defensive
    prompt = img_pat.sub("", init["value"]).strip()
    if not prompt:
        empty += 1
        continue

    messages = []
    for turn in convs[1:]:
        role = "assistant" if turn["from"] in ("gpt", "assistant", "bot") else turn["from"]
        messages.append({
            "role": role,
            "parts": [{
                "type": "response",
                "content": img_pat.sub("", turn["value"]),
                "metadata": "{}",
                "name": "",
                "args": "",
            }],
        })

    records.append({
        "conversation_id": row["id"],
        "dataset_source": "InnovatorLab/Innovator-VL-Instruct-46M",
        "original_metadata": "{}",
        "system_prompt": {"content": "", "metadata": "{}"},
        "initial_prompt": {"role": "user", "content": prompt, "metadata": "{}"},
        "available_functions": [],
        "conversation_branches": [{"messages": messages}],
        "created_timestamp": ts,
    })

ds = Dataset.from_list(records)
DatasetDict({"train": ds}).save_to_disk(DST)
print(f"converted: {len(ds)} samples ({empty} empty skipped) → {DST}")
