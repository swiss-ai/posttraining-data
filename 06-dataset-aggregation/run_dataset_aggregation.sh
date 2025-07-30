#!/bin/bash

mixture_name="apartus-sft-mixture-3"
python dataset-aggregation.py \
  --mixture_name "${mixture_name}" \
  --config_path "data-mixtures.yaml" \
  --output "/capstor/store/cscs/swissai/infra01/posttrain_data/06_sft_mixtures/${mixture_name}"