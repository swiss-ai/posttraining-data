#!/bin/bash

dataset_path="/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/Llama-Nemotron-Post-Training-Dataset_wo_math_code"
python license_filtering.py --dataset_path="${dataset_path}"
