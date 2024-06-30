#!/bin/bash
MODELSCOPE_BASE="/root/.cache/modelscope/hub/LLM-Research"

cd LLaMA-Factory

python3 src/export_model.py \
    --model_name_or_path ${MODELSCOPE_BASE}/Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path ${MODELSCOPE_BASE}/Meta-Llama-3-8B-Instruct-Adapter \
    --template llama3 \
    --finetuning_type lora \
    --export_dir ../Meta-Llama-3-8B-Instruct-zh-10k \
    --export_size 2 \
    --export_legacy_format false

cd ..