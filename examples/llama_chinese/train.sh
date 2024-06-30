#!/bin/bash
MODELSCOPE_BASE="/root/.cache/modelscope/hub/LLM-Research"

cd LLaMA-Factory

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCLL_P2P_DISABLE="1"
export NCLL_IB_DISABLE="1"

CUDA_VISIBLE_DEVICES=0 python3 src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path ${MODELSCOPE_BASE}/Meta-Llama-3-8B-Instruct \
    --preprocessing_num_workers 16 \
    --flash_attn auto \
    --dataset alpaca_zh,alpaca_gpt4_zh,oaast_sft_zh \
    --template llama3 \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --output_dir ${MODELSCOPE_BASE}/Meta-Llama-3-8B-Instruct-Adapter \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 5000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --finetuning_type lora \
    --fp16 \
    --lora_rank 8 \
    --report_to none
    
cd ..