#!/bin/bash

export INSTALL_PATH="/kaggle/input/llm-prompt-recovery-utils/packages/packages"

packages=(peft trl accelerate datasets huggingface_hub transformers bitsandbytes safetensors sentence-transformers)
for pkg in "${packages[@]}"; do
  pip install -q -U $pkg --no-index --find-links "${INSTALL_PATH}"
  if [ $? -ne 0 ]; then
        echo "Installation of $pkg failed"
        exit 1
    fi
done

pip install -q -U datasets==2.16.0 --no-index --find-links "${INSTALL_PATH}" --no-warn-conflicts

if [ $? -ne 0 ]; then
    echo "Installation of datasets failed"
    exit 1
fi

python3 /kaggle/input/llm-prompt-recovery-utils/train.py \
    --train_data_file_path /kaggle/input/llm-prompt-recovery-utils/data.csv \
    --model_path /kaggle/input/gemma/transformers/2b-it/2 \
    --save_dir /kaggle/working/gemma_2b_it_ft \
    --test_data_file_path /kaggle/input/llm-prompt-recovery/test.csv