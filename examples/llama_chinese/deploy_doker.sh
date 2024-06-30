#!/bin/bash

docker run -itd \
    -p 6006:6006 \
    -v ./Meta-Llama-3-8B-Instruct-zh-10k:/models \
    --gpus all ghcr.io/ggerganov/llama.cpp:server-cuda \
    -m models/meta-llama-3-8b-instruct-zh-10k.Q8_0.gguf \
    -c 512 \
    --host 0.0.0.0 \
    --port 6006 \
    --n-gpu-layers 99