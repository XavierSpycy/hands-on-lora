#!/bin/bash

llama.cpp/llama-server -m ./Meta-Llama-3-8B-Instruct-zh-10k/meta-llama-3-8b-instruct-zh-10k.Q8_0.gguf \
    -c 512 \
    -b 64 \
    -n 256 \
    -t 12 \
    --repeat_penalty 1.2 \
    --top_k 20 \
    --top_p 0.5 \
    --host 0.0.0.0 \
    --port 6006