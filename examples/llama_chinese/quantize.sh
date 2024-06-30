#!/bin/bash

# python llama.cpp/convert-hf-to-gguf.py ./Meta-Llama-3-8B-Instruct-zh-10k/ --outfile ./Meta-Llama-3-8B-Instruct-zh-10k/meta-llama-3-8b-instruct-zh-10k.gguf --outtype f16

python llama.cpp/convert-hf-to-gguf.py ./Meta-Llama-3-8B-Instruct-zh-10k/ --outfile ./Meta-Llama-3-8B-Instruct-zh-10k/meta-llama-3-8b-instruct-zh-10k.Q8_0.gguf --outtype q8_0

# Or
# llama.cpp/llama-quantize ./Meta-Llama-3-8B-Instruct-zh-10k/ ./Meta-Llama-3-8B-Instruct-zh-10k/meta-llama-3-8b-instruct-zh-10k.Q8_0.gguf Q8_0