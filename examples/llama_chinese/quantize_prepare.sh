#!/bin/bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

sudo apt update
sudo apt install g++

make

cd ..