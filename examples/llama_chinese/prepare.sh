#!/bin/bash
python3 download_model.py

python3 inference.py

sudo apt install git -y

git clone https://github.com/hiyouga/LLaMA-Factory.git

python3 -m pip install --upgrade pip

cd LLaMA-Factory

git checkout ce17eccf451649728cf7b45312fd7f75d3a8a246

python3 -m pip install -r requirements.txt