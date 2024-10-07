export TOKENIZERS_PARALLELISM=true
nohup python3 train.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct &> train.log &