import json
import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,)

from qwen2ner.default import DEFAULT_SYSTEM_PROMPT

def main(args):
    dataset = load_dataset(path='json', data_files={'train': 'dataset/train.json'}, split='train')
    prompt = dataset['full_text'][2]
    tokens = dataset['tokens'][2]
    labels = dataset['labels'][2]
    ground_truth = "\n".join([json.dumps({'token': token, 'label': label}, ensure_ascii=False) for token, label in zip(tokens, labels) if label != 'O'])
    
    model_name_or_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Ground truths: {ground_truth};\nPredictions: {response}.")

if __name__ == '__main__':
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    args = parser.parse_args()
    main(args)