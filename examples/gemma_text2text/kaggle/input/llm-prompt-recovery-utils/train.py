import os
import warnings
import argparse

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_file_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--test_data_file_path', type=str)
parser.add_argument('--num_epochs', type=int, default=5)

def format_texts(original_text, rewritten_text, prompt_end=""):
    formatted_text = (
        "Task: Generate a rewrite prompt based on the context below.\n"
        "===\n"
        f"Original Text: {original_text}\n"
        "===\n"
        f"Rewritten Text: {rewritten_text}\n"
        "===\n"
        + prompt_end
    )
    return formatted_text

def generate_mixed_text(example):
    return format_texts(example['original_text'], example['rewritten_text'])

def formatting_func(example):
    prompt_end = f"Output Rewrite Prompt: {example['rewrite_prompt']}<eos>"
    return [format_texts(example['original_text'], example['rewritten_text'], prompt_end)]

def formatting_func_texts(original_text, rewritten_text):
    prompt_end = "Output Rewrite Prompt: "
    return format_texts(original_text, rewritten_text, prompt_end)

def inference(text, model, tokenizer, device='cuda'):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    args = parser.parse_args()

    model_path = args.model_path
    train_data_file_path = args.train_data_file_path
    save_dir = args.save_dir
    test_data_file_path = args.test_data_file_path
    num_epochs = args.num_epochs
    
    lora_config = LoraConfig(
        r=12,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
    
    data = pd.read_csv(train_data_file_path)
    data['mixed_texts'] = data.apply(generate_mixed_text, axis=1)
    train_data, valid_data = train_test_split(data, test_size=0.1, shuffle=True)
    train_dataset = Dataset.from_pandas(train_data)
    valid_dataset = Dataset.from_pandas(valid_data)
    data_dict = DatasetDict()
    data_dict['train'] = train_dataset
    data_dict['valid'] = valid_dataset
    test = pd.read_csv(test_data_file_path)

    data_dict = data_dict.map(lambda samples: tokenizer(samples["mixed_texts"]), batched=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="/kaggle/working/",
        optim="paged_adamw_8bit",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="none")
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=data_dict['train'],
        eval_dataset=data_dict["valid"],
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=256)
    
    trainer.train()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    trainer.save_model(save_dir)

    adapter_model = PeftModel.from_pretrained(model, save_dir)
    merged_model = adapter_model.merge_and_unload()

    rewrite_prompts = []

    for original_text, rewritten_text in zip(test['original_text'], test['rewritten_text']):
        text = formatting_func_texts(original_text, rewritten_text)
        outputs = inference(text, merged_model, tokenizer)
        rewrite_prompts.append(outputs.split('Output Rewrite Prompt:')[-1])

    test['rewrite_prompt'] = pd.Series(rewrite_prompts)
    test = test[['id', 'rewrite_prompt']]
    test.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()