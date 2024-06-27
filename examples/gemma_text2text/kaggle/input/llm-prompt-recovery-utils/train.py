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

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_file_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--test_data_file_path', type=str)

def formatting_func(example):
    output_texts = []
    for i in range(len(example)):
        text = f"""Please recover the rewrite prompt based on Original text and Rewritten text below: 
        {example['mixed_texts'][i]}:
        Rewrite prompt: {example['rewrite_prompt'][i]}"""
        output_texts.append(text)
    return output_texts

def inference(text, model, tokenizer, device='cuda'):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    args = parser.parse_args()
    train_data_file_path = args.train_data_file_path
    model_path = args.model_path
    save_dir = args.save_dir
    test_data_file_path = args.test_data_file_path
    
    data = pd.read_csv(train_data_file_path)
    data['mixed_texts'] = data.apply(lambda row: f"Original text: {row['original_text']}, Rewritten text: {row['rewritten_text']}", axis=1)
    train_data, valid_data = train_test_split(data, test_size=0.1, shuffle=True)
    train_dataset = Dataset.from_pandas(train_data)
    valid_dataset = Dataset.from_pandas(valid_data)
    data_dict = DatasetDict()
    data_dict['train'] = train_dataset
    data_dict['valid'] = valid_dataset
    test = pd.read_csv(test_data_file_path)

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")

    data_dict = data_dict.map(lambda samples: tokenizer(samples["mixed_texts"]), batched=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        num_train_epochs=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="/kaggle/working/",
        optim="paged_adamw_8bit",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=data_dict['train'],
        eval_dataset=data_dict["valid"],
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=128)
    
    trainer.train()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    trainer.save_model(save_dir)

    adapter_model = PeftModel.from_pretrained(model, save_dir)
    merged_model = adapter_model.merge_and_unload()

    rewrite_prompts = []

    for original_text, rewritten_text in zip(test['original_text'], test['rewritten_text']):
        text = f"Original text: {original_text}, Rewritten text: {rewritten_text}\nRewrite prompt:"
        outputs = inference(text, merged_model, tokenizer)
        rewrite_prompts.append(outputs.split('Rewrite prompt:')[-1])

    test['rewrite_prompt'] = pd.Series(rewrite_prompts)
    test = test[['id', 'rewrite_prompt']]
    test.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()