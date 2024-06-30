import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="/root/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct")
parser.add_argument("--prompt", type=str, default="你好，你是谁？")

class ChatModel:
    def __init__(self, model_dir):
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def chat(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

        terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9)
        
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

def main():
    args = parser.parse_args()
    model = ChatModel(args.model_dir)
    print(model.chat(args.prompt))

if __name__ == "__main__":
    main()