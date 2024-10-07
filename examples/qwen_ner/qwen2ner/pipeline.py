import os
from typing import (
    Optional, 
    Dict, 
    Any,
)

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
)
from trl import SFTTrainer
from peft import PeftModel

from .peft_config import get_lora_config, get_bnb_config
from .training_args import get_training_args

class Qwen2NERPipeline:
    
    def __init__(self, model_name_or_path: str, data_folder: str = 'dataset') -> None:
        self.model_name_or_path = model_name_or_path
        self.data_folder = data_folder
    
    def load_dataset(
        self,
        path: str = 'csv',
        data_files: Dict[str, str] = {'train': 'train.csv'},
        test_size: float = 0.1,
        seed: Optional[int] = 3407,
    ) -> None:
        data_files = {split: os.path.join(self.data_folder, data_file) for split, data_file in data_files.items()}
        dataset = load_dataset(path=path, data_files=data_files, split='train')
        dataset = dataset.train_test_split(test_size=test_size, seed=seed)
        self.train_dataset = dataset['train']
        self.valid_dataset = dataset['test']

    def init_model(
            self, 
        ) -> None:

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map="auto", 
            quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16,
        )
        self.model.config.use_cache = False
        self.model.get_memory_footprint()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
    
    def init_data_collator(self) -> None:
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model,
            padding=False)
    
    def get_peft_config(
            self,
            lora_kwargs: Dict[str, Any] = {},
            bnb_kwargs: Dict[str, Any] = {},
        ) -> None:
        self.peft_config = get_lora_config(**lora_kwargs)
        self.bnb_config = get_bnb_config(**bnb_kwargs)
    
    def get_training_args(self) -> None:
        self.training_args = get_training_args(output_dir=self.model_name_or_path.split('/')[-1] + '-SFT')

    def train(
            self, 
            resume_from_checkpoint: Optional[str] = None,
            final_checkpoint: str = 'checkpoint-optimal',
        ):
        trainer = SFTTrainer(
            self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer,
            peft_config=self.peft_config,
            data_collator=self.data_collator,
        )

        trainer.train(resume_from_checkpoint)

        trainer.model.save_pretrained(final_checkpoint)
        self.tokenizer.save_pretrained(final_checkpoint)

        self.final_checkpoint = final_checkpoint
        torch.cuda.empty_cache()

    def merge(self):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            return_dict=True,
            quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model = PeftModel.from_pretrained(base_model, self.final_checkpoint)
        model = model.merge_and_unload()

        # save model and tokenizer
        model.save_pretrained(self.model_name_or_path.split('/')[-1] + '-SFT')
        tokenizer.save_pretrained(self.model_name_or_path.split('/')[-1] + '-SFT')

    def __call__(self):
        self.load_dataset()
        self.get_peft_config()
        self.init_model()
        self.init_data_collator()
        self.get_training_args()
        self.train()
        self.merge()