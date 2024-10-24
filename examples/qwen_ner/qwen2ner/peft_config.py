import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig

def get_lora_config(
        task_type: str = 'CAUSAL_LM', 
        r: int = 128,
        target_modules: str = 'all-linear',
        lora_alpha: int = 256,
        lora_dropout: float = 0.02,
        fan_in_fan_out: bool = False,
        bias: str = 'lora_only',
        use_rslora: bool = True,
        # init_lora_weights: str = 'pissa',
    ) -> LoraConfig:

    return LoraConfig(
        task_type=task_type,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        fan_in_fan_out=fan_in_fan_out,
        bias=bias,
        use_rslora=use_rslora,
        # init_lora_weights=init_lora_weights,
    )

def get_bnb_config(
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        llm_int8_threshold: float = 6.0,
        bnb_4bit_compute_dtype: str = 'bfloat16',
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
    ) -> BitsAndBytesConfig:
    
    return BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        llm_int8_threshold=llm_int8_threshold,
        bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )