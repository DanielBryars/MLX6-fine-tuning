import sys
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset

from llama2scratch import test_slang

import trl
print(trl.__version__)

from trl import SFTTrainer
import inspect
print(inspect.getfile(SFTTrainer))

base_model = "meta-llama/Llama-2-7b-chat-hf"
new_model = "llama-2-7b-chat-cold"

dataset = load_dataset("json", data_files="slang.jsonl")["train"]

# Format into prompt + response for training
#def format_example(example):
#    return {
#        "text": f"{example['prompt']}\n{example['completion']}"
#    }

def format_example(example):
    #see https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-2/
    return {
        "text": f"<s>[INST] {example['prompt']} [/INST] {example['completion']}</s>"
    }


dataset = dataset.map(format_example)

DEBUG_DS = False

if DEBUG_DS:
    for i, row in enumerate(dataset):
        print("Training Example:")
        print(row["text"])
        if i >= 4:  # limit to first 5 rows
            break

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)


training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb", 
    run_name="llama-slang-001"   
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

#packing=False,
    #max_seq_length=None,
    #dataset_text_field="text",


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    processing_class=tokenizer,
    args=training_params,
    
)

print("BEFORE FINETUNING")
model.eval()
test_slang(model,tokenizer,file_path="slang_terms_short.json")

model.train()

trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

print()
print("AFTER FINETUNING")

model.eval()
test_slang(model,tokenizer)     
