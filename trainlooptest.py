from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Dummy data
data = {"text": ["Hello world!", "How are you?"]}
dataset = Dataset.from_dict(data)

# Model + tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# SFT config
args = SFTConfig(
    output_dir="test",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    dataset_text_field="text",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,    
    args=args,
    processing_class=tokenizer,
)


