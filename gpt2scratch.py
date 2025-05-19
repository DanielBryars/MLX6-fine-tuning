import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()


if (__name__ == "__main__"):

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "What is the capital of France?"
    
    print(prompt)
    print("....")

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(response)
