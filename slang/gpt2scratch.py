import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model and tokenizer
model_name = "gpt2-xl"  # You can use "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
model.to(device)

print (f"Model: {model_name}")

if (__name__ == "__main__"):

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "What is the capital of France?"
    
        print(prompt)
        
    while(True):
        print("....")

        inputs = tokenizer(prompt, return_tensors="pt")

        inputs.to(device)

        outputs = model.generate(
            **inputs, 
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id)
    
        outputs = outputs.cpu()
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(response)

        prompt = input("Enter a prompt:")