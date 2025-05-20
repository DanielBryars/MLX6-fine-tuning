import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

print (f"Model: {model_name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "What is the capital of France?"
        print(prompt)

    while True:
        print("....")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)

        lastprompt = prompt
        prompt = input(f"{model_name} ready. Enter a prompt (or hit <enter> to repeat the last prompt):")
        if prompt == "":
            #enter to repeat last prompt
            prompt = lastprompt