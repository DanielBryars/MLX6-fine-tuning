import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "meta-llama/Llama-2-7b-chat-hf"

def process_prompt(model, tokenizer, prompt):
    prompt=f"<s>[INST] {prompt}  [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response


print (f"Model: {model_name}")

def test_slang(model, tokenizer, file_path="slang_terms_since_2022.json"):
    for slang, year, before, after in load_slang(file_path):

            prompt = f"What does '{slang}' mean?"
            response = process_prompt(model, tokenizer,prompt)
            print(f"Actual:{response}")
            print(f"{before}")
            print(f"{after}")
        
    sys.exit()


def load_slang(file_path = "slang_terms_since_2022.json"):
    with open(file_path, "r") as f:
        slang_data = json.load(f)

    for entry in slang_data:
        yield entry['Term'], entry['Year Became Slang'], entry['Meaning Before'], entry['Meaning After']

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = ""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    if (prompt == ""):
        test_slang(model,tokenizer)        

    while True:
        print("....")
        response = process_prompt(prompt)
        print(response)

        lastprompt = prompt
        prompt = input(f"{model_name} ready. Enter a prompt (or hit <enter> to repeat the last prompt):")
        if prompt == "":
            #enter to repeat last prompt
            prompt = lastprompt