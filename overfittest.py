from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

# Configuration
num_image_patches = 49
clip_model_name = "openai/clip-vit-base-patch32"
llm_model_id = "meta-llama/Llama-3.1-8B"
csv_path = "data.csv"
batch_size = 1
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained(clip_model_name).vision_model.eval().to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# Load LLM and tokenizer
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_tokenizer.pad_token = llm_tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(llm_model_id, torch_dtype=torch.float16, device_map="auto")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
llm_model = get_peft_model(base_model, lora_config)

# Add special tokens for image patches
special_tokens = {
    "additional_special_tokens": [f"<image_patch_{i}>" for i in range(num_image_patches)] + [
        "<StartDiagram>", "<EndDiagram>", "<Large>", "<Medium>", "<Small>",
        "<Circle>", "<Square>", "<Triangle>",
        "<TopLeft>", "<TopRight>", "<BottomLeft>", "<BottomRight>"
    ]
}
llm_tokenizer.add_special_tokens(special_tokens)
llm_model.resize_token_embeddings(len(llm_tokenizer))

clip_dim = clip_model.config.hidden_size
llm_dim = llm_model.config.hidden_size
clip_to_llm = nn.Linear(clip_dim, llm_dim).to(device)

# Load example DSL
example_dsl = Path("example.dsl.txt").read_text().strip()

# Dataset definition
class DSLDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = llm_tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dsl = row["dsl"]
        image_path = Path(row["image_path"])

        image_patch_tokens = " ".join([f"<image_patch_{i}>" for i in range(num_image_patches)])

        prompt = (
            "Here's a DSL for representing very simple diagrams:\n\n"
            "<Square> <Triangle> <Circle>\n"
            "<TopLeft> <TopRight> <BottomLeft> <BottomRight> <Large> <Medium> <Small>\n\n"
            "Using that DSL, this diagram:\n"
            f"{image_patch_tokens}\n"
            "corresponds to this markup:\n"
            f"{example_dsl}\n\n"
            "Now use the DSL to write the markup which would generate the following diagram:\n"
            f"{image_patch_tokens}"
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
        label_ids = self.tokenizer(dsl, return_tensors="pt").input_ids.squeeze(0)

        inputs = clip_processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt")
        image_tensor = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = clip_model(image_tensor, output_hidden_states=True)
            patch_embeds = outputs.last_hidden_state[:, 1:, :].squeeze(0).float()
            projected = clip_to_llm(patch_embeds).detach()

        return {
            "input_ids": input_ids,
            "image_patches": projected,
            "labels": label_ids
        }

# Collate function
def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    image_embeds = [b["image_patches"] for b in batch]

    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=llm_tokenizer.pad_token_id)
    labels_padded = torch.full_like(input_ids, fill_value=-100)
    for i, label in enumerate(labels):
        if len(label) > input_ids.shape[1]:
            raise ValueError("Label sequence is longer than input sequence")
        labels_padded[i, -len(label):] = label

    input_ids = input_ids.to(device)
    inputs_embeds = llm_model.get_input_embeddings()(input_ids)

    for i, patches in enumerate(image_embeds):
        for j in range(num_image_patches):
            tok_id = llm_tokenizer.convert_tokens_to_ids(f"<image_patch_{j}>")
            idxs = (input_ids[i] == tok_id).nonzero(as_tuple=True)[0]
            if len(idxs):
                inputs_embeds[i, idxs[0]] = patches[j].to(inputs_embeds.dtype)

    return {
        "inputs_embeds": inputs_embeds.to(device),
        "labels": labels_padded.to(device)
    }

# Training loop
def train():
    dataset = DSLDataset(csv_path)
    sample = dataset[0]
    dataset = [sample]

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    llm_model.train()
    optimizer = torch.optim.AdamW(llm_model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(loop):
            optimizer.zero_grad()
            outputs = llm_model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        # Inference pass
        llm_model.eval()
        with torch.no_grad():
            batch = next(iter(train_loader))
            input_ids = dataset[0]["input_ids"].unsqueeze(0).to(device)
            output = llm_model.generate(
                inputs_embeds=batch["inputs_embeds"],
                max_length=input_ids.shape[1] + 64,
                do_sample=False
            )
            decoded = llm_tokenizer.decode(output[0], skip_special_tokens=True)
            print("--- Model Output After Epoch", epoch + 1, "---")
            print(decoded)
            print("--------------------------------------")
        llm_model.train()

# Call train to start training
train()
