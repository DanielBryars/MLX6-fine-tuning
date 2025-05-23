import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
import open_clip
from PIL import Image
from pathlib import Path
import pandas as pd
import random
import wandb
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

# -------------------------
# Configuration
# -------------------------
llm_model_id = "meta-llama/Llama-3.1-8B"
clip_model_name = 'ViT-B-32'
clip_pretrained = 'laion2b_s34b_b79k'
csv_path = "data.csv"
batch_size = 4
epochs = 10
example_dsl_filename = "example.dsl.txt"
example_png_filename = "example.png"

# -------------------------
# Initialise Weights & Biases
# -------------------------
run = wandb.init(project="MLX6-image-to-dsl-003", config={
    "model": llm_model_id,
    "clip_encoder": clip_model_name,
    "clip_weights": clip_pretrained,
    "batch_size": batch_size,
    "epochs": epochs,
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lr": 5e-5
})

config = run.config 

# -------------------------
# Load LLaMA 3.1 8B tokenizer and model (with LoRA)
# -------------------------
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_tokenizer.pad_token = llm_tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

llm_model = get_peft_model(base_model, lora_config)

special_tokens = {
    "additional_special_tokens": [
        "<start_of_image>", "<end_of_image>",
        "<StartDiagram>", "<EndDiagram>",
        "<Large>", "<Medium>", "<Small>",
        "<Circle>", "<Square>", "<Triangle>",
        "<TopLeft>", "<TopRight>", "<BottomLeft>", "<BottomRight>"
    ]
}
llm_tokenizer.add_special_tokens(special_tokens)
llm_model.resize_token_embeddings(len(llm_tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    clip_model_name,
    pretrained=clip_pretrained,
    device=device
)
clip_model.eval()

# Load example image and DSL
example_dsl = Path(example_dsl_filename).read_text().strip()
example_image_tensor = clip_preprocess(Image.open(example_png_filename).convert("RGB")).unsqueeze(0).to(device)
with torch.no_grad():
    example_clip_emb = clip_model.encode_image(example_image_tensor).squeeze(0).float()
example_projected = nn.Linear(clip_model.visual.output_dim, llm_model.config.hidden_size).to(device)(example_clip_emb).unsqueeze(0)

# Projection from CLIP to LLM hidden size
clip_dim = clip_model.visual.output_dim
llm_dim = llm_model.config.hidden_size
clip_to_llm = nn.Linear(clip_dim, llm_dim).to(device)

# -------------------------
# Dataset
# -------------------------
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

        example_prompt = (
            "Here's a DSL for representing very simple diagrams:\n\n"
            "<Square> <Triangle> <Circle>\n"
            "<TopLeft> <TopRight> <BottomLeft> <BottomRight> <Large> <Medium> <Small>\n\n"
            "Using that DSL, this diagram:\n"
            "<start_of_image><end_of_image>\n"
            "corresponds to this markup:\n"
            f"{example_dsl}\n\n"
            "Now use the DSL to write the markup which would generate the following diagram:\n"
            "<start_of_image><end_of_image>"
        )
        prompt = example_prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
        label_ids = self.tokenizer(dsl, return_tensors="pt").input_ids.squeeze(0)

        image_tensor = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_emb = clip_model.encode_image(image_tensor).squeeze(0).float()
        projected_clip = clip_to_llm(clip_emb).unsqueeze(0)

        return {
            "input_ids": input_ids,
            "clip_embedding": projected_clip,
            "labels": label_ids
        }

# -------------------------
# Validation
# -------------------------
def evaluate_validation():
    llm_model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = llm_model(**batch)
            total_loss += outputs.loss.item()
            count += 1
    avg_loss = total_loss / count if count > 0 else float("inf")
    wandb.log({"val_loss": avg_loss})
    print(f"Validation loss: {avg_loss:.4f}")
    llm_model.train()

# -------------------------
# Collate Function
# -------------------------
def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    image_embeds = [b["clip_embedding"] for b in batch]

    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=llm_tokenizer.pad_token_id)
    labels_padded = torch.full_like(input_ids, fill_value=-100)
    for i, label in enumerate(labels):
        if len(label) > input_ids.shape[1]:
            raise ValueError("Label sequence is longer than input sequence")
        labels_padded[i, -len(label):] = label

    input_ids = input_ids.to(device)
    inputs_embeds = llm_model.get_input_embeddings()(input_ids)

    for i, embed in enumerate(image_embeds):
        idxs = (input_ids[i] == llm_tokenizer.convert_tokens_to_ids("<start_of_image>")).nonzero(as_tuple=True)[0]
        if len(idxs) >= 2:
            inputs_embeds[i, idxs[0]] = example_projected.squeeze(0).detach().to(inputs_embeds.dtype)
            inputs_embeds[i, idxs[1]] = embed.to(inputs_embeds.dtype)

    return {
        "inputs_embeds": inputs_embeds.to(device),
        "labels": labels_padded.to(device)
    }

# -------------------------
# Prepare Data
# -------------------------
dataset = DSLDataset(csv_path)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# -------------------------
# Inference Logging
# -------------------------
def log_sample_inference():
    llm_model.eval()
    with torch.no_grad():
        sample = random.choice(val_dataset)
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        output = llm_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 64,
            do_sample=False
        )
        decoded = llm_tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"{decoded}")
        wandb.log({"sample_prediction": decoded})

    llm_model.train()

# -------------------------
# Training Loop
# -------------------------
def train():
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
            wandb.log({"train_loss": loss.item()})
            loop.set_postfix(loss=loss.item())

        save_dir = f"checkpoints/lora-mlx6-epoch{epoch+1}"
        llm_model.save_pretrained(save_dir)
        wandb.save(f"{save_dir}/*")

        with torch.no_grad():
            log_sample_inference()
            evaluate_validation()

# -------------------------
# Start Training
# -------------------------
train()
