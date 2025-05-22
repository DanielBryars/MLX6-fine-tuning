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

# -------------------------
# Initialise Weights & Biases
# -------------------------
wandb.init(project="MLX6-image-to-dsl-001", config={
    "model": llm_model_id,
    "clip_encoder": clip_model_name,
    "clip_weights": clip_pretrained,
    "batch_size": batch_size,
    "epochs": epochs,
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "accum_steps": 4,
    "lr": 5e-5
})

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

start_image_token_id = llm_tokenizer.convert_tokens_to_ids("<start_of_image>")
end_image_token_id = llm_tokenizer.convert_tokens_to_ids("<end_of_image>")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    clip_model_name,
    pretrained=clip_pretrained,
    device=device
)
clip_model.eval()

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

        prompt = (
            "Here's a DSL for representing very simple diagrams,\n\n"
            "<Square> <Triangle> <Circle>\n"
            "<TopLeft> <TopRight> <BottomLeft> <BottomRight> <Large> <Medium> <Small>\n\n"
            "Now write the DSL that corresponds to the following diagram:\n"
            "<start_of_image><end_of_image>"
        )
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
        label_ids = self.tokenizer(dsl, return_tensors="pt").input_ids.squeeze(0)

        image_tensor = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_emb = clip_model.encode_image(image_tensor).squeeze(0).float()
        projected_clip = clip_to_llm(clip_emb).unsqueeze(0)  # shape: (1, hidden_dim)

        return {
            "input_ids": input_ids,
            "clip_embedding": projected_clip,
            "labels": label_ids
        }

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
        labels_padded[i, -len(label):] = label

    input_ids = input_ids.to(device)
    inputs_embeds = llm_model.get_input_embeddings()(input_ids)

    for i, embed in enumerate(image_embeds):
        # Find the <start_of_image> token and insert the image embedding
        idx = (input_ids[i] == start_image_token_id).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            inputs_embeds[i, idx[0]] = embed.to(inputs_embeds.dtype)

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
        wandb.log({"sample_prediction": decoded})
    llm_model.train()

# -------------------------
# Training Loop
# -------------------------
def train():
    model = llm_model.train()
    from torch.cuda.amp import autocast, GradScaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()
    step_count = 0

    for epoch in range(epochs):
        log_sample_inference()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(loop):
            if step_count % 4 == 0:
                optimizer.zero_grad()
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss / 4
            scaler.scale(loss).backward()
            if (step_count + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.empty_cache()
                wandb.log({"train_loss": loss.item()})
                loop.set_postfix(loss=loss.item())
            step_count += 1

        save_dir = f"checkpoints/lora-mlx6-epoch{epoch+1}"
        llm_model.save_pretrained(save_dir)
        wandb.save(f"{save_dir}/*")

# -------------------------
# Start Training
# -------------------------
train()
