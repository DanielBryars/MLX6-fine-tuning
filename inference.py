import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import open_clip
from PIL import Image
from pathlib import Path

# -------------------------
# Configuration
# -------------------------
llm_model_id = "meta-llama/Llama-3.1-8B"
clip_model_name = 'ViT-B-32'
clip_pretrained = 'laion2b_s34b_b79k'

# -------------------------
# Load LLaMA 3.1 8B tokenizer and model
# -------------------------
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# -------------------------
# Add special tokens
# -------------------------
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

# -------------------------
# Load CLIP model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    clip_model_name,
    pretrained=clip_pretrained,
    device=device
)
clip_model.eval()

# -------------------------
# Prompt Assembler Class
# -------------------------
class CLIPPromptAssembler(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, clip_dim: int = 512, max_seq_len: int = 128,
                 start_image_token_id: int = None, end_image_token_id: int = None):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.clip_proj = nn.Linear(clip_dim, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        if start_image_token_id is None or end_image_token_id is None:
            raise ValueError("start_image_token_id and end_image_token_id must be provided")

        self.start_image_token_id = start_image_token_id
        self.end_image_token_id = end_image_token_id

    def forward(self, full_prompt_ids, example_image_clip, target_image_clip):
        B = full_prompt_ids.size(0)

        start_tok = torch.full((B, 1), self.start_image_token_id, dtype=torch.long, device=full_prompt_ids.device)
        end_tok = torch.full((B, 1), self.end_image_token_id, dtype=torch.long, device=full_prompt_ids.device)

        start_emb = self.token_embed(start_tok)
        end_emb = self.token_embed(end_tok)

        prompt_emb = self.token_embed(full_prompt_ids)
        example_img_emb = self.clip_proj(example_image_clip).unsqueeze(1)
        target_img_emb = self.clip_proj(target_image_clip).unsqueeze(1)

        full_stream = torch.cat([
            prompt_emb,
            start_emb, example_img_emb, end_emb,
            start_emb, target_img_emb, end_emb
        ], dim=1)

        total_len = full_stream.size(1)
        pos_ids = torch.arange(total_len, device=full_prompt_ids.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos_ids)

        return full_stream + pos_emb

# -------------------------
# Helper to load and encode image
# -------------------------
def encode_image(image_path: Path):
    image = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image).float()
    return image_features

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    d_model = llm_model.config.hidden_size
    vocab_size = len(llm_tokenizer)
    assembler = CLIPPromptAssembler(
        vocab_size=vocab_size,
        d_model=d_model,
        clip_dim=512,
        max_seq_len=512,
        start_image_token_id=start_image_token_id,
        end_image_token_id=end_image_token_id
    ).to(device)

    # Build structured prompt with DSL explanation and example
    example_dsl = Path("example.dsl.txt").read_text().strip()

    explanation = (
        "Here's a DSL for representing very simple diagrams,\n\n"
        "<Square> <Triangle> <Circle>\n"
        "<TopLeft> <TopRight> <BottomLeft> <BottomRight> <Large> <Medium> <Small>\n\n"
        "For example:\n"
        f"{example_dsl}\n\n"
        "would result in the following diagram:\n"
        "<start_of_image>"
    )

    final_instruction = ("\nNow write the DSL that corresponds to the next diagram:\n<start_of_image>")
    full_prompt_text = explanation + "<end_of_image>" + final_instruction

    with torch.no_grad():
        full_prompt_ids = llm_tokenizer(full_prompt_text, return_tensors="pt").input_ids.to(device)

        example_clip = encode_image(Path("example.png"))
        target_clip = encode_image(Path("reference.png"))

        embedded_prompt = assembler(full_prompt_ids, example_clip, target_clip).to(torch.float16)
        print("Assembled prompt shape:", embedded_prompt.shape)

        attention_mask = torch.ones(embedded_prompt.shape[:2], dtype=torch.long, device=embedded_prompt.device)

        # Generate output DSL
        output_ids = llm_model.generate(
            inputs_embeds=embedded_prompt,
            attention_mask=attention_mask,
            max_length=embedded_prompt.shape[1] + 64,
            do_sample=False
        )
        output_text = llm_tokenizer.decode(output_ids[0], skip_special_tokens=False)
        print("\nGenerated DSL for reference image:\n", output_text)

        # Extract predicted DSL portion (assume it starts with <StartDiagram>)
        start = output_text.find("<StartDiagram>")
        end = output_text.find("<EndDiagram>")
        if start != -1 and end != -1:
            predicted_dsl = output_text[start:end+len("<EndDiagram>")]
            print("\nParsed DSL:\n", predicted_dsl)

            # Render the image from the predicted DSL
            from renderer import render_dsl  # assume this returns a PIL Image
            rendered_image = render_dsl(predicted_dsl)
            rendered_image.save("predicted.png")
            print("Saved rendered DSL to predicted.png")
        else:
            print("Could not extract DSL from model output.")
