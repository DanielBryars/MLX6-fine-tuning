import torch
import open_clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Load model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)  # Add batch dim

# Get image embedding
@torch.no_grad()
def get_clip_image_embedding(image_tensor):
    image_features = model.encode_image(image_tensor)
    return image_features / image_features.norm(dim=-1, keepdim=True)  # Normalised

# Example usage
if __name__ == "__main__":
    image_tensor = load_image("your_image.png")
    embedding = get_clip_image_embedding(image_tensor)
    print("Image embedding shape:", embedding.shape)
