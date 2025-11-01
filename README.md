# MLX6-fine-tuning

Image-to-DSL fine-tuning project using LLaMA 3.1 8B with CLIP vision encoder and LoRA for parameter-efficient training.

## Overview

This project fine-tunes a large language model (LLaMA 3.1 8B) to generate Domain Specific Language (DSL) markup from images of simple diagrams. The system uses:

- **CLIP vision encoder** to extract visual features from diagram images
- **LLaMA 3.1 8B** as the base language model for text generation
- **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
- **Few-shot prompting** with example diagrams to guide generation

## Architecture

```
Image → CLIP Encoder → Projection Layer → LLaMA 3.1 8B (LoRA) → DSL Output
```

The model takes diagram images as input and generates structured markup describing the shapes, sizes, and positions in a custom DSL.

### DSL Format

The DSL supports:
- **Shapes**: `<Circle>`, `<Square>`, `<Triangle>`
- **Sizes**: `<Large>`, `<Medium>`, `<Small>`
- **Positions**: `<TopLeft>`, `<TopRight>`, `<BottomLeft>`, `<BottomRight>`

Example:
```
<StartDiagram>
  <Large> <Circle> <TopLeft>
  <Medium> <Square> <TopRight>
  <Small> <Triangle> <BottomLeft>
<EndDiagram>
```

## Features

- Parameter-efficient fine-tuning with LoRA (r=8, alpha=32)
- CLIP ViT-B-32 encoder pretrained on LAION-2B
- Custom projection layer from CLIP embedding space to LLaMA hidden space
- Few-shot prompting with example images embedded in context
- Weights & Biases integration for experiment tracking
- Train/validation split with periodic evaluation
- Checkpoint saving per epoch

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch
- Transformers (HuggingFace)
- PEFT (Parameter-Efficient Fine-Tuning)
- open-clip-torch
- wandb
- Pillow

## Data Format

The training data is structured as CSV with columns:
- `image_path`: Path to diagram image
- `dsl`: Corresponding DSL markup

## Training

### Configuration

Key hyperparameters in `finetune.py`:
- **Model**: `meta-llama/Llama-3.1-8B`
- **CLIP**: `ViT-B-32` from LAION-2B
- **Batch size**: 4
- **Epochs**: 10
- **LoRA rank**: 8
- **Learning rate**: 5e-5

### Run Training

```bash
python finetune.py
```

This will:
1. Load the LLaMA 3.1 8B model with LoRA adapters
2. Load CLIP encoder and create projection layer
3. Train on `data.csv` with 90/10 train/val split
4. Log metrics to Weights & Biases
5. Save checkpoints to `checkpoints/lora-mlx6-epoch{N}/`
6. Periodically log sample inferences and validation loss

### Training Features

- **Gradient accumulation** handled via batch processing
- **Mixed precision** (FP16) for efficient GPU usage
- **Special tokens** for diagram elements and image boundaries
- **Example embedding** injected at first `<start_of_image>` token
- **Target embedding** injected at second `<start_of_image>` token

## Inference

```bash
python inference.py
```

Loads a trained checkpoint and generates DSL from test images.

## Additional Scripts

- **`train.py`** - Alternative training script
- **`overfittest.py`** - Overfitting test on small dataset
- **`create_images.py`** - Generate synthetic diagram images from DSL
- **`renderer.py`** - Render diagrams from DSL markup
- **`clip_test.py`** - Test CLIP encoder functionality
- **`llama_test.py`** - Test LLaMA model loading
- **`gpucheck.py`** - Check GPU availability

## Dataset Generation

Generate synthetic training data:

```bash
python create_images.py
```

This creates diagram images from DSL specifications and populates `data.csv`.

## Project Structure

```
MLX6-fine-tuning/
├── finetune.py              # Main training script
├── train.py                 # Alternative training script
├── inference.py             # Inference script
├── create_images.py         # Dataset generation
├── renderer.py              # DSL to image renderer
├── data.csv                 # Training data
├── images/                  # Generated diagram images
├── checkpoints/             # Saved model checkpoints
├── docs/                    # Documentation and papers
│   ├── Design.png          # Architecture diagram
│   └── *.pdf               # Reference papers
├── example.dsl.txt          # Example DSL markup
├── example.png              # Example diagram image
└── requirements.txt         # Python dependencies
```

## Experiment Tracking

The project uses Weights & Biases for tracking:
- Training and validation loss
- Sample predictions per epoch
- Hyperparameters
- Model checkpoints

View experiments at: `wandb.ai/<your-username>/MLX6-image-to-dsl-003`

## Model Architecture Details

### LoRA Configuration
- Target modules: `q_proj`, `v_proj` (query and value projections in attention)
- Rank: 8 (low-rank matrices)
- Alpha: 32 (scaling factor)
- Dropout: 0.1

### CLIP Integration
- Vision encoder: ViT-B-32
- Output dimension: 512
- Projected to LLaMA hidden size (4096 for 8B model)
- Frozen during training (only projection layer trained)

## Use Cases

- Learning image-to-code generation
- Understanding multimodal fine-tuning
- Experimenting with LoRA and parameter-efficient methods
- Building diagram understanding systems
- Studying vision-language model integration

## References

Papers in `docs/`:
- Direct Preference Optimization (DPO)
- Fine-tuning methodologies

## Notes

- Requires CUDA-capable GPU (model uses FP16)
- LLaMA 3.1 8B requires HuggingFace access token
- ~16GB GPU memory required for training
- Training time: ~1-2 hours for 10 epochs on RTX 4090
