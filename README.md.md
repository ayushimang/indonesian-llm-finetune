# Indonesian LLM Fine-tuning on AMD ROCm

Fine-tuning Llama 3.2 (3B) for Indonesian language tasks using QLoRA on AMD GPUs with ROCm.

## Why AMD GPU?

- **Memory requirement:** Fine-tuning 3B model with QLoRA needs 40-80GB VRAM
- **AMD MI250/MI300** provides the necessary memory for full fine-tuning
- This training cannot run on consumer GPUs (RTX 4090 only 24GB)

## Requirements

- AMD GPU with ROCm support (MI250 or MI300 recommended)
- 80GB+ VRAM
- ROCm 7.0+

## Quick Start

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/indonesian-llm-finetune
cd indonesian-llm-finetune

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py