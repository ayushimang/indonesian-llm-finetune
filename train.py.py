"""
Fine-tuning Llama 3.2 3B for Indonesian language tasks
Using QLoRA + ROCm on AMD GPUs
Requires: 40-80GB VRAM (MI250 or similar)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os

def main():
    print("=" * 60)
    print("Indonesian LLM Fine-tuning with ROCm")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No ROCm GPU detected. This script requires AMD GPU with ROCm.")
        return
    
    # Model configuration
    model_name = "meta-llama/Llama-3.2-3B"  # Or use "microsoft/phi-2" for smaller test
    
    # 4-bit quantization to save memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Load Indonesian dataset (example using Malay/Indonesian dataset)
    print("\nLoading Indonesian text dataset...")
    dataset = load_dataset("mesolitica/instruction-22k-indonesian", split="train[:1000]")
    
    def format_prompt(example):
        return tokenizer(
            f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./indonesian-llm-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=200,
        eval_strategy="no",
        learning_rate=2e-4,
        fp16=True,
        push_to_hub=False,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    print("\nStarting fine-tuning...")
    print("This will take 24-48 hours on AMD MI250 GPU")
    trainer.train()
    
    # Save model
    model.save_pretrained("./indonesian-llm-final")
    tokenizer.save_pretrained("./indonesian-llm-final")
    print("\n✅ Fine-tuning complete! Model saved to ./indonesian-llm-final")

if __name__ == "__main__":
    main()