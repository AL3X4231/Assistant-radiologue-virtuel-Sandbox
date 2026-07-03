import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, prepare_model_for_kbit_training

from src.data.dataset import ChestXrayDataset
from src.models.lora_config import get_quantization_config, get_lora_config

# Constants
MODEL_ID = "google/medgemma-4b-it" 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")
DATASET_FILE = os.path.join(PROCESSED_DIR, "dataset_train.jsonl")
IMAGES_DIR = os.path.join(BASE_DIR, "..", "..", "data", "raw", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "checkpoints", "medgemma-qlora")

def main():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    print("Loading model in 4-bit...")
    quant_config = get_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map={"": 0} # Force placement on GPU 0 to prevent bitsandbytes CPU offload errors
    )
    
    print("Preparing model for LoRA training...")
    # Enable gradient checkpointing for VRAM savings
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Loading dataset...")
    # Using max_length=512 to save memory. 
    # VLM context can be long because of image patches, so monitor if 512 is enough.
    dataset = ChestXrayDataset(DATASET_FILE, IMAGES_DIR, processor, max_length=512)
    
    # 6GB VRAM is very tight. We must use batch_size=1, gradient accumulation, and paged_adamw_8bit
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16, # Simulate batch size of 16
        optim="paged_adamw_8bit", # Saves VRAM by paging optimizer states
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        num_train_epochs=1, # Adjust based on time
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        fp16=True, # Use Mixed Precision
        remove_unused_columns=False, # Important for multimodal datasets in HF
        report_to="none" # Switch to "tensorboard" if you want to track metrics
    )
    
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving final adapter to {OUTPUT_DIR}/final")
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

if __name__ == "__main__":
    main()
