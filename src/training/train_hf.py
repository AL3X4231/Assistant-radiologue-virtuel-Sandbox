import os
import torch
from datasets import load_dataset, Image
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig
import argparse

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "models/medgemma-4b-4bit" 

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MedGemma (Hugging Face Standard)")
    parser.add_argument("--data_path", type=str, required=True, help="Chemin vers dataset_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/medgemma-hf")
    args = parser.parse_args()

    print("🚀 Initialisation de l'entraînement Standard (Hugging Face) sur Cluster Multi-GPU...")

    print("⏳ Chargement du modèle...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # Récupération de l'ID du GPU assigné à ce processus
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Évite que les 5 processus tentent de charger le modèle et d'initialiser CUDA 
    # à la milliseconde près (ce qui fait planter le pilote NVIDIA sur les anciennes cartes)
    import time
    print(f"⏳ Processus {local_rank} en attente de son tour pour ne pas surcharger le pilote...")
    time.sleep(local_rank * 3) # Décale le chargement de 3 secondes par carte
    
    torch.cuda.set_device(local_rank)

    # Chargement du modèle PRÉ-QUANTIFIÉ (Zéro charge sur le CPU du serveur !)
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            local_files_only=True,
            device_map={"": local_rank} 
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            local_files_only=True,
            device_map={"": local_rank}
        )

    # On active le gradient checkpointing classique HF (Beaucoup plus stable pour Pascal)
    model.gradient_checkpointing_enable()

    print("🧠 Configuration LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ==========================================
    # PRÉPARATION DU DATASET
    # ==========================================
    print(f"📊 Chargement du dataset depuis {args.data_path}...")
    dataset = load_dataset("json", data_files={"train": args.data_path})["train"]
    
    images_dir = os.path.join(os.path.dirname(args.data_path), "../raw/images")
    
    def prepare_dataset(example):
        # Format conversation -> texte pur avec le template du processeur
        text = processor.apply_chat_template(example["conversations"], tokenize=False, add_generation_prompt=False)
        img_path = os.path.join(images_dir, example["image"])
        return {"text": text, "image": img_path}

    dataset = dataset.map(prepare_dataset)
    dataset = dataset.cast_column("image", Image())
    print(f"✅ Dataset chargé : {len(dataset)} exemples.")

    # Data collator customisé pour le Vision-Language Model
    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        images = [ex["image"] for ex in examples]
        
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Le loss est calculé sur input_ids
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # ==========================================
    # ENTRAÎNEMENT
    # ==========================================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, 
        warmup_steps=50,
        num_train_epochs=3, 
        learning_rate=2e-4,
        fp16=True, 
        bf16=False, # Pascal ne supporte pas bf16
        logging_steps=10,
        optim="adamw_8bit", 
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        remove_unused_columns=False, # Vital pour que les images ne soient pas supprimées du dataset
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    print("🔥 Début du Fine-Tuning ! (Méthode HF standard)")
    trainer.train()

    print(f"✅ Entraînement terminé ! Sauvegarde dans {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("🏁 Tout est bon !")

if __name__ == "__main__":
    main()
