import os
import torch
from datasets import load_dataset, Image
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer
from transformers import TrainingArguments
import argparse

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "google/medgemma-4b-it" 
MAX_SEQ_LENGTH = 2048 # Peut être augmenté si besoin (Unsloth gère très bien les longs contextes)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MedGemma avec Unsloth (16-bit LoRA)")
    parser.add_argument("--data_path", type=str, required=True, help="Chemin vers dataset_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/medgemma-unsloth")
    args = parser.parse_args()

    print("🚀 Initialisation de l'entraînement Unsloth sur Cluster Multi-GPU (5x 8Go)...")
    
    # Force float16 car les cartes Pascal (GTX 1070 Ti) ne supportent pas bfloat16
    dtype = torch.float16
    print(f"✅ Type de précision forcé : {dtype} (Pascal Architecture)")

    # ==========================================
    # 1. CHARGEMENT DU MODÈLE (UNSLOTH)
    # ==========================================
    print("⏳ Chargement du modèle...")
    # NOTE: Si medgemma est un modèle multimodal (images), on tente d'utiliser FastVisionModel.
    # Si c'est un modèle texte classique (qui reçoit les images sous forme d'embedding préalable), 
    # vous devrez peut-être utiliser FastLanguageModel à la place.
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name = MODEL_ID,
            load_in_4bit = True, # OBLIGATOIRE pour tenir sur 8 Go de VRAM
            use_gradient_checkpointing = "unsloth", 
            dtype = dtype
        )
    except Exception as e:
        print("⚠️ Le modèle ne semble pas être reconnu comme un modèle Vision par Unsloth. Tentative en mode Texte pur...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_ID,
            load_in_4bit = True, 
            dtype = dtype
        )

    # ==========================================
    # 2. CONFIGURATION LORA (PEFT)
    # ==========================================
    print("🧠 Configuration des adaptateurs LoRA...")
    if hasattr(FastVisionModel, "get_peft_model"):
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers = False, # On gèle la vision pour n'entraîner que la logique médicale (plus rapide)
            finetune_language_layers = True,
            r = 16, # Rank LoRA (16 est un excellent compromis)
            lora_alpha = 16,
            lora_dropout = 0, # Unsloth optimise avec 0 dropout
            bias = "none",
        )
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0, 
            bias = "none",
        )

    # ==========================================
    # 3. PRÉPARATION DU DATASET
    # ==========================================
    print(f"📊 Chargement du dataset depuis {args.data_path}...")
    dataset = load_dataset("json", data_files={"train": args.data_path})["train"]
    
    # 3.1 Chargement réel des images (requis pour Vision)
    images_dir = os.path.join(os.path.dirname(args.data_path), "../raw/images")
    def format_image_path(example):
        example["image"] = os.path.join(images_dir, example["image"])
        return example
        
    dataset = dataset.map(format_image_path)
    dataset = dataset.cast_column("image", Image())
    
    print(f"✅ Dataset chargé : {len(dataset)} exemples.")

    # ==========================================
    # 4. CONFIGURATION DE L'ENTRAÎNEMENT
    # ==========================================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1, # 1 seule image par carte pour ne pas exploser les 8 Go
        gradient_accumulation_steps=8, # On simule un batch size de 8 (1 * 8)
        warmup_steps=50,
        num_train_epochs=3, 
        learning_rate=2e-4,
        fp16=True, # Forcé pour Pascal
        bf16=False,
        logging_steps=10,
        optim="adamw_8bit", # Optimiseur très léger en VRAM
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        remove_unused_columns=False, # IMPORTANT pour les modèles Vision
        report_to="none" 
    )

    FastVisionModel.for_training(model) # Obligatoire pour Unsloth Vision

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        dataset_text_field="", # Doit être vide pour la vision
        dataset_kwargs={"skip_prepare_dataset": True}, # Empêche TRL de supprimer l'image
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # ==========================================
    # 5. LANCEMENT DU FINE-TUNING
    # ==========================================
    print("🔥 Début du Fine-Tuning ! (Allez prendre un café ☕)")
    trainer_stats = trainer.train()

    print(f"✅ Entraînement terminé ! Sauvegarde dans {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("🏁 Tout est bon !")

if __name__ == "__main__":
    main()
