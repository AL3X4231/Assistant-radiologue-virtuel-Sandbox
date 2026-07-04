import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import FastVisionModel
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

    print("🚀 Initialisation de l'entraînement Unsloth sur cluster 40Go...")
    
    # Vérification BFloat16 (Fortement recommandé pour les A100/H100/RTX 30xx+)
    bfloat16_support = is_bfloat16_supported()
    dtype = torch.bfloat16 if bfloat16_support else torch.float16
    print(f"✅ Type de précision utilisé : {dtype}")

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
            load_in_4bit = False, # On met False car on a 40 Go de VRAM ! On veut la qualité max (16-bit)
            use_gradient_checkpointing = "unsloth", 
            dtype = dtype
        )
    except Exception as e:
        print("⚠️ Le modèle ne semble pas être reconnu comme un modèle Vision par Unsloth. Tentative en mode Texte pur...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_ID,
            load_in_4bit = False, 
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
    print(f"✅ Dataset chargé : {len(dataset)} exemples.")

    # ==========================================
    # 4. CONFIGURATION DE L'ENTRAÎNEMENT
    # ==========================================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4, # Avec 40Go, on peut monter le batch size !
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=3, # 3 epochs pour voir si on atteint les 75-80%
        learning_rate=2e-4,
        fp16=not bfloat16_support,
        bf16=bfloat16_support,
        logging_steps=10,
        optim="adamw_8bit", # Optimiseur très léger en VRAM
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none" # Mettez "wandb" ou "tensorboard" si vous voulez des graphiques
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="conversations", # Dépend du format (Llava utilise 'conversations')
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
