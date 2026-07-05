#!/bin/bash
# Script d'installation et de lancement pour le Cluster Multi-GPU (5x 1070 Ti)

echo "================================================="
echo "🚀 INITIALISATION DU CLUSTER MULTI-GPU (5x 8Go)"
echo "================================================="

# 1. Création de l'environnement virtuel si inexistant
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

echo "🔌 Activation de l'environnement virtuel..."
source venv/bin/activate

# 2. Installation des dépendances Unsloth & Accelerate
echo "⚙️ Installation de PyTorch (Version compatible CUDA 12.1 pour vos anciens drivers)..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "⚙️ Installation de xformers (après PyTorch)..."
pip install xformers --index-url https://download.pytorch.org/whl/cu121

echo "⚙️ Installation de Unsloth et Accelerate..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install "trl<0.9.0" peft accelerate bitsandbytes

# Configuration automatique d'Accelerate pour utiliser toutes les cartes disponibles
echo "⚙️ Configuration de Accelerate (Multi-GPU)..."
accelerate config default

# 3. Lancement de l'entraînement
echo "🔥 Lancement de l'entraînement Multi-GPU..."
# On utilise accelerate launch pour distribuer la charge sur toutes les cartes
echo "🚀 Lancement de l'entraînement avec Hugging Face Standard..."
nohup accelerate launch --num_processes 5 src/training/train_hf.py --data_path data/processed/dataset_train_10k.jsonl > logs/training_hf.log 2>&1 &

echo "✅ Entraînement lancé en arrière-plan sur les 5 cartes !"
echo "👀 Pour voir l'avancée en temps réel, tapez :"
echo "tail -f logs/training_unsloth.log"
