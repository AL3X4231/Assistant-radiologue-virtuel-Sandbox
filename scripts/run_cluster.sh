#!/bin/bash
# Script d'installation et de lancement pour le Cluster 40Go (Ubuntu/Linux)

echo "================================================="
echo "🚀 INITIALISATION DU CLUSTER POUR UNSLOTH"
echo "================================================="

# 1. Création de l'environnement virtuel si inexistant
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

echo "🔌 Activation de l'environnement virtuel..."
source venv/bin/activate

# 2. Installation des dépendances Unsloth
echo "⚙️ Installation des dépendances ultra-rapides (Unsloth)..."
# Unsloth recommande ces commandes pour une installation propre sur Linux
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# 3. Lancement de l'entraînement
echo "🔥 Lancement de l'entraînement Unsloth..."
# On utilise nohup pour que l'entraînement continue même si vous fermez Tabby/SSH
nohup python src/training/train_unsloth.py --data_path data/processed/dataset_train_10k.jsonl > logs/training_unsloth.log 2>&1 &

echo "✅ Entraînement lancé en arrière-plan !"
echo "👀 Pour voir l'avancée en temps réel, tapez :"
echo "tail -f logs/training_unsloth.log"
