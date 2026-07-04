# Assistant Radiologue Virtuel (Baseline)

Ce projet permet d'évaluer les performances du modèle "vision-language" `google/medgemma-4b-it` sur l'analyse de radiographies thoraciques, en utilisant un jeu de 150 images de test extraites du dataset NIH ChestX-ray14.

Le but de ce dépôt est de pouvoir générer une **Matrice de Confusion** et des statistiques médicales (Sensibilité, Spécificité) pour évaluer la "Baseline" de l'IA (avant tout processus de fine-tuning).

## 🛠️ Prérequis
- Python 3.10 ou supérieur (testé sur 3.13)
- Une carte graphique NVIDIA avec au moins 6 Go de VRAM (8 Go+ recommandés).
- (Windows) Les drivers NVIDIA à jour.

## 🚀 Installation & Workflow

Suivez ces étapes pour installer et lancer l'évaluation sur votre machine.

### 1. Cloner le projet
```bash
git clone <VOTRE_LIEN_GITHUB>
cd Assistant-radiologue-virtuel-Sandbox
```

### 2. Télécharger le fichier CSV Ground Truth
Puisque le dataset médical d'origine pèse très lourd, le dossier `data/` n'est pas versionné sur GitHub. 
Cependant, le script d'évaluation a besoin du fichier contenant les vrais diagnostics pour comparer avec les réponses de l'IA.
- Procurez-vous le fichier original `Data_Entry_2017.csv` (issu du NIH Dataset).
- Créez les dossiers manquants et placez-y le fichier à cet endroit précis : 
  `data/raw/metadata/Data_Entry_2017.csv`

*(Note : Le dossier `test_samples/` contenant les 150 images à évaluer est déjà inclus dans ce dépôt).*

### 3. Créer un environnement virtuel
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(Sur Mac/Linux : `source venv/bin/activate`)*

### 4. Installer PyTorch (Version GPU / CUDA)
**Étape très importante !** N'installez pas PyTorch avec la commande par défaut, sinon il utilisera votre CPU (ce qui serait infiniment lent). Forcez l'installation de la version CUDA compatible avec votre système (ex: CUDA 12.1 ou 12.4).
```powershell
# Exemple pour CUDA 12.4 :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 5. Installer les dépendances du projet
Une fois PyTorch installé correctement, installez le reste des outils (HuggingFace, Pandas, Scikit-learn, etc.) :
```powershell
pip install -r requirements.txt
```

### 6. Lancer l'Évaluation
Il ne reste plus qu'à démarrer l'inférence. Le modèle va analyser les 150 images de `test_samples/` (cela peut prendre 15 à 30 minutes selon votre carte graphique).
```powershell
python src/evaluation/evaluate_baseline.py
```

### 📊 Résultats
Une fois le script terminé, vous trouverez deux nouveaux éléments générés dans le projet :
1. `data/processed/baseline_results.csv` : L'historique complet, image par image, de ce que l'IA a répondu par rapport à la réalité.
2. `data/processed/confusion_matrix.png` : L'image graphique présentant la matrice de confusion globale avec les pourcentages finaux de Sensibilité et Spécificité !
