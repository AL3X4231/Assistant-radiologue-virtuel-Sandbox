import os
import time
import json
import re
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "google/medgemma-4b-it"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

TEST_DIR = os.path.join(PROJECT_DIR, "test_samples")
CSV_FILE = os.path.join(PROJECT_DIR, "data", "raw", "metadata", "Data_Entry_2017.csv")
OUTPUT_CSV = os.path.join(PROJECT_DIR, "data", "processed", "pipeline_results.csv")
OUTPUT_PLOT = os.path.join(PROJECT_DIR, "data", "processed", "confusion_matrix_pipeline.png")

# ==========================================
# PROMPTS SÉPARÉS
# ==========================================

# Étape 1 : Le Classifieur pur
CLASSIFIER_PROMPT = """You are an AI vision tool designed to analyze frontal chest X-rays.
Analyze the chest X-ray image and return ONLY a single valid JSON object.
Your ONLY task is to detect if there is any abnormal sign (opacity, mass, effusion, cardiomegaly, etc.) or if it is strictly normal.

The JSON must contain exactly one key:
{
  "predicted_class": "normal" | "suspicion_opacite" | "incertain"
}
"""

# Étape 2 : L'Explainer (Générateur de rapport)
EXPLAINER_PROMPT_TEMPLATE = """You are an AI vision tool designed to write medical reports for frontal chest X-rays.
The classification for this image has already been confirmed by the primary system as: {predicted_class}.
DO NOT CONTRADICT THIS CLASSIFICATION. Your job is to provide the reasoning and visual evidence that led to this classification.

Return ONLY a single valid JSON object containing exactly these keys:
{
  "image_quality": "bonne" | "moyenne" | "mauvaise",
  "visual_evidence": "Factual description of the signs visible on the image that explain the class.",
  "confidence": 0.95, 
  "justification": "1 sentence explaining why this classification makes sense given the visual evidence.",
  "limitations": "List any technical limits (e.g. rotated, poor inspiration)."
}
"""

WARNING_TEXT = "Outil pédagogique uniquement — aucune valeur diagnostique réelle. Ne jamais utiliser ce résultat pour une décision médicale."

def get_ground_truth(df, image_name):
    row = df[df['Image Index'] == image_name]
    if row.empty:
        return "inconnu"
    labels = row.iloc[0]['Finding Labels']
    if "No Finding" in labels:
        return "normal"
    else:
        return "suspicion_opacite"

def extract_json_from_text(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    return {}

def run_inference(processor, model, image, prompt_text):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
    generated_text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return extract_json_from_text(generated_text)

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    print("Loading Ground Truth from CSV...")
    df = pd.read_csv(CSV_FILE)
    
    images = [f for f in os.listdir(TEST_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(images)} images in {TEST_DIR}.")
    
    print("Loading Base Model in 16-bit (Optimized for Cluster 40GB)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # Précision BFloat16 recommandée pour cluster
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto"
    )
    model.eval()

    results = []
    start_time_total = time.time()
    
    print("Starting Pipeline Inference Loop (2 steps per image)...")
    for img_name in tqdm(images, desc="Evaluating images"):
        img_path = os.path.join(TEST_DIR, img_name)
        gt_class = get_ground_truth(df, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {img_name}: {e}")
            continue

        start_img_time = time.time()
        
        # STEP 1: CLASSIFIER
        class_json = run_inference(processor, model, image, CLASSIFIER_PROMPT)
        predicted_class = class_json.get("predicted_class", "incertain")
        
        # STEP 2: EXPLAINER
        explainer_prompt = EXPLAINER_PROMPT_TEMPLATE.format(predicted_class=predicted_class)
        explainer_json = run_inference(processor, model, image, explainer_prompt)
        
        img_duration = time.time() - start_img_time
        
        # Standardisation pour la matrice de confusion
        standardized_pred = "suspicion_opacite"
        if isinstance(predicted_class, str) and predicted_class.lower().strip() == "normal":
            standardized_pred = "normal"
        
        # Fusion des données
        row_data = {
            "image_name": img_name,
            "ground_truth": gt_class,
            "raw_prediction": predicted_class,
            "standardized_prediction": standardized_pred,
            "time_seconds": round(img_duration, 2),
            "image_quality": explainer_json.get("image_quality", ""),
            "predicted_class": predicted_class,
            "confidence": explainer_json.get("confidence", 0.0),
            "visual_evidence": explainer_json.get("visual_evidence", ""),
            "justification": explainer_json.get("justification", ""),
            "limitations": explainer_json.get("limitations", ""),
            "warning": WARNING_TEXT,
            "error": "" if predicted_class else "Erreur de classification JSON"
        }
        results.append(row_data)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Metrics
    end_time_total = time.time()
    print(f"TOTAL EVALUATION TIME: {(end_time_total - start_time_total) / 60.0:.2f} MINUTES")
    print(f"AVERAGE TIME PER IMAGE: {res_df['time_seconds'].mean():.2f} SECONDS")
    
    eval_df = res_df[res_df["ground_truth"] != "inconnu"]
    if not eval_df.empty:
        cm = confusion_matrix(eval_df["ground_truth"], eval_df["standardized_prediction"], labels=["normal", "suspicion_opacite"])
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred: Normal', 'Pred: Suspicion'], yticklabels=['Vrai: Normal', 'Vrai: Suspicion'])
        plt.title(f"Matrice de Confusion (Pipeline 2-Steps)\nAcc: {accuracy:.2%} | Sens: {sensitivity:.2%} | Spec: {specificity:.2%}")
        plt.ylabel("Vérité Terrain")
        plt.xlabel("Prédiction du Modèle")
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT)
        print(f"Confusion Matrix saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
