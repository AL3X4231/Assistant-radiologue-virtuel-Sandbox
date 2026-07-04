import os
import time
import json
import re
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
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
OUTPUT_CSV = os.path.join(PROJECT_DIR, "data", "processed", "baseline_results.csv")
OUTPUT_PLOT = os.path.join(PROJECT_DIR, "data", "processed", "confusion_matrix.png")

RAW_PROMPT = """You are an AI vision tool designed to analyze frontal chest X-rays for educational purposes.

Analyze the chest X-ray image and respond with a single valid JSON object — no text before or after it.

IMPORTANT: Follow this reasoning process strictly, in order:
  Step 1 → Observe and list every sign directly visible on the image.
  Step 2 → Write your reasoning connecting those signs to a class.
  Step 3 → Assign the class based ONLY on what you described in Step 1.
  Step 4 → Compute confidence using the rule below.

The JSON must contain the following keys in this exact order:

- "image_quality": "bonne" | "moyenne" | "mauvaise"

- "visual_evidence": Factual, anatomically localized description of every sign directly
  visible on the image (lung zones, heart borders, pleural space, etc.).
  Do NOT infer or assume findings that are not clearly visible.
  If nothing abnormal is detected, write: "No abnormal sign detected."

- "reasoning": 2 to 3 sentences explicitly linking your visual findings to the class
  you are about to assign. Use this structure:
  "I observed [X] in [anatomical zone]. This is consistent with [Y].
  Therefore, the predicted class is [Z]."

- "predicted_class": "normal" | "suspicion_opacite" | "incertain"
  SAFETY PRINCIPLE: Detecting all pathologies is the highest priority.
  Missing a pathology is the worst possible error. Prioritize sensitivity for "suspicion_opacite"
  (class it if any abnormal finding is present, even subtle) over specificity for "normal".

  Apply these rules strictly, in priority order:

  RULE A - ABNORMAL FINDINGS:
  IF visual_evidence mentions ANY of the following:
  opacity, consolidation, infiltrate, mass, nodule, effusion, pneumothorax, cardiomegaly,
  edema, atelectasis, widened mediastinum, fracture, or ANY abnormal sign.
  THEN "predicted_class" MUST BE "suspicion_opacite".

  RULE B - NORMAL:
  IF AND ONLY IF visual_evidence explicitly states "No abnormal sign detected."
  OR describes perfectly clear lungs, normal heart, and no abnormalities.
  THEN "predicted_class" MUST BE "normal".

  RULE C - UNCERTAIN / POOR QUALITY:
  IF the image is too poor quality to read, or if findings are contradictory.
  THEN "predicted_class" MUST BE "incertain".

- "confidence": Float between 0.00 and 1.00.

- "justification": 1 sentence explaining why RULE A, B, or C was applied.

- "limitations": List any technical limits (e.g., "rotated", "poor inspiration", "subtle finding").

- "warning": MUST BE EXACTLY:
  "Outil pédagogique uniquement — aucune valeur diagnostique réelle. Ne jamais utiliser ce résultat pour une décision médicale."
"""

def get_ground_truth(df, image_name):
    # Find the row in the CSV
    row = df[df['Image Index'] == image_name]
    if row.empty:
        return "inconnu"
    
    labels = row.iloc[0]['Finding Labels']
    if "No Finding" in labels:
        return "normal"
    else:
        return "suspicion_opacite"

def extract_json_from_text(text):
    """Try to extract a JSON object from the model's text output."""
    # Find anything that looks like a JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            return data, data.get("predicted_class", "erreur_format")
        except json.JSONDecodeError:
            return {"error": "erreur_json"}, "erreur_json"
    return {"error": "pas_de_json"}, "pas_de_json"

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # 1. Load Metadata
    print("Loading Ground Truth from CSV...")
    df = pd.read_csv(CSV_FILE)
    
    images = [f for f in os.listdir(TEST_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(images)} images in {TEST_DIR}.")
    
    # 2. Load Model in 4-bit (Baseline)
    print("Loading Base Model in 4-bit...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    # Important: using device_map={"": 0} to force it on GPU for 1660 Ti
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    model.eval()

    # 3. Inference Loop
    results = []
    start_time_total = time.time()
    
    print("Starting Inference Loop...")
    for img_name in tqdm(images, desc="Evaluating images"):
        img_path = os.path.join(TEST_DIR, img_name)
        gt_class = get_ground_truth(df, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {img_name}: {e}")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": RAW_PROMPT}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        # We only generate up to 512 tokens to save time
        start_img_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False # Greedy decoding for reproducible results
            )
        img_duration = time.time() - start_img_time
        
        # Decode output
        generated_text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        json_data, pred_class = extract_json_from_text(generated_text)
        
        # Standardize predictions to strictly "normal" or "suspicion_opacite" for the matrix
        # If the model fails or says "incertain", we might treat it as a failure to predict.
        # For medical safety, usually "incertain" or errors are grouped with "suspicion_opacite" to force review.
        standardized_pred = "suspicion_opacite"
        if pred_class and isinstance(pred_class, str) and pred_class.lower().strip() == "normal":
            standardized_pred = "normal"
        
        row_data = {
            "image_name": img_name,
            "ground_truth": gt_class,
            "raw_prediction": pred_class,
            "standardized_prediction": standardized_pred,
            "time_seconds": round(img_duration, 2),
            "image_quality": json_data.get("image_quality", ""),
            "predicted_class": json_data.get("predicted_class", ""),
            "confidence": json_data.get("confidence", ""),
            "visual_evidence": json_data.get("visual_evidence", ""),
            "justification": json_data.get("justification", ""),
            "limitations": json_data.get("limitations", ""),
            "warning": json_data.get("warning", ""),
            "error": json_data.get("error", "")
        }
        results.append(row_data)

    # Save to CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

    # 4. Compute Metrics
    end_time_total = time.time()
    total_time_minutes = (end_time_total - start_time_total) / 60.0
    avg_time_per_image = res_df["time_seconds"].mean()
    
    print("\n" + "="*50)
    print(f"EVALUATION COMPLETED IN {total_time_minutes:.2f} MINUTES")
    print(f"AVERAGE TIME PER IMAGE: {avg_time_per_image:.2f} SECONDS")
    print("="*50 + "\n")

    # Filter out any "inconnu" ground truths (images not in CSV) just in case
    eval_df = res_df[res_df["ground_truth"] != "inconnu"]
    
    y_true = eval_df["ground_truth"]
    y_pred = eval_df["standardized_prediction"]
    
    # We define "suspicion_opacite" as Positive (1) and "normal" as Negative (0)
    labels = ["normal", "suspicion_opacite"]
    
    if not eval_df.empty:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = accuracy_score(y_true, y_pred)
        # Sensitivity = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"Total Images Evaluated: {len(eval_df)}")
        print(f"Vrais Positifs (TP): {tp} (Malades bien détectés)")
        print(f"Vrais Négatifs (TN): {tn} (Sains bien détectés)")
        print(f"Faux Positifs (FP): {fp} (Sains détectés malades)")
        print(f"Faux Négatifs (FN): {fn} (Malades non détectés - DANGER)")
        print("-" * 30)
        print(f"Accuracy (Précision globale): {accuracy:.2%}")
        print(f"Sensitivity (Rappel / Vrais malades trouvés): {sensitivity:.2%}")
        print(f"Specificity (Vrais sains trouvés): {specificity:.2%}")

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Pred: Normal', 'Pred: Suspicion'], 
                    yticklabels=['Vrai: Normal', 'Vrai: Suspicion'])
        plt.title(f"Matrice de Confusion (Base Model)\nAcc: {accuracy:.2%} | Sens: {sensitivity:.2%} | Spec: {specificity:.2%}")
        plt.ylabel("Vérité Terrain")
        plt.xlabel("Prédiction du Modèle")
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT)
        print(f"Confusion Matrix saved to {OUTPUT_PLOT}")
    else:
        print("No valid evaluation data found.")

if __name__ == "__main__":
    main()
