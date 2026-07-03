import os
import json
import re
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.streamers import BaseStreamer # <-- L'importation corrigée est ici
from PIL import Image
from tqdm import tqdm

# ==========================================
# CONFIGURATION ET PROMPT
# ==========================================
MODEL_ID = "google/medgemma-4b-it"

# Chemin relatif au script (recherche dans le dossier data/raw/images)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw", "images", "00000003_000.png")

# Le texte brut de tes instructions
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

  1. "suspicion_opacite" → YOUR PRIMARY DETECTOR CLASS. Use it for ANY abnormal
     parenchymal or pleural finding, no matter how subtle. This includes:
     opacity, consolidation, pneumonia-like findings, infiltrate (even very subtle),
     effusion, atelectasis, mass, nodule, pleural thickening, pneumothorax,
     or borderline heart size with any other subtle sign.
     If you are 100% certain a finding is present, use this class.

  2. "normal" → ONLY if visual_evidence contains ZERO signs of active parenchyma
     or pleural pathology. Minor interstitial markings within typical ranges,
     old healed findings, or clear non-pathological vascular markings should NOT
     prevent this classification.
     This requires "bonne" quality and confidence >= 0.85. If a patient is truly sick
     and you class "normal," you have failed. When in any doubt about presence, choose
     "suspicion_opacite".

  3. "incertain" → YOUR LAST RESORT CLASS. Use only when:
     - The visual signs are contradictory or completely uninterpretable (e.g., severe
       artifact obscuring the critical zone, major rotation).
     - The image quality is "mauvaise" to the point where even a general assessment of
       lung fields is impossible.
     - You are split between two different *abnormal* interpretations of a *clear* finding.
     Do NOT use "incertain" for subtle or borderline findings of opacities
     (use "suspicion_opacite"). Do NOT use "incertain" for subtle interstitial markings
     in an otherwise clear field (use "normal" if within typical ranges).

- "confidence": A float between 0.0 and 1.0, computed using this rule:
  0.85 – 0.95 → multiple clear, distinct signs OR a single clear, definite sign.
  0.60 – 0.80 → single sign that is subtle OR significant artifacts / poor image quality.
  0.40 – 0.60 → uninterpretable image OR extreme artifacts.

- "justification": 2 to 3 sentences of clinical reasoning connecting visual_evidence
  to predicted_class. Must not contradict reasoning.

- "limitations": Factors limiting interpretation (image quality, rotation, artifacts,
  missing clinical context, etc.). If none, write "None identified."

- "warning": "Outil pédagogique uniquement — aucune valeur diagnostique réelle.
  Ne jamais utiliser ce résultat pour une décision médicale."
"""

# Balise <image> requise par les modèles Gemma 3 / MedGemma
FORMATTED_PROMPT = f"<image>\n{RAW_PROMPT}"


# ==========================================
# SYSTEME DE BARRE DE CHARGEMENT (STREAMER)
# ==========================================
class TokenProgressBarStreamer(BaseStreamer):
    """Un streamer personnalisé pour afficher la progression de la génération token par token."""
    def __init__(self, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        self.pbar = None
        self.is_prompt_skipped = False

    def put(self, value):
        # Le premier lot envoyé par transformers contient les tokens du prompt original.
        # On l'ignore pour ne commencer à compter qu'à partir de la génération réelle.
        if not self.is_prompt_skipped:
            self.is_prompt_skipped = True
            self.pbar = tqdm(
                total=self.max_new_tokens, 
                desc="[4/4] Analyse de l'image", 
                unit=" token",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            return
        
        # Incrémenter la barre à chaque nouveau token généré
        if self.pbar is not None:
            self.pbar.update(value.numel())

    def end(self):
        # Fermer proprement la barre de chargement une fois l'inférence terminée
        if self.pbar is not None:
            self.pbar.close()


def extract_json_from_text(text):
    """Extrait le JSON si le modèle ajoute des balises ou du texte autour."""
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            return None
    except json.JSONDecodeError:
        return None

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERREUR] L'image {IMAGE_PATH} est introuvable. Modifiez IMAGE_PATH dans le code.")
        return

    # ==========================================
    # DÉTECTION DU MATÉRIEL
    # ==========================================
    if torch.cuda.is_available():
        print("[INFO] GPU NVIDIA détecté ! Mode haute performance (CUDA) activé.")
        target_device = "cuda"
        model_dtype = torch.float16
    else:
        print("[ATTENTION] Aucun GPU NVIDIA détecté. Bascule sur le processeur (CPU).")
        print("[ATTENTION] L'inférence sera très lente sur CPU (plusieurs minutes par image).")
        target_device = "cpu"
        model_dtype = torch.float32

    print("\n[1/4] Chargement du processeur...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"[2/4] Chargement du modèle MedGemma 4B IT (en 4-bit pour économiser la mémoire)...")
    
    # Configuration pour charger le modèle en 4-bit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=model_dtype
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto", 
        quantization_config=quantization_config,
    )
    
    print("[3/4] Traitement de l'image (PNG) et du prompt multimodal...")
    
    # Load LoRA adapter if it exists
    adapter_path = os.path.join(BASE_DIR, "..", "..", "checkpoints", "medgemma-qlora", "final")
    if os.path.exists(adapter_path):
        from peft import PeftModel
        print(f"[*] Chargement des poids affinés (LoRA) depuis {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    image = Image.open(IMAGE_PATH).convert("RGB")
    
    # Création de la structure de conversation multimodale
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": RAW_PROMPT}
            ]
        }
    ]
    
    # Application du template de chat spécifique à Gemma
    formatted_prompt = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True
    )
    
    # Préparation des inputs
    inputs = processor(
        text=formatted_prompt, 
        images=image, 
        return_tensors="pt"
    ).to(model.device)

    print("[4/4] Initialisation du pipeline de génération...")
    max_tokens_to_generate = 512
    
    # Instanciation de notre barre de chargement
    progress_streamer = TokenProgressBarStreamer(max_new_tokens=max_tokens_to_generate)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens_to_generate,
            temperature=0.1,
            do_sample=True,
            streamer=progress_streamer # Liaison du streamer au processus de génération
        )

    # Décodage en ignorant le prompt de départ dans la sortie
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    raw_result = processor.decode(generated_tokens, skip_special_tokens=True)
    
    print("\n" + "="*60)
    print("RÉSULTAT BRUT GÉNÉRÉ :")
    print("-" * 60)
    print(raw_result)
    print("="*60)

    # Tentative d'extraction et de formatage du JSON
    parsed_json = extract_json_from_text(raw_result)
    
    if parsed_json:
        print("\n[SUCCÈS] JSON VALIDE EXTRAIT :")
        print(json.dumps(parsed_json, indent=4, ensure_ascii=False))
    else:
        print("\n[ATTENTION] Le modèle n'a pas produit un JSON valide.")

if __name__ == "__main__":
    main()