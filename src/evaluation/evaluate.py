import os
import json
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "google/medgemma-4b-it"
ADAPTER_PATH = os.path.join(BASE_DIR, "..", "..", "checkpoints", "medgemma-qlora", "final")
TEST_LIST = os.path.join(BASE_DIR, "..", "..", "data", "raw", "metadata", "test_list.txt")
CSV_FILE = os.path.join(BASE_DIR, "..", "..", "data", "raw", "metadata", "Data_Entry_2017.csv")

def main():
    if not os.path.exists(ADAPTER_PATH):
        print(f"Error: No trained adapter found at {ADAPTER_PATH}. Train the model first.")
        return
        
    print("Loading Base Model & Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    print("Loading Trained LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print("Evaluation script ready! Next steps:")
    print("1. Implement inference loop on the first 100 images from test_list.txt")
    print("2. Parse JSON output.")
    print("3. Compare predicted_class with Data_Entry_2017 labels to compute Sensitivity/Specificity.")
    
if __name__ == "__main__":
    main()
