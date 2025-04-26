import os
import base64
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
from google.cloud import vision
import torch


HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

csv_path = os.path.join(DATA_DIR, "drugbank_clean.csv")
drug_info_index_path = os.path.join(DATA_DIR, "drug_index.faiss")
drug_name_embeddings_path = os.path.join(DATA_DIR, "drug_name_embeddings.npy")
drug_name_index_path = os.path.join(DATA_DIR, "drug_name_index.faiss")

# ---- Load Light Resources at Startup ----

print("[Scan] Loading DrugBank dataset...")
df = pd.read_csv(csv_path)
drug_names = df["name"].fillna("").tolist()

print("[Scan] Loading drug info FAISS index...")
drug_info_index = faiss.read_index(drug_info_index_path)

print("[Scan] Loading precomputed drug name embeddings...")
drug_name_embeddings = np.load(drug_name_embeddings_path)

print("[Scan] Loading FAISS index for drug names...")
drug_name_index = faiss.read_index(drug_name_index_path)

print("[Scan] Loading SentenceTransformer (MiniLM)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', token=HUGGINGFACE_TOKEN)

print("[Scan] Initializing Google Vision API client...")
vision_client = vision.ImageAnnotatorClient()

print("[Scan] Precomputed drug name FAISS index ready.")

# ---- Model Caching ----
_cached_generator = None

def load_generator_once():
    global _cached_generator
    if _cached_generator is None:
        print("[Model] Loading Mistral 7B model for the first time...")
        _cached_generator = hf_pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            device=0,
            use_auth_token=HUGGINGFACE_TOKEN
        )
        print("[Model] Mistral 7B model loaded and cached.")
    else:
        print("[Model] Reusing cached Mistral 7B model.")
    return _cached_generator

# ---- Helper Functions ----

def decode_base64_image(content):
    print("[Scan] Decoding uploaded base64 image...")
    header, encoded = content.split(",", 1)
    binary_data = base64.b64decode(encoded)
    return binary_data

def ocr_google_image(image_bytes):
    print("[OCR] Running OCR using Google Vision API...")
    image = vision.Image(content=image_bytes)
    response = vision_client.text_detection(image=image)
    if response.text_annotations:
        print("[OCR] OCR text detected.")
    else:
        print("[OCR] No text detected.")
    return response.text_annotations[0].description if response.text_annotations else ""

def find_best_drug(ocr_text):
    print("[Match] Finding best matching drug name (FAISS search)...")
    
    clean_text = ''.join(e for e in ocr_text if e.isalnum() or e.isspace())
    if not clean_text.strip():
        print("[Match] OCR text is empty after cleaning.")
        return None

    ocr_embedding = embedder.encode([clean_text])
    D, I = drug_name_index.search(np.array(ocr_embedding), k=1)
    best_idx = I[0][0]
    best_match = drug_names[best_idx]

    print(f"[Match] FAISS best match found: {best_match}")
    return best_match

def retrieve_drug_info(drug_name):
    print(f"[Retrieve] Retrieving drug info for: {drug_name}")
    query_emb = embedder.encode([drug_name])
    D, I = drug_info_index.search(np.array(query_emb), k=1)
    print("[Retrieve] Drug info retrieved.")
    return df.iloc[I[0][0]].to_dict()

def build_prompt(info):
    return f"""
You are a medical assistant. Summarize the following drug information.

Name: {info.get('name', 'N/A')}
Description: {info.get('description', 'N/A')}
Indication: {info.get('indication', 'N/A')}
Mechanism of Action: {info.get('mechanism_of_action', 'N/A')}
Toxicity: {info.get('toxicity', 'N/A')}

Summarize this for a general audience.
"""

def generate_summary(info, model, tokenizer):
    print("[Model] Generating summary using Mistral 7B...")
    prompt = build_prompt(info)
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **input_ids,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print("[Model] Summary generation complete.")

# ---- Main Scan Function ----

def scan_medicine(contents, model, tokenizer):
    print("[Scan] Starting full medicine scan process...")
    try:
        image_bytes = decode_base64_image(contents)
        ocr_text = ocr_google_image(image_bytes)

        drug_name = find_best_drug(ocr_text)
        print(f"[Match] Final matched drug: {drug_name}")

        if drug_name:
            drug_info = retrieve_drug_info(drug_name)
            summary_text = generate_summary(drug_info, model, tokenizer)
            print("[Scan] Full scan process completed successfully.")
            return drug_name, summary_text, None
        else:
            print("[Scan] No drug name matched.")
            return None, None, "Could not detect a matching medicine name."
    except Exception as e:
        print(f"[Scan] Error during scanning: {str(e)}")
        return None, None, f"Error: {str(e)}"
