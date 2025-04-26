import os
import base64
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
from google.cloud import vision


# Huggingface Token from Environment
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# Path setup
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

csv_path = os.path.join(DATA_DIR, "drugbank_clean.csv")
index_path = os.path.join(DATA_DIR, "drug_index.faiss")

# ---- Light Resources Loaded at Startup ----
# Load Dataset
print("Loading DrugBank data...")
df = pd.read_csv(csv_path)

# Load FAISS Index
print("Loading FAISS index...")
index = faiss.read_index(index_path)

# Load Embedding Model (SentenceTransformer)
print("Loading SentenceTransformer (MiniLM-L6-v2)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', token=HUGGINGFACE_TOKEN)

# Load Google Vision API Client
print("Initializing Google Vision Client...")
vision_client = vision.ImageAnnotatorClient()

print("Medicine scanning backend ready (model lazy loading).")

# ---- Helper Functions ----

def decode_base64_image(content):
    """Decode base64-encoded uploaded image."""
    print("[Scan] Decoding uploaded base64 image...")
    header, encoded = content.split(",", 1)
    binary_data = base64.b64decode(encoded)
    return binary_data

def ocr_google_image(image_bytes):
    """Run OCR using Google Vision API."""
    print("[Scan] Running OCR with Google Vision API...")
    image = vision.Image(content=image_bytes)
    response = vision_client.text_detection(image=image)
    if response.text_annotations:
        print("[OCR] Text detected successfully.")
    else:
        print("[OCR] No text detected.")
    return response.text_annotations[0].description if response.text_annotations else ""

def find_best_drug(ocr_text, drug_names):
    print("[Scan] Searching best matching drug name (optimized)...")
    # Clean OCR text
    clean_text = ''.join(e for e in ocr_text if e.isalnum() or e.isspace())
    words = clean_text.split()

    candidates = []
    if not words:
        print("[Scan] OCR produced empty or bad text after cleaning.")
        return None

    # Try to match full cleaned OCR text to known drug names first
    match, score = process.extractOne(' '.join(words), drug_names)
    if score > 85:
        print(f"[Match] Full OCR best match found: {match} (Score: {score})")
        return match

    # Fall back: Try matching individual words if needed
    for word in words:
        match, score = process.extractOne(word, drug_names)
        if score > 85:
            candidates.append((match, score))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_match = candidates[0][0]
        print(f"[Match] Fallback best word match found: {best_match}")
        return best_match
    else:
        print("[Match] No good drug match found.")
        return None

def retrieve_drug_info(drug_name):
    """Retrieve drug details from FAISS index."""
    print(f"[Scan] Retrieving drug info for: {drug_name}")
    query_emb = embedder.encode([drug_name])
    D, I = index.search(np.array(query_emb), k=1)
    print("[Retrieve] Drug info retrieved.")
    return df.iloc[I[0][0]].to_dict()

def build_prompt(info):
    """Build prompt for the LLM summarization."""
    return f"""
You are a medical assistant. Summarize the following drug information.

Name: {info.get('name', 'N/A')}
Description: {info.get('description', 'N/A')}
Indication: {info.get('indication', 'N/A')}
Mechanism of Action: {info.get('mechanism_of_action', 'N/A')}
Toxicity: {info.get('toxicity', 'N/A')}

Summarize this for a general audience.
"""

def generate_summary(info):
    """Load the Hugging Face model lazily, with Huggingface Token."""
    print("[Scan] Loading Mistral 7B model from HuggingFace...")
    generator = hf_pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        device=0,
        use_auth_token=HUGGINGFACE_TOKEN
    )
    print("[Model] Mistral 7B loaded. Starting summary generation...")
    prompt = build_prompt(info)
    output = generator(prompt, max_new_tokens=250)[0]["generated_text"]
    print("[Summary] Summary generation complete.")
    return output

# ---- Main High-Level Scan Function ----

def scan_medicine(contents):
    """
    Full scan pipeline:
    1. Decode image
    2. OCR extract text
    3. Match to drug names
    4. Retrieve drug info
    5. Summarize with LLM
    """
    print("[Scan] Starting full medicine scan process...")
    try:
        image_bytes = decode_base64_image(contents)

        ocr_text = ocr_google_image(image_bytes)

        drug_name = find_best_drug(ocr_text, df["name"].tolist())

        if drug_name:
            drug_info = retrieve_drug_info(drug_name)
            summary_text = generate_summary(drug_info)
            print("[Scan] Full scan process completed successfully.")
            return drug_name, summary_text, None
        else:
            print("[Scan] No drug name matched.")
            return None, None, "Could not detect a matching medicine name."
    except Exception as e:
        print(f"[Scan] Error during scanning: {str(e)}")
        return None, None, f"Error: {str(e)}"
