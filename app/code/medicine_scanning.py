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


# Path setup
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

csv_path = os.path.join(DATA_DIR, "drugbank_clean.csv")
index_path = os.path.join(DATA_DIR, "drug_index.faiss")

# Load Dataset
print("🔹 Loading DrugBank data...")
df = pd.read_csv(csv_path)

# Load FAISS Index
print("🔹 Loading FAISS index...")
index = faiss.read_index(index_path)

# Load Embedding Model
print("🔹 Loading SentenceTransformer...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load LLM Model
print("🔹 Loading Mistral 7B Model...")
generator = hf_pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device=0  # Move to GPU if available
)

# Load Google Vision API Client
print("🔹 Initializing Google Vision Client...")
vision_client = vision.ImageAnnotatorClient()

print("Medicine scanning module ready.")

# ---- Helper Functions ----

def decode_base64_image(content):
    """Decode base64-encoded uploaded image."""
    header, encoded = content.split(",", 1)
    binary_data = base64.b64decode(encoded)
    return binary_data

def ocr_google_image(image_bytes):
    """Run OCR using Google Vision API."""
    image = vision.Image(content=image_bytes)
    response = vision_client.text_detection(image=image)
    return response.text_annotations[0].description if response.text_annotations else ""

def find_best_drug(ocr_text, drug_names):
    """Find best matching drug name from OCR text."""
    words = ocr_text.split()
    candidates = []

    for word in words:
        match, score = process.extractOne(word, drug_names)
        if score > 85:
            candidates.append((match, score))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)  # highest score first
        return candidates[0][0]
    else:
        return None

def retrieve_drug_info(drug_name):
    """Retrieve drug details from FAISS index."""
    query_emb = embedder.encode([drug_name])
    D, I = index.search(np.array(query_emb), k=1)
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
    """Generate a human-readable summary using the LLM."""
    prompt = build_prompt(info)
    output = generator(prompt, max_new_tokens=250)[0]["generated_text"]
    return output

# ---- High-Level Function ----

def scan_medicine(contents):
    """
    Full scan pipeline:
    1. Decode image
    2. OCR extract text
    3. Match to drug names
    4. Retrieve drug info
    5. Summarize with LLM
    """
    try:
        image_bytes = decode_base64_image(contents)
        ocr_text = ocr_google_image(image_bytes)

        drug_name = find_best_drug(ocr_text, df["name"].tolist())

        if drug_name:
            drug_info = retrieve_drug_info(drug_name)
            summary_text = generate_summary(drug_info)
            return drug_name, summary_text, None
        else:
            return None, None, "Could not detect a matching medicine name."
    except Exception as e:
        return None, None, f"Error: {str(e)}"
