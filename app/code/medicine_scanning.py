import os
import re
import base64
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
from google.cloud import vision
import torch
from dash import html
from fuzzywuzzy import process


HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

csv_path = os.path.join(DATA_DIR, "finalized_drugbank_data.csv")
drug_info_index_path = os.path.join(DATA_DIR, "drug_info_index_v3.faiss")
drug_name_embeddings_path = os.path.join(DATA_DIR, "drug_name_embeddings_v3.npy")
drug_name_index_path = os.path.join(DATA_DIR, "drug_name_index_v3.faiss")

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

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(DATA_DIR, "google-cloud-service-key.json")

print("[Scan] Initializing Google Vision API client...")
vision_client = vision.ImageAnnotatorClient()

print("[Scan] Precomputed drug name FAISS index ready.")

# # ---- Model Caching ----
# _cached_generator = None

# def load_generator_once():
#     global _cached_generator
#     if _cached_generator is None:
#         print("[Model] Loading Mistral 7B model for the first time...")
#         _cached_generator = hf_pipeline(
#             "text-generation",
#             model="mistralai/Mistral-7B-Instruct-v0.2",
#             device=0,
#             use_auth_token=HUGGINGFACE_TOKEN
#         )
#         print("[Model] Mistral 7B model loaded and cached.")
#     else:
#         print("[Model] Reusing cached Mistral 7B model.")
#     return _cached_generator

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

def better_clean_text(ocr_text):
    # 1. Only keep alphanumeric and space
    clean_text = ''.join(e if (e.isalnum() or e.isspace()) else ' ' for e in ocr_text)
    
    # 2. Lowercase everything
    clean_text = clean_text.lower()

    # 3. Split into lines and words
    words = clean_text.split()

    # 4. Remove pure numbers and short garbage
    words = [w for w in words if not w.isdigit() and len(w) > 2]

    # 5. Remove duplicates but preserve order
    seen = set()
    filtered_words = []
    for word in words:
        if word not in seen:
            seen.add(word)
            filtered_words.append(word)

    # 6. Rebuild clean text
    final_text = ' '.join(filtered_words)

    return final_text

def extract_all_candidate_drug_names(ocr_text):
    lines = ocr_text.split('\n')

    candidates = []
    for line in lines:
        line = line.strip()
        if re.match(r'^[A-Za-z\s\-]+$', line) and 2 <= len(line) <= 30:
            candidates.append(line.lower())

    return candidates  # Return list of candidates

bad_keywords = ["batch", "before", "expiry", "best", "no", "lot"]
def is_bad_candidate(candidate):
    for bad_word in bad_keywords:
        if bad_word in candidate.lower():
            return True
    return False

trusted_drugs = [
    "calpol", "amoxicillin", "simvastatin", "omeprazole", "losartan",
    "amlodipine", "metformin", "salbutamol", "cetirizine", "ibuprofen", "air-x"
]

fruit_keywords = ["lemon", "orange", "banana", "grape", "cherry", "apple", "mango", "berry", "peach"]
def contains_fruit_word(text):
    for fruit in fruit_keywords:
        if fruit in text.lower():
            return True
    return False

def find_best_drug(ocr_text):
    print("[Match] Finding best matching drug name (FAISS search)...")

    # clean_text = ''.join(e for e in ocr_text if e.isalnum() or e.isspace())
    # print("[Check OCR] clean_text: ", clean_text)
    # if not clean_text.strip():
    #     print("[Match] OCR text is empty after cleaning.")
    #     return None, 0.0

    candidates = extract_all_candidate_drug_names(ocr_text)

    if not candidates:
        print("[Error] No candidate drug names found in OCR text.")
        return None, 0.0

    best_match = None
    best_confidence = 0.0

    for candidate in candidates:
        if is_bad_candidate(candidate):
            print(f"[Skip] Ignoring bad candidate: {candidate}")
            continue
        clean_candidate = better_clean_text(candidate)
        candidate_words = clean_candidate.lower().split()

        # --- NEW: Direct match check ---
        for trusted_drug in trusted_drugs:
            if trusted_drug in candidate_words:
                print(f"[Direct Match] {candidate} matched trusted drug: {trusted_drug}")
                return trusted_drug, 100.0  # Bypass FAISS, 100% confidence

        embedding = embedder.encode([clean_candidate], normalize_embeddings=True)
        D, I = drug_name_index.search(np.array(embedding), k=1)

        idx = I[0][0]
        raw_similarity = D[0][0]
        similarity = max(min(raw_similarity, 1.0), 0.0)
        confidence = similarity * 100

        match_name = drug_names[idx]

        # --- NEW: Fruit penalty ---
        if contains_fruit_word(clean_candidate):
            print(f"[Penalty] Candidate contains fruit word: {candidate}. Reducing confidence.")
            confidence *= 0.7  # 30% penalty

        print(f"[Candidate] {candidate} -> {match_name} ({confidence:.2f}%)")

        if confidence > best_confidence:
            best_confidence = confidence
            best_match = match_name

    return best_match, best_confidence

# def find_best_drug(ocr_text):
#     print("[Match] Finding best matching drug name (FAISS search)...")
    
#     # --- 1. Clean OCR text ---
#     clean_text = ''.join(e for e in ocr_text if e.isalnum() or e.isspace())
#     print("[Check OCR] clean_text:", clean_text)
    
#     if not clean_text.strip():
#         print("[Match] OCR text is empty after cleaning.")
#         return None, None  # Important: return None for both match and distance

#     # --- 2. Embed OCR text ---
#     ocr_embedding = embedder.encode([clean_text])
#     ocr_embedding = np.array(ocr_embedding)
#     print("[OCR Embedding] embed result:", ocr_embedding)

#     # --- 3. Search FAISS index (k=1) ---
#     D, I = drug_name_index.search(ocr_embedding, k=1)
#     best_idx = I[0][0]
#     best_distance = D[0][0]
#     best_match = drug_names[best_idx]

#     print(f"[Match] FAISS best match found: {best_match} (distance: {best_distance})")

#     # --- 4. Add distance checking ---
#     return best_match, best_distance

def retrieve_drug_info(drug_name):
    print(f"[Retrieve] Retrieving drug info for: {drug_name}")
    query_emb = embedder.encode([drug_name])
    # FAISS cosine similarity search
    D, I = drug_info_index.search(np.array(query_emb), k=1)
    print("[Retrieve] Drug info retrieved.")
    return df.iloc[I[0][0]].to_dict()

# def build_prompt(info):
#     return f"""
# You are a medical assistant. Summarize the following drug information.

# Name: {info.get('name', 'N/A')}
# Description: {info.get('description', 'N/A')}
# Indication: {info.get('indication', 'N/A')}
# Mechanism of Action: {info.get('mechanism_of_action', 'N/A')}
# Toxicity: {info.get('toxicity', 'N/A')}

# Summarize this for a general audience.
# """

def generate_summary(info, model, tokenizer):
    print("[Model] Generating summary using Mistral 7B...")

    # Adjust the prompt to focus on structured Q&A format
    prompt = f"""
    You are a medical assistant providing clear and simple explanations for patients.

    The following drug information is provided:
    - Name: {info.get('name', 'N/A')}
    - Description: {info.get('description', 'N/A')}
    - Indication: {info.get('indication', 'N/A')}
    - Mechanism of Action: {info.get('mechanism_of_action', 'N/A')}
    - Toxicity: {info.get('toxicity', 'N/A')}

    Please provide a summary with answers to the following questions:
    - What is this medicine for? (General use)
    - How is it used? (Dosage and instructions)
    - What are the common side effects?
    - Are there any important precautions or warnings for patients?

    Please format the output like:
    - What is this medicine for?
    - Answer: [Answer here]
    
    - How is it used?
    - Answer: [Answer here]
    
    - What are the common side effects?
    - Answer: [Answer here]
    
    - Are there any important precautions or warnings for patients?
    - Answer: [Answer here]

    Please avoid complex terms, and keep it short and simple.
    """

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **input_ids,
            max_new_tokens=250,
            temperature=0.95,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output[0][input_ids["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print("[Model] Summary generation complete.")
    return output_text.strip()


def parse_summary(summary_text):
    """Parse the AI-generated summary into structured Q&A format for patient clarity."""
    print("[Postprocess] Parsing the summary into Q&A format...")

    # Initialize a dictionary to store the parsed data
    qna = {
        "use": "",
        "dosage": "",
        "side_effects": "",
        "precautions": ""
    }

    # Split the summary by lines
    lines = summary_text.split("\n")
    
    # Prepare placeholders for questions and answers
    current_question = None
    current_answer = []

    # Define the phrases to check for
    phrase_to_check = {
        "use": "what is this medicine for?",
        "dosage": "how is it used?",
        "side_effects": "what are the common side effects?",
        "precautions": "are there any important precautions or warnings for patients?"
    }

    for line in lines:
        line = line.strip()

        # Check for question phrases (exact matching)
        for question, phrase in phrase_to_check.items():
            if phrase.lower() in line.lower():  # Check if the phrase exists in the line
                if current_question and current_answer:
                    qna[current_question] = " ".join(current_answer).strip()
                current_question = question
                current_answer = []  # Reset for the new question
                break  # Stop checking after the first match

        # If line contains the answer, extract the text after ": "
        if '- Answer:' in line:
            # Split by ': ' and get the part after it, strip any extra whitespace
            answer = line.split(': ', 1)[1].strip()
            current_answer.append(answer)

    # Ensure the last question's answer is also added
    if current_question and current_answer:
        qna[current_question] = " ".join(current_answer).strip()

    # Return the structured data for Q&A, formatted into HTML list items
    # return [
    #     html.Li(html.B("Use: "), qna['use']),
    #     html.Li(html.B("Dosage: "), qna['dosage']),
    #     html.Li(html.B("Common Side Effects: "), qna['side_effects']),
    #     html.Li(html.B("Precautions: "), qna['precautions'])
    # ]
    return qna


def split_into_bullets(summary_text):
    """Split the model summary text into bullet points."""
    print("[Postprocess] Splitting model summary into bullets...")
    lines = summary_text.split('\n')
    bullets = []

    for line in lines:
        line = line.strip()
        if line and len(line) > 5:
            bullets.append(html.Li(line))

    return bullets

def fallback_bullets(info):
    """Fallback to building bullet points manually from info fields."""
    print("[Postprocess] Fallback: Building bullets manually...")
    bullets = []
    if info.get("description"):
        bullets.append(html.Li(f"Description: {info['description']}"))
    if info.get("indication"):
        bullets.append(html.Li(f"Indication: {info['indication']}"))
    if info.get("mechanism_of_action"):
        bullets.append(html.Li(f"Mechanism: {info['mechanism_of_action']}"))
    if info.get("toxicity"):
        bullets.append(html.Li(f"Toxicity: {info['toxicity']}"))
    return bullets

# ---- Main Scan Function ----

def scan_medicine(contents, model, tokenizer):
    print("[Scan] Starting full medicine scan process...")
    try:
        image_bytes = decode_base64_image(contents)
        ocr_text = ocr_google_image(image_bytes)
        print("[OCR] image scan result: ", ocr_text)

        # --- New check: OCR text should not be empty or garbage ---
        if not ocr_text or len(ocr_text.strip()) < 5:
            print("[Scan] OCR text too short, likely not a medicine package.")
            return None, None, "No readable text found. Please upload a clear medicine package image."

        # --- Drug matching ---
        drug_name, match_percentage = find_best_drug(ocr_text)

        if drug_name is None:
            return None, None, "This image does not appear to be a medicine package. Please insert a medicine package."

        # --- Check if match percentage is acceptable ---
        if match_percentage < 70:
            print("⚠ Warning: Low confidence match. Please check OCR image.")

        # # --- Check if distance is acceptable ---
        # if distance > 0.6:
        #     print(f"[Scan] Match distance too high ({distance}). Rejecting image.")
        #     return None, None, "This image does not appear to be a medicine package. Please insert a medicine package."

       # --- Otherwise, continue normal flow ---
        drug_info = retrieve_drug_info(drug_name)
        print(f"DRUG INFO: ", drug_info)
        summary_text = generate_summary(drug_info, model, tokenizer)
        print(f"Generated Summary Text: {summary_text}")

        if summary_text:
            bullets = parse_summary(summary_text)  # This returns the list of HTML Li elements
            print('[Result] Check bullets: ', bullets)
        else:
            bullets = fallback_bullets(drug_info)
            print('[Result] Fallback Check bullets: ', bullets)

        print("[Scan] Full scan process completed successfully.")
        return drug_name, bullets, None  # No error
    except Exception as e:
        print(f"[Scan] Error during scanning: {str(e)}")
        return None, None, f"Error: {str(e)}"
