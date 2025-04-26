# load_model.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch

# Move Huggingface cache to E: drive
os.environ['HF_HOME'] = "E:/huggingface_cache"
os.environ['HUGGINGFACE_HUB_CACHE'] = "E:/huggingface_cache"

def prepare_and_save_model(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    local_save_dir: str = "app/code/models/mistral-7B-instruct-v02",
) -> None:
    """Download model from Huggingface and save locally."""

    # Load Huggingface token
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file!")

    # Login
    login(token=token)

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True
    )

    # Download model
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Safe for CPU
        device_map="auto"           # Auto map to CPU
    )

    # Create save directory if not exist
    os.makedirs(local_save_dir, exist_ok=True)

    # Save model and tokenizer
    print(f"Saving tokenizer and model to {local_save_dir}...")
    tokenizer.save_pretrained(local_save_dir)
    model.save_pretrained(local_save_dir)
    print("Model and tokenizer saved locally!")

if __name__ == "__main__":
    prepare_and_save_model()
