import os
import sys
from datasets import load_dataset

# Specify the dataset repo and local save path
dataset_name = "lmms-lab/VQAv2"
save_dir = os.path.join(".", "datasets", "VQAv2_local")

# Use Hugging Face mirror in PRC if HF is not accessible
def set_hf_mirror():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("Set Hugging Face mirror: https://hf-mirror.com")

def test_hf_connection():
    import requests
    try:
        r = requests.get("https://huggingface.co", timeout=5)
        if r.status_code == 200:
            print("Hugging Face is accessible.")
            return True
    except Exception:
        pass
    print("Hugging Face is NOT accessible, switching to mirror.")
    return False

def is_dataset_downloaded(path):
    # Check for dataset metadata file
    return os.path.exists(os.path.join(path, "dataset_info.json"))

if os.path.exists(save_dir) and is_dataset_downloaded(save_dir):
    print(f"The dataset already exists and appears complete at {save_dir}. Skipping download.")
    sys.exit(0)
else:
    os.makedirs(save_dir, exist_ok=True)

# Try to use HF, fallback to mirror if needed
try:
    import requests
    if not test_hf_connection():
        set_hf_mirror()
except ImportError:
    print("requests not installed, skipping HF connectivity test.")

# Download the dataset and save to local directory
dataset = load_dataset(dataset_name, split="validation")
dataset.save_to_disk(save_dir)

print(f"{dataset_name} validation split downloaded and saved to {save_dir}")