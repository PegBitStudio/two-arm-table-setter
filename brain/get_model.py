"""Download the AI model (3.1 GB, once): Intel's OpenVINO INT4 build of Qwen3-VL-4B.

    python brain/get_model.py
"""
from huggingface_hub import snapshot_download

from language import MODEL_DIR

if __name__ == "__main__":
    snapshot_download("OpenVINO/Qwen3-VL-4B-Instruct-int4-ov", local_dir=MODEL_DIR)
    print(f"model ready in {MODEL_DIR}")
