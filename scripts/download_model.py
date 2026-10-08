from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
snapshot_download("facebook/dinov2-base",revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",cache_dir=ROOT/"artifacts/hf",allow_patterns=["config.json","preprocessor_config.json","model.safetensors"])
