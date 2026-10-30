"""Download the pinned original model to the workspace cache, never modify it."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "work" / "hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from huggingface_hub import snapshot_download

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
if __name__ == "__main__":
    path = snapshot_download(
        MODEL_ID, revision=REVISION,
        allow_patterns=["*.json", "model.safetensors", "README.md"],
        max_workers=4,
    )
    (ROOT / "work" / "base_model_path.txt").write_text(path, encoding="utf-8")
    print("Pinned original model:", path, flush=True)
