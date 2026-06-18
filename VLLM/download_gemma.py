"""Download google/gemma-4-12B-it into /workspace/Plexus/VLLM (Xet-backed, ~24 GB).
Token via env: HF_TOKEN. Run:  HF_TOKEN=... python download_gemma.py
"""
import os
from huggingface_hub import snapshot_download

DEST = "/workspace/Plexus/VLLM/gemma-4-12B-it"
print(f"[dl] google/gemma-4-12B-it -> {DEST}", flush=True)
path = snapshot_download(
    "google/gemma-4-12B-it",
    local_dir=DEST,
    token=os.environ.get("HF_TOKEN"),
    ignore_patterns=["*.gguf"],
    max_workers=8,
)
print("[dl] DONE ->", path, flush=True)
