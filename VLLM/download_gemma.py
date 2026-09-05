"""Download google/gemma-4-12B-it beside this checkout, into <repo>/VLLM (Xet-backed, ~24 GB).

Token via env: HF_TOKEN. Run:  HF_TOKEN=... python download_gemma.py
`GEMMA_DIR` overrides the destination, and it is the same variable `describe_video.py` reads --
so the place this writes to is the place that one loads from.
"""
import os
from huggingface_hub import snapshot_download

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.environ.get("GEMMA_DIR", os.path.join(_REPO, "VLLM", "gemma-4-12B-it"))
print(f"[dl] google/gemma-4-12B-it -> {DEST}", flush=True)
path = snapshot_download(
    "google/gemma-4-12B-it",
    local_dir=DEST,
    token=os.environ.get("HF_TOKEN"),
    ignore_patterns=["*.gguf"],
    max_workers=8,
)
print("[dl] DONE ->", path, flush=True)
