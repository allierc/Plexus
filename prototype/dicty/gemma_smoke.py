"""gemma_smoke.py -- offline smoke test for Gemma-3-12B-it on a rendered sim montage.
Run AFTER the weights are on the shared path. Confirms the VLM loads + captions an image.

  GEMMA_DIR=/groups/saalfeld/home/allierc/models/gemma-3-12b-it \
  PYTHONPATH=/workspace/Plexus/src:/workspace/Plexus/prototype \
  /workspace/.conda_envs/neural-graph-linux/bin/python gemma_smoke.py [image.png]
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor   # gemma-4 unified (encoder-free) loader, per model card

MODEL = os.environ.get("GEMMA_DIR", "/workspace/Plexus/VLLM/gemma-4-12B-it")
IMG = sys.argv[1] if len(sys.argv) > 1 else "good_vs_real.png"   # any rendered montage
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

assert os.path.isdir(MODEL), f"weights not found at {MODEL} -- download them on an internet node first"
print(f"[load] {MODEL} on {DEV} ...", flush=True)
model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype="bfloat16", device_map=DEV)
proc = AutoProcessor.from_pretrained(MODEL)

img = Image.open(IMG).convert("RGB")
PROMPT = ("This is a time-ordered montage of a cell/particle simulation (left=early, right=late). "
          "Describe the AGGREGATION MECHANISM and morphology in 3 sentences: is it a single dominant "
          "cluster, many small mounds, strands/filaments, or diffuse? Does it concentrate or stay spread "
          "over time? Name the most likely mechanism family (chemotactic aggregation, long-range "
          "attraction/coarsening, differential adhesion, excitable relay waves, external field, or soft-body fusion).")
# image BEFORE text (modality order recommended by the Gemma-4 card)
msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt",
                                  enable_thinking=False).to(model.device)   # captions, not chain-of-thought
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
resp = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
text = proc.parse_response(resp) if hasattr(proc, "parse_response") else resp   # strips control/thought tokens
print("\n=== Gemma-4 caption of", IMG, "===\n" + str(text).strip())
