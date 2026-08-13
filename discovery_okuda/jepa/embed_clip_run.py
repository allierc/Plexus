"""Embed every Okuda run's strip.png with a FROZEN openai/clip-vit-base-patch32.

TILING DECISION (fixed BEFORE any sanity number was computed):
The strip is not an arbitrary wide photo, it is an exact 4x8 montage of 440x450 panels
(8 timepoints across; rows = mesh view A, mesh view B, chemical field on the surface,
equatorial cross-section). We cut it on that true panel grid, so every panel enters CLIP
at its native ~1:1 aspect through the standard CLIP preprocessing, with no squashing and
no rescaling of the object relative to its panel (so apparent size, which is morphology
here, survives). We then mean-pool the 8 timepoints WITHIN each row, L2-normalise each
row mean, and concatenate the 4 rows -> 2048-d, so that a shape descriptor is never
averaged together with a chemistry descriptor. A plain 512-d mean over all 32 panels is
also saved for comparison.

Writes /workspace/Plexus/discovery_okuda/jepa/embed_clip.npz
"""
import os, sys, json, time
os.environ.setdefault("HF_HOME", "/workspace/Plexus/discovery_okuda/jepa/hf")

import numpy as np
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPModel, CLIPImageProcessor

Image.MAX_IMAGE_PIXELS = None

JEPA = "/workspace/Plexus/discovery_okuda/jepa"
MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda:0"
NROW, NCOL = 4, 8
ROW_NAMES = ["mesh_view_a", "mesh_view_b", "chem_field", "cross_section"]

d = np.load(os.path.join(JEPA, "design.npz"), allow_pickle=True)
run_names = np.array([str(x) for x in d["run_names"]])
run_dirs = np.array([str(x) for x in d["run_dirs"]])
has_strip = d["has_strip"].astype(bool)

idx = np.where(has_strip)[0]
print(f"[info] {len(idx)} of {len(run_names)} runs flagged has_strip", flush=True)

model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad_(False)
proc = CLIPImageProcessor.from_pretrained(MODEL_ID)
DIM = model.config.projection_dim
print(f"[info] {MODEL_ID} loaded frozen on {DEVICE}; projection dim = {DIM}", flush=True)


def cut_panels(path):
    """Return list of 32 PIL RGB panels in row-major order, or None if the grid does not fit."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        W, H = im.size
        if W % NCOL or H % NROW:
            return None, (W, H)
        pw, ph = W // NCOL, H // NROW
        panels = [
            im.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph))
            for r in range(NROW)
            for c in range(NCOL)
        ]
    return panels, (pw, ph)


E_row = np.zeros((len(run_names), NROW * DIM), dtype=np.float32)
E_mean = np.zeros((len(run_names), DIM), dtype=np.float32)
E_panels = np.zeros((len(run_names), NROW * NCOL, DIM), dtype=np.float32)
ok = np.zeros(len(run_names), dtype=bool)
panel_size = None
bad = []

t0 = time.time()
pool = ThreadPoolExecutor(max_workers=6)


def load(i):
    return i, cut_panels(os.path.join(run_dirs[i], "strip.png"))


with torch.no_grad():
    for k, (i, (panels, psz)) in enumerate(pool.map(load, idx)):
        if panels is None:
            bad.append((run_names[i], psz))
            continue
        panel_size = psz
        px = proc(images=panels, return_tensors="pt")["pixel_values"].to(DEVICE)
        f = model.get_image_features(pixel_values=px)              # (32, DIM)
        f = torch.nn.functional.normalize(f.float(), dim=-1)       # per-panel unit
        E_panels[i] = f.cpu().numpy()
        rows = f.view(NROW, NCOL, DIM).mean(dim=1)                 # (4, DIM) time-pool
        rows = torch.nn.functional.normalize(rows, dim=-1)
        v = rows.reshape(-1)
        E_row[i] = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        m = torch.nn.functional.normalize(f.mean(dim=0), dim=-1)
        E_mean[i] = m.cpu().numpy()
        ok[i] = True
        if (k + 1) % 50 == 0:
            print(f"[info] {k+1}/{len(idx)}  {time.time()-t0:.0f}s", flush=True)

print(f"[info] embedded {ok.sum()} runs in {time.time()-t0:.0f}s; panel size {panel_size}", flush=True)
if bad:
    print("[warn] grid did not fit for:", bad, flush=True)

# ---------------------------------------------------------------- sanity check
name2i = {n: i for i, n in enumerate(run_names)}


def cos(E, a, b):
    return float(E[name2i[a]] @ E[name2i[b]])


rng = np.random.default_rng(0)
sanity = {}
for tag, E in [("row_concat_2048", E_row), ("mean_pool_512", E_mean)]:
    sub = np.where(ok)[0]
    pairs = []
    while len(pairs) < 200:
        a, b = rng.choice(sub, 2, replace=False)
        pairs.append(float(E[a] @ E[b]))
    pairs = np.array(pairs)
    s = {
        "cos_b_star__b_star_sharp": cos(E, "b_star", "b_star_sharp"),
        "cos_b_star__b_null_plain": cos(E, "b_star", "b_null_plain"),
        "cos_b_flower__b_flower_death": cos(E, "b_flower", "b_flower_death"),
        "random_pairs_n": 200,
        "random_pairs_mean": float(pairs.mean()),
        "random_pairs_std": float(pairs.std()),
        "random_pairs_min": float(pairs.min()),
        "random_pairs_max": float(pairs.max()),
    }
    sanity[tag] = s
    print(f"\n=== sanity: {tag} ===", flush=True)
    for k2, v2 in s.items():
        print(f"  {k2}: {v2:.4f}" if isinstance(v2, float) else f"  {k2}: {v2}", flush=True)

# per-row breakdown of the star/sphere contrast, on the primary embedding
print("\n=== per-row cosine (primary row_concat, each row L2-normed separately) ===", flush=True)
per_row = {}
Rn = E_panels / np.maximum(np.linalg.norm(E_panels, axis=-1, keepdims=True), 1e-12)
for r, rname in enumerate(ROW_NAMES):
    Er = Rn[:, r * NCOL:(r + 1) * NCOL, :].mean(axis=1)
    Er = Er / np.maximum(np.linalg.norm(Er, axis=-1, keepdims=True), 1e-12)
    sub = np.where(ok)[0]
    rp = np.array([float(Er[a] @ Er[b]) for a, b in rng.choice(sub, (200, 2))])
    per_row[rname] = {
        "cos_b_star__b_star_sharp": cos(Er, "b_star", "b_star_sharp"),
        "cos_b_star__b_null_plain": cos(Er, "b_star", "b_null_plain"),
        "cos_b_flower__b_flower_death": cos(Er, "b_flower", "b_flower_death"),
        "random_pairs_mean": float(rp.mean()),
        "random_pairs_std": float(rp.std()),
    }
    print(f"  {rname}: " + "  ".join(f"{k2}={v2:.4f}" for k2, v2 in per_row[rname].items()), flush=True)

TILING = (
    "Cut the strip on its true 4x8 montage grid into 32 native-aspect 440x450 panels "
    "(rows = mesh view A, mesh view B, chemical field, equatorial section; columns = 8 timepoints), "
    "embed each panel through standard CLIP preprocessing, mean-pool the 8 timepoints within each "
    "row, L2-normalise each row mean and concatenate the 4 rows, so no panel is ever squashed and a "
    "shape descriptor is never averaged with a chemistry descriptor."
)

np.savez_compressed(
    os.path.join(JEPA, "embed_clip.npz"),
    E=E_row,
    E_row=E_row,
    E_mean=E_mean,
    E_panels=E_panels,
    embedded=ok,
    run_names=run_names,
    run_dirs=run_dirs,
    model_id=MODEL_ID,
    device=DEVICE,
    dim=DIM,
    primary="E_row",
    grid=np.array([NROW, NCOL]),
    row_names=np.array(ROW_NAMES),
    panel_size=np.array(panel_size),
    tiling_decision=TILING,
    sanity_json=json.dumps(sanity),
    per_row_json=json.dumps(per_row),
)
print("\n[info] wrote", os.path.join(JEPA, "embed_clip.npz"), flush=True)
