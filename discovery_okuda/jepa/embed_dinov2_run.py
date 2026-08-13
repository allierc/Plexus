"""Embed every Okuda run's strip.png with a FROZEN facebook/dinov2-large.

TILING DECISION
---------------
strip.png is not a photo, it is a regular montage: 3520x1800 = 8 columns
(8 time points) x 4 rows (4 render channels: shaded surface, second view,
chemical-field colouring, mid-plane section outline), i.e. 32 panels of
440x450 px each.  So we cut the image on ITS OWN GRID instead of squashing
the whole 1.96:1 canvas into 224x224.  Each 440x450 panel is already
near-square, so it is resized to 224x224 with a 2% aspect change and NO
crop (a centre-crop would eat the scale bar and part of the object).
We mean-pool the 8 time panels WITHIN each row, L2-normalise each row
block, and concatenate the 4 rows -> 4096-d.  Rationale: the 4 rows are
different physical readouts; averaging them together (a flat 32-panel
mean) would let a chemical-pattern change cancel a shape change.

Outputs /workspace/Plexus/discovery_okuda/jepa/embed_dinov2.npz
"""
import os, json, time
os.environ.setdefault("HF_HOME", "/workspace/Plexus/discovery_okuda/jepa/hf")

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel
from concurrent.futures import ThreadPoolExecutor

Image.MAX_IMAGE_PIXELS = None

JEPA = "/workspace/Plexus/discovery_okuda/jepa"
MODEL_ID = "facebook/dinov2-large"
DEVICE = "cuda:1"
NROW, NCOL = 4, 8
RES = 224

# ImageNet statistics -- the normalisation DINOv2 was trained with.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_tiles(path):
    """-> uint8 (32,224,224,3), panel order row-major (row0col0..row3col7)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        W, H = im.size
        pw, ph = W // NCOL, H // NROW
        out = np.empty((NROW * NCOL, RES, RES, 3), dtype=np.uint8)
        k = 0
        for r in range(NROW):
            for c in range(NCOL):
                tile = im.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph))
                out[k] = np.asarray(tile.resize((RES, RES), Image.BICUBIC))
                k += 1
    return out, (W, H, pw, ph)


def main():
    d = np.load(os.path.join(JEPA, "design.npz"), allow_pickle=True)
    run_names = np.asarray(d["run_names"])
    run_dirs = np.asarray(d["run_dirs"])
    has_strip = np.asarray(d["has_strip"]).astype(bool)

    idx = np.where(has_strip)[0]
    paths = [os.path.join(run_dirs[i], "strip.png") for i in idx]
    ok = np.array([os.path.exists(p) for p in paths])
    if not ok.all():
        print("MISSING strip.png for", [run_names[i] for i, o in zip(idx, ok) if not o])
    idx = idx[ok]
    paths = [p for p, o in zip(paths, ok) if o]
    n = len(idx)
    print(f"embedding {n} strips (of {len(run_names)} runs, has_strip={has_strip.sum()})")

    torch.manual_seed(0)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    dim = model.config.hidden_size
    print(f"model {MODEL_ID}  hidden_size(dim) = {dim}  patch = {model.config.patch_size}")

    mean_d, std_d = MEAN.to(DEVICE), STD.to(DEVICE)
    panels = np.empty((n, NROW * NCOL, dim), dtype=np.float32)

    geom = None
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex, torch.no_grad():
        for j, (tiles, g) in enumerate(ex.map(load_tiles, paths)):
            geom = g
            x = torch.from_numpy(tiles).to(DEVICE)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - mean_d) / std_d
            out = model(pixel_values=x)
            # CLS token after the final layernorm == pooler_output for Dinov2Model
            cls = out.last_hidden_state[:, 0]
            panels[j] = cls.float().cpu().numpy()
            if (j + 1) % 50 == 0:
                print(f"  {j+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    print(f"forward done in {time.time()-t0:.0f}s; panel geometry W,H,pw,ph = {geom}")

    def l2(a, axis=-1):
        return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), 1e-12)

    # PRIMARY: per-row mean over the 8 time panels, each row L2-normed, concatenated.
    rows = panels.reshape(n, NROW, NCOL, dim).mean(axis=2)      # (n,4,dim)
    rows = l2(rows, axis=2)
    emb = l2(rows.reshape(n, NROW * dim), axis=1).astype(np.float32)

    # ALTERNATIVE kept for downstream: flat mean over all 32 panels.
    emb_meanpool = l2(panels.mean(axis=1), axis=1).astype(np.float32)
    # ALTERNATIVE: last time column only (final morphology), rows concatenated.
    last = panels.reshape(n, NROW, NCOL, dim)[:, :, -1, :]
    emb_last = l2(l2(last, axis=2).reshape(n, NROW * dim), axis=1).astype(np.float32)

    names = np.array([run_names[i] for i in idx])
    pos = {nm: k for k, nm in enumerate(names)}

    def cos(a, b, E):
        return float(np.dot(E[pos[a]], E[pos[b]]))

    PAIRS = [("b_star", "b_star_sharp"), ("b_star", "b_null_plain"),
             ("b_flower", "b_flower_death")]
    rng = np.random.default_rng(0)
    ii = rng.integers(0, n, 400)
    jj = rng.integers(0, n, 400)
    keep = ii != jj
    ii, jj = ii[keep][:200], jj[keep][:200]

    sanity = {}
    for tag, E in [("row_concat_4096", emb), ("meanpool_1024", emb_meanpool),
                   ("last_frame_4096", emb_last)]:
        rp = np.sum(E[ii] * E[jj], axis=1)
        s = {f"cos({a},{b})": cos(a, b, E) for a, b in PAIRS}
        s["random_pairs_mean"] = float(rp.mean())
        s["random_pairs_std"] = float(rp.std())
        s["random_pairs_min"] = float(rp.min())
        s["random_pairs_max"] = float(rp.max())
        sanity[tag] = s
        print(f"\n-- {tag} --")
        for k, v in s.items():
            print(f"   {k:34s} {v:+.4f}")

    TILING = (
        "strip.png is a regular 8x4 montage (8 time points x 4 render channels), so it is cut "
        "on its own grid into 32 panels of 440x450 px, each resized (not cropped) to 224x224 "
        "with a 2% aspect change; the 8 time panels are mean-pooled within each row, each row "
        "block is L2-normalised, and the 4 rows are concatenated -- keeping the four render "
        "channels from averaging into one another."
    )

    np.savez_compressed(
        os.path.join(JEPA, "embed_dinov2.npz"),
        emb=emb, emb_meanpool=emb_meanpool, emb_last=emb_last,
        panels=panels.astype(np.float16),
        run_names=names, run_dirs=np.array([run_dirs[i] for i in idx]),
        design_index=idx.astype(np.int64),
        model_id=np.array(MODEL_ID), dim=np.array(dim),
        emb_dim=np.array(emb.shape[1]),
        feature="cls_token_final_layernorm",
        grid=np.array([NROW, NCOL]), panel_size=np.array([geom[2], geom[3]]),
        resize=np.array(RES), device=np.array(DEVICE),
        tiling_decision=np.array(TILING),
        sanity_json=np.array(json.dumps(sanity)),
    )
    with open(os.path.join(JEPA, "embed_dinov2_sanity.json"), "w") as f:
        json.dump({"model": MODEL_ID, "dim": int(dim), "n_embedded": int(n),
                   "tiling": TILING, "sanity": sanity}, f, indent=2)
    print("\nsaved", os.path.join(JEPA, "embed_dinov2.npz"), "emb", emb.shape)


if __name__ == "__main__":
    main()
