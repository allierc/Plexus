#!/usr/bin/env python
"""showcase -- the FINAL render for an embryogenesis spec: one simulation, captured with the
engine's per-frame hook (stress & deformation are NOT in the trajectory), producing

  * blob.mp4          -- cells-in-a-blob overlay (blue material + coloured cells)
  * summary2x2.mp4    -- the 2x2 summary:  a) blob   b) stress   c) deformation   d) cell tracks
  * *_evolution.png   -- static montages of both
  * metrics.json      -- confinement / stability numbers

then ARCHIVES everything under archive/<name>/ and, unless --no-caption, runs the VLM captioner
on the mp4s (the always-caption rule) so the emergent SHAPES get described.

    python prototype/embryogenesis/showcase.py <spec.yaml> [tag=show] [frames=1500] [stride=3] \
        [key=val overrides ...] [--no-caption]
"""
import os, sys, json, time, shutil, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "active_matter2"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import plexus.operators   # noqa
import am2_ops            # noqa
import plexus.schema as S
from plexus.engine import run
from plexus.paths import set_data_root, graphs_data_path
from plexus.generators.mpm_grid_diag import _stress_norm
from embryo_render import _type_colors, _blob_cmap, _draw, _ffmpeg
from tune import _apply, metrics   # reuse override + metric helpers

set_data_root(os.path.join(HERE, "data"))
PRE = "embryogenesis"
ARCHIVE = os.path.join(HERE, "archive")


def _mp4(frames_dir, out, fps=25):
    ff = _ffmpeg()
    if not ff:
        print("[showcase] no ffmpeg -> skip", out); return
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(frames_dir, "f%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", out], check=False)


def _panel_scatter(ax, X, val, W, cmap, vmin, vmax, title):
    ax.set_facecolor("black"); ax.set_xlim(0, W); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.scatter(X[:, 0], X[:, 1], c=val, s=2.0, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="none")
    ax.set_title(title, color="white", fontsize=10)


def _draw_tracks(ax, hist, at, colors, W, tail):
    ax.set_facecolor("black"); ax.set_xlim(0, W); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title("cell tracks", color="white", fontsize=10)
    seg = hist[-tail:].copy()                                # [k, N, 2]
    origin = (np.abs(seg[..., 0]) < 1e-4) & (np.abs(seg[..., 1]) < 1e-4)   # unborn/dormant -> no tail
    seg[origin] = np.nan
    if len(seg) >= 2:
        for ti, col in enumerate(colors):
            m = at == ti
            xs = seg[:, m, 0]; ys = seg[:, m, 1]              # [k, n]
            for j in range(0, xs.shape[1], 2):                # thin for speed
                ax.plot(xs[:, j], ys[:, j], color=col, lw=0.5, alpha=0.5)
    cur = hist[-1]
    for ti, col in enumerate(colors):
        m = at == ti
        ax.scatter(cur[m, 0], cur[m, 1], s=6, c=[col], edgecolors="none")


def main():
    spec_path = sys.argv[1]
    args = sys.argv[2:]
    no_cap = "--no-caption" in args
    ov = dict(kv.split("=", 1) for kv in args if "=" in kv)
    tag = ov.pop("tag", "show"); frames = int(ov.pop("frames", 1500))
    stride = int(ov.pop("stride", 3)); tail = int(ov.pop("tail", 30))
    sim = S.load(spec_path); sim.n_frames = frames
    sim.name = f"{sim.name}_{tag}"
    for k, v in ov.items():
        _apply(sim, k, v)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    colors = _type_colors(sim); blob = _blob_cmap(sim)
    W = float(getattr(sim, "world_size", [1.0])[0])
    print(f"[showcase] {sim.name}: frames={frames} stride={stride} overrides={ov}", flush=True)

    caps = {"aX": [], "mX": [], "stress": [], "fnorm": [], "occ": []}
    at_box = {}
    t0 = time.time()

    def hook(H, frame):
        if frame % stride:
            return
        a = H.levels["agent"]; p = H.levels["mpm_particle"]
        if "at" not in at_box:
            at_box["at"] = a.node_type.detach().cpu().numpy().copy()
        D = p.F.shape[-1]
        eye = torch.eye(D, device=p.F.device)
        caps["aX"].append(a.get("pos").detach().cpu().numpy().copy())
        caps["occ"].append(a.occ.detach().cpu().numpy().copy())
        caps["mX"].append(p.get("pos").detach().cpu().numpy().copy())
        caps["stress"].append(_stress_norm(p.F.detach(), p.mu, p.la).cpu().numpy())
        caps["fnorm"].append((p.F.detach() - eye).reshape(p.n, -1).norm(dim=1).cpu().numpy())

    run(sim, out_path=None, device=dev, on_frame=hook)
    aX = np.array(caps["aX"]); mX = np.array(caps["mX"])
    stress = np.array(caps["stress"]); fnorm = np.array(caps["fnorm"]); at = at_box["at"]
    T = aX.shape[0]
    # two-blue material: mark the outer elastic MEMBRANE (deep blue) vs inner core (light blue),
    # by INITIAL radius, if the disc has a liquid layer.
    two_blue = any("liquid" in str(t.get("layers", "")) for t in sim.sets.get("cell", {}).get("types", {}).values())
    mem = None
    if two_blue:
        r0m = np.linalg.norm(mX[0] - np.array([0.5, 0.5]), axis=1)
        mem = r0m > 0.90 * np.quantile(r0m, 0.99)          # outer shell = membrane
    s_lo, s_hi = np.percentile(stress, 2), np.percentile(stress, 98)
    f_lo, f_hi = np.percentile(fnorm, 2), np.percentile(fnorm, 98)
    print(f"[showcase] captured {T} frames in {time.time()-t0:.0f}s", flush=True)

    d = graphs_data_path(PRE, sim.name); os.makedirs(d, exist_ok=True)

    occ = np.array(caps["occ"])

    def draw2x2(fig, k):
        axs = fig.subplots(2, 2)
        lv = occ[k] > 0
        _draw(axs[0, 0], mX[k], aX[k][lv], at[lv], colors, blob, W, mem_mask=mem); axs[0, 0].set_title("cells + material", color="white", fontsize=10)
        _panel_scatter(axs[0, 1], mX[k], stress[k], W, "inferno", s_lo, s_hi, "stress")
        _panel_scatter(axs[1, 0], mX[k], fnorm[k], W, "viridis", f_lo, f_hi, "deformation")
        _draw_tracks(axs[1, 1], aX[:k + 1][:, lv], at[lv], colors, W, tail)

    # static 2x2 at 5 timepoints (evolution montage)
    ks = np.linspace(0, T - 1, 5).astype(int)
    for tag2, kk in [("final", [T - 1])]:
        pass
    fig = plt.figure(figsize=(8, 8)); fig.patch.set_facecolor("black"); draw2x2(fig, T - 1)
    fig.tight_layout(); fig.savefig(os.path.join(d, "summary2x2_final.png"), dpi=120, facecolor="black"); plt.close(fig)

    # 2x2 movie
    tmp = tempfile.mkdtemp()
    for k in range(T):
        fig = plt.figure(figsize=(8, 8)); fig.patch.set_facecolor("black")
        draw2x2(fig, k); fig.tight_layout()
        fig.savefig(os.path.join(tmp, f"f{k:05d}.png"), dpi=100, facecolor="black"); plt.close(fig)
    _mp4(tmp, os.path.join(d, "summary2x2.mp4"))
    # blob-only movie (panel a) + evolution montage
    for k in range(T):
        fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor("black")
        lv = occ[k] > 0
        _draw(ax, mX[k], aX[k][lv], at[lv], colors, blob, W, mem_mask=mem)
        fig.savefig(os.path.join(tmp, f"f{k:05d}.png"), dpi=100, facecolor="black"); plt.close(fig)
    _mp4(tmp, os.path.join(d, "blob.mp4"))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2)); fig.patch.set_facecolor("black")
    for i, k in enumerate(ks):
        lv = occ[k] > 0
        _draw(axes[i], mX[k], aX[k][lv], at[lv], colors, blob, W, mem_mask=mem); axes[i].set_title(f"t={k*stride} n={int(lv.sum())}", color="white", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(d, "blob_evolution.png"), dpi=120, facecolor="black"); plt.close(fig)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)

    # metrics: Phase-1 scientific observables (collapse / deform / flow / migration / segregation / accel)
    from embryo_metrics import phase1_from_arrays
    occ = np.array(caps["occ"])
    r0 = 0.024
    for o in sim.operators:
        if o.op == "repel":
            r0 = float(o.params.get("r0", r0))
    m = phase1_from_arrays(aX, occ, at, mX, r0=r0)
    m.update(frames=frames, seconds=round(time.time() - t0, 1))
    with open(os.path.join(d, "metrics.json"), "w") as fh:
        json.dump({"name": sim.name, **m}, fh, indent=2)
    print(f"[showcase] {sim.name}: " + "  ".join(f"{k}={v}" for k, v in m.items()), flush=True)

    # archive
    adir = os.path.join(ARCHIVE, sim.name); os.makedirs(adir, exist_ok=True)
    for f in ("summary2x2.mp4", "summary2x2_final.png", "blob.mp4", "blob_evolution.png", "metrics.json"):
        src = os.path.join(d, f)
        if os.path.isfile(src):
            shutil.copy2(src, adir)

    # VLM caption (always, unless suppressed)
    if not no_cap:
        _caption([os.path.join(d, "blob.mp4"), os.path.join(d, "summary2x2.mp4")], d)
        for cf in ("video_descriptions.txt",):
            src = os.path.join(graphs_data_path(), cf)
    print(f"[showcase] archived -> {adir}", flush=True)


def _caption(movies, data_dir):
    repo = os.path.abspath(os.path.join(HERE, "..", ".."))
    gemma = os.environ.get("GEMMA_DIR", os.path.join(repo, "VLLM", "gemma-4-12B-it"))
    script = os.path.join(repo, "VLLM", "describe_video.py")
    if not (os.path.isdir(gemma) and os.path.isfile(script)):
        print("[showcase] no VLM weights -> skip caption", flush=True); return
    out = os.path.join(data_dir, "captions.txt")
    movies = [m for m in movies if os.path.isfile(m)]
    print(f"[showcase] captioning {len(movies)} movie(s) -> {out}", flush=True)
    env = dict(os.environ, GEMMA_DIR=gemma)
    subprocess.run([sys.executable, script, *movies, "--root", data_dir, "--out", out, "--append"],
                   check=False, env=env)
    if os.path.isfile(out):
        shutil.copy2(out, os.path.join(ARCHIVE, os.path.basename(data_dir), "captions.txt"))


if __name__ == "__main__":
    main()
