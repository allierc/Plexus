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
    from matplotlib.collections import LineCollection
    ax.set_facecolor("black"); ax.set_xlim(0, W); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title("cell tracks", color="white", fontsize=10)
    seg = hist[-tail:].copy()                                # [k, N, 2]  (longer tail = more history)
    origin = (np.abs(seg[..., 0]) < 1e-4) & (np.abs(seg[..., 1]) < 1e-4)   # unborn/dormant -> no tail
    seg[origin] = np.nan
    N = at.shape[0]
    s = float(max(1.5, 7.0 * (400.0 / max(N, 1)) ** 0.5))    # SAME dot size as the top-left panel
    k = seg.shape[0]
    if k >= 2:
        rec = (np.arange(1, k, dtype=float) / max(k - 1, 1)) ** 1.6   # recency 0(old)->1(new)
        for ti, col in enumerate(colors):
            P = seg[:, at == ti, :]                          # [k, nm, 2]
            nm = P.shape[1]
            if nm == 0:
                continue
            segs = np.stack([P[:-1], P[1:]], axis=2).reshape(-1, 2, 2)   # [(k-1)*nm, 2, 2]
            alpha = np.repeat(rec, nm) * 0.8                 # TRANSPARENT tail -> opaque head
            valid = ~np.isnan(segs).any(axis=(1, 2))
            rgba = np.zeros((segs.shape[0], 4)); rgba[:, :3] = col; rgba[:, 3] = alpha
            ax.add_collection(LineCollection(segs[valid], colors=rgba[valid], linewidths=1.2))
    cur = hist[-1]
    for ti, col in enumerate(colors):
        m = at == ti
        ax.scatter(cur[m, 0], cur[m, 1], s=s, c=[col], edgecolors="none")


_PANEL_KEYS = [
    ("shape", ["fourier_m2", "fourier_m3", "circularity", "shape_index", "deform_rms"]),
    ("organization", ["nn_mean", "gr_peak", "density_cv", "contact_same"]),
    ("flow", ["speed", "polar_order", "net_circulation", "msd", "corr_length_xi"]),
    ("topology", ["t1_rate"]),
    ("partition", ["segregation_index", "mixing_entropy", "mi_type_x"]),
    ("coupling", ["stress_cell_corr", "deform_cell_corr", "div_stress_angle"]),
]


def _scorecard_panel(sc, name, path):
    """Grid of metric-EVOLUTION line plots (value vs 5/25/50/75/100% of the run)."""
    ev = sc["evolution"]; pcts = sc["pcts"]
    items = [(fam, k) for fam, ks in _PANEL_KEYS for k in ks if k in ev]
    ncol = 4; nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.2 * nrow))
    fig.patch.set_facecolor("black"); axes = np.atleast_1d(axes).ravel()
    fcol = {"shape": "#7fd1ff", "organization": "#ffd24a", "flow": "#8affc1",
            "partition": "#ff7a7a", "coupling": "#c79bff"}
    for ax, (fam, k) in zip(axes, items):
        y = [np.nan if v is None else v for v in ev[k]]
        ax.set_facecolor("black"); ax.plot(pcts, y, "-o", color=fcol.get(fam, "w"), lw=1.5, ms=3)
        ax.set_title(f"{fam}:{k}", color="white", fontsize=7.5)
        ax.tick_params(colors="white", labelsize=6); [s.set_color("#444") for s in ax.spines.values()]
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle(f"scorecard evolution — {name}", color="white", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=110, facecolor="black"); plt.close(fig)


def main():
    spec_path = sys.argv[1]
    args = sys.argv[2:]
    no_cap = "--no-caption" in args
    ov = dict(kv.split("=", 1) for kv in args if "=" in kv)
    tag = ov.pop("tag", "show"); frames = int(ov.pop("frames", 1500))
    stride = int(ov.pop("stride", 3)); tail = int(ov.pop("tail", 60))   # longer tails on the tracks panel
    sim = S.load(spec_path); sim.n_frames = frames
    sim.name = f"{sim.name}_{tag}"
    for k, v in ov.items():
        _apply(sim, k, v)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    colors = _type_colors(sim); blob = _blob_cmap(sim)
    W = float(getattr(sim, "world_size", [1.0])[0])
    print(f"[showcase] {sim.name}: frames={frames} stride={stride} overrides={ov}", flush=True)

    caps = {"aX": [], "mX": [], "stress": [], "fnorm": [], "occ": [], "saxis": []}
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
        if D == 2:                                            # principal STRAIN/stress axis angle per particle (for div_stress_angle)
            Fd = p.F.detach(); Cg = Fd.transpose(-1, -2) @ Fd     # right Cauchy-Green (symmetric)
            evec = torch.linalg.eigh(Cg)[1][:, :, -1]            # max-eigenvalue eigenvector
            caps["saxis"].append(torch.atan2(evec[:, 1], evec[:, 0]).cpu().numpy())

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

    # metrics: hard-failure gate (phase1) + the QUANTITATIVE SCORECARD (shape/org/flow/partition/coupling,
    # each at 5/25/50/75/100% of the run) -> the loop DECIDES on numbers, not on visual captions.
    from embryo_metrics import phase1_from_arrays
    import scorecard as SC
    occ = np.array(caps["occ"])
    r0 = 0.024
    for o in sim.operators:
        if o.op == "repel":
            r0 = float(o.params.get("r0", r0))
    m = phase1_from_arrays(aX, occ, at, mX, r0=r0)
    m.update(frames=frames, seconds=round(time.time() - t0, 1))
    sc = SC.compute({"aX": aX, "occ": occ, "at": at, "mX": mX, "stress": stress, "fnorm": fnorm,
                     "saxis": np.array(caps["saxis"]) if caps["saxis"] else None},
                    W=W, r0=r0, dt=float(getattr(sim, "dt", 0.002)))
    m.update(sc["final"])                                     # final scorecard metrics ride along for ranking
    with open(os.path.join(d, "metrics.json"), "w") as fh:
        json.dump({"name": sim.name, **m}, fh, indent=2)
    with open(os.path.join(d, "scorecard.json"), "w") as fh:
        json.dump({"name": sim.name, "pcts": sc["pcts"], "final": sc["final"], "evolution": sc["evolution"]}, fh, indent=2)
    _scorecard_panel(sc, sim.name, os.path.join(d, "scorecard.png"))
    print(f"[showcase] {sim.name}: " + "  ".join(f"{k}={v}" for k, v in list(m.items())[:14]), flush=True)

    # archive
    adir = os.path.join(ARCHIVE, sim.name); os.makedirs(adir, exist_ok=True)
    for f in ("summary2x2.mp4", "summary2x2_final.png", "blob.mp4", "blob_evolution.png",
              "metrics.json", "scorecard.json", "scorecard.png"):
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
