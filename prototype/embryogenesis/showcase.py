#!/usr/bin/env python
"""showcase -- the FINAL render for an embryogenesis spec: one simulation, captured with the
engine's per-frame hook (stress & deformation are NOT in the trajectory), producing

  * summary2x2.mp4    -- the 2x2 summary:  a) blob (+n_cells/n_mpm)  b) stress
                         c) deformation + cell tracks   d) material flow field
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


def _panel_flow(ax, X, vel, W, vmin, vmax, title, quiver_n=400):
    """Material flow: particles coloured by SPEED |v| + a subsampled unit-direction quiver.
    Visualizes the inner circulation that drives membrane deformation."""
    ax.set_facecolor("black"); ax.set_xlim(0, W); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")
    spd = np.linalg.norm(vel, axis=1)
    ax.scatter(X[:, 0], X[:, 1], c=spd, s=2.0, cmap="turbo", vmin=vmin, vmax=vmax, edgecolors="none")
    N = X.shape[0]
    if N > 0:
        step = max(1, N // quiver_n)
        Xs, Vs = X[::step], vel[::step]
        mag = np.linalg.norm(Vs, axis=1, keepdims=True); mag[mag == 0] = 1.0
        U = Vs / mag * 0.028                                  # fixed-length arrows -> show DIRECTION field
        ax.quiver(Xs[:, 0], Xs[:, 1], U[:, 0], U[:, 1], color="white", alpha=0.45,
                  angles="xy", scale_units="xy", scale=1, width=0.003, headwidth=3, headlength=4)
    ax.set_title(title, color="white", fontsize=10)


def _draw_tracks(ax, hist, at, colors, W, tail, bg=True, title="cell tracks"):
    from matplotlib.collections import LineCollection
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    if bg:                                                    # bg=False -> OVERLAY on an existing panel
        ax.set_facecolor("black")
    if title:
        ax.set_title(title, color="white", fontsize=10)
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


def _organo_panel(mX, mocc, W, name, path):
    """Diagnostic panel for the Phase-3 organogenesis geometry: what the mask/skeleton/buds SAW.
    Left: body outline + skeleton (cyan) with tips (red) + branchpoints (yellow). Right: growth
    localization (grown material green over original blue). Plus the headline metrics."""
    import scorecard_organo as ORG
    lm = mocc[-1] > 0
    pd = ORG.panel_data(mX[-1][lm], W=W)
    grown = lm & (~(mocc[0] > 0))
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6)); fig.patch.set_facecolor("black")
    ext = [0, W, 0, 1]
    for ax in axs:
        ax.set_facecolor("black"); ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    # left: geometry
    axs[0].imshow(pd["body"].T, origin="lower", extent=ext, cmap="Greys_r", alpha=0.35, vmin=0, vmax=1)
    if pd["skel"].any():
        sy, sx = np.nonzero(pd["skel"].T); axs[0].scatter(sx / pd["skel"].shape[0] * W, sy / pd["skel"].shape[1],
                                                          s=0.5, c="#38e0e0")
    for m, col in ((pd["bpts"], "#ffd21e"), (pd["tips"], "#ff4d4d")):
        if m.any():
            yy, xx = np.nonzero(m.T); axs[0].scatter(xx / m.shape[0] * W, yy / m.shape[1], s=14, c=col, edgecolors="none")
    if pd["contour"] is not None:
        c = pd["contour"]; axs[0].plot(c[:, 0] / pd["body"].shape[0] * W, c[:, 1] / pd["body"].shape[1], "w-", lw=0.7)
    g = pd["metrics"]
    axs[0].set_title(f"geometry  n_buds={int(g['n_buds'])}  bpts={int(g['n_branchpoints'])}  "
                     f"solid={g['solidity']:.2f}", color="white", fontsize=9)
    # right: growth localization
    axs[1].scatter(mX[-1][lm & ~grown, 0], mX[-1][lm & ~grown, 1], s=1.5, c="#4a7fff", edgecolors="none")
    if grown.any():
        axs[1].scatter(mX[-1][grown, 0], mX[-1][grown, 1], s=1.5, c="#37d67a", edgecolors="none")
    axs[1].set_title(f"growth (green=new)  bud_score={g['bud_score']:.2f}  brn_score={g['branch_score']:.1f}",
                     color="white", fontsize=9)
    fig.suptitle(f"organogenesis geometry — {name}", color="white", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=110, facecolor="black"); plt.close(fig)


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

    caps = {"aX": [], "mX": [], "stress": [], "fnorm": [], "occ": [], "mocc": [], "saxis": []}
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
        caps["mocc"].append(p.occ.detach().cpu().numpy().copy())     # material occupancy: hide dormant grow_reserve
        caps["stress"].append(_stress_norm(p.F.detach(), p.mu, p.la).cpu().numpy())
        caps["fnorm"].append((p.F.detach() - eye).reshape(p.n, -1).norm(dim=1).cpu().numpy())
        if D == 2:                                            # principal STRAIN/stress axis angle per particle (for div_stress_angle)
            Fd = p.F.detach(); Cg = Fd.transpose(-1, -2) @ Fd     # right Cauchy-Green (symmetric)
            evec = torch.linalg.eigh(Cg)[1][:, :, -1]            # max-eigenvalue eigenvector
            caps["saxis"].append(torch.atan2(evec[:, 1], evec[:, 0]).cpu().numpy())

    run(sim, out_path=None, device=dev, on_frame=hook)
    aX = np.array(caps["aX"]); mX = np.array(caps["mX"])
    mocc = np.array(caps["mocc"]); mlive = mocc > 0                 # LIVE material only (dormant reserve hidden)
    stress = np.array(caps["stress"]); fnorm = np.array(caps["fnorm"]); at = at_box["at"]
    T = aX.shape[0]
    # two-blue material: mark the outer elastic MEMBRANE (deep blue) vs inner core (light blue),
    # by INITIAL radius, if the disc has a liquid layer.
    two_blue = any("liquid" in str(t.get("layers", "")) for t in sim.sets.get("cell", {}).get("types", {}).values())
    mem = None
    if two_blue:
        r0m = np.linalg.norm(mX[0] - np.array([0.5, 0.5]), axis=1)
        mem = r0m > 0.90 * np.quantile(r0m, 0.99)          # outer shell = membrane
    # colour ranges over LIVE material only (dormant reserve is frozen at F=I -> would bias the low end)
    s_lo, s_hi = np.percentile(stress[mlive], 2), np.percentile(stress[mlive], 98)
    f_lo, f_hi = np.percentile(fnorm[mlive], 2), np.percentile(fnorm[mlive], 98)
    # material flow: per-frame particle displacement (velocity up to the constant stride*dt factor)
    mvel = np.zeros_like(mX)
    if T > 1:
        mvel[1:] = mX[1:] - mX[:-1]
    vnrm = np.linalg.norm(mvel, axis=2)
    v_lo, v_hi = np.percentile(vnrm[mlive], 2), np.percentile(vnrm[mlive], 98)
    if v_hi <= v_lo:
        v_hi = v_lo + 1e-9
    print(f"[showcase] captured {T} frames in {time.time()-t0:.0f}s", flush=True)

    d = graphs_data_path(PRE, sim.name); os.makedirs(d, exist_ok=True)

    occ = np.array(caps["occ"])

    def draw2x2(fig, k):
        axs = fig.subplots(2, 2)
        lv = occ[k] > 0                                   # live agents
        lm = mlive[k]                                     # live material (dormant grow_reserve hidden)
        memk = mem[lm] if mem is not None else None
        _draw(axs[0, 0], mX[k][lm], aX[k][lv], at[lv], colors, blob, W, mem_mask=memk)
        axs[0, 0].set_title("cells + material", color="white", fontsize=10)
        axs[0, 0].text(0.02, 0.98, f"n_cells {int(lv.sum())}\nn_mpm {int(lm.sum())}",
                       transform=axs[0, 0].transAxes, color="white", fontsize=8, va="top", ha="left",
                       bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.5))
        _panel_scatter(axs[0, 1], mX[k][lm], stress[k][lm], W, "inferno", s_lo, s_hi, "stress")
        # deformation with the cell TRACKS overlaid on top (bottom-left)
        _panel_scatter(axs[1, 0], mX[k][lm], fnorm[k][lm], W, "viridis", f_lo, f_hi, "deformation + cell tracks")
        _draw_tracks(axs[1, 0], aX[:k + 1][:, lv], at[lv], colors, W, tail, bg=False, title=None)
        # material flow field (bottom-right)
        _panel_flow(axs[1, 1], mX[k][lm], mvel[k][lm], W, v_lo, v_hi, "material flow")

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
    # blob evolution montage (5 timepoints) -- the blob-only movie (blob.mp4) is intentionally NOT rendered
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2)); fig.patch.set_facecolor("black")
    for i, k in enumerate(ks):
        lv = occ[k] > 0; lm = mlive[k]; memk = mem[lm] if mem is not None else None
        _draw(axes[i], mX[k][lm], aX[k][lv], at[lv], colors, blob, W, mem_mask=memk); axes[i].set_title(f"t={k*stride} n={int(lv.sum())}", color="white", fontsize=9)
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
    # PHASE-3 ORGANOGENESIS geometry (outline / bud / branch / localization) from the live tissue mask
    import scorecard_organo as ORG
    og = ORG.compute({"mX": mX, "mocc": mocc, "fnorm": fnorm, "aX": aX, "at": at, "occ": occ}, W=W)
    _ogkeys = ("n_buds", "bud_score", "bud_len_bodyR", "bud_neck_ratio", "bud_persistence",
               "n_tips", "n_branchpoints", "branch_score", "branch_persistence", "tree_depth",
               "fragment_count", "solidity", "convexity", "aspect_ratio", "circularity",
               "growth_bud_overlap", "pattern_growth_overlap")
    m.update({f"org_{k}": og["final"].get(k) for k in _ogkeys})   # headline organo scores ride along
    with open(os.path.join(d, "metrics.json"), "w") as fh:
        json.dump({"name": sim.name, **m}, fh, indent=2)
    with open(os.path.join(d, "scorecard.json"), "w") as fh:
        json.dump({"name": sim.name, "pcts": sc["pcts"], "final": sc["final"], "evolution": sc["evolution"],
                   "organo": og}, fh, indent=2)
    _scorecard_panel(sc, sim.name, os.path.join(d, "scorecard.png"))
    try:
        _organo_panel(mX, mocc, W, sim.name, os.path.join(d, "organo.png"))
    except Exception as e:
        print(f"[showcase] organo panel skipped: {e}", flush=True)
    print(f"[showcase] {sim.name}: " + "  ".join(f"{k}={v}" for k, v in list(m.items())[:14]), flush=True)

    # save the EFFECTIVE run spec (base yaml + this slot's overrides) so every archive is self-documenting
    try:
        base_txt = open(spec_path).read()
    except OSError:
        base_txt = "# (base spec unreadable)\n"
    ovr = "  ".join(f"{k}={v}" for k, v in ov.items()) or "(none)"
    header = (f"# === effective run spec ===\n"
              f"# name:      {sim.name}\n"
              f"# base spec: {spec_path}\n"
              f"# frames={frames}  stride={stride}\n"
              f"# overrides: {ovr}\n"
              f"# (the override values above are applied ON TOP of the base yaml below)\n"
              f"# ==========================\n")
    with open(os.path.join(d, "spec.yaml"), "w") as fh:
        fh.write(header + base_txt)

    # archive
    adir = os.path.join(ARCHIVE, sim.name); os.makedirs(adir, exist_ok=True)
    for f in ("spec.yaml", "summary2x2.mp4", "summary2x2_final.png", "blob_evolution.png",
              "metrics.json", "scorecard.json", "scorecard.png", "organo.png"):
        src = os.path.join(d, f)
        if os.path.isfile(src):
            shutil.copy2(src, adir)

    # VLM caption (always, unless suppressed)
    if not no_cap:
        _caption([os.path.join(d, "summary2x2.mp4")], d)
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
