#!/usr/bin/env python
"""Standalone DIFFERENTIABLE mechanics analysis of a run.

    python analyze_forces.py <archive_folder> [n_frames]

Re-runs the deterministic sim behind <folder> (from its preset / spec.yaml), then quantifies, per recorded
frame and per cell: FORCE (=-grad U, torch-differentiable), STRESS (isotropic pressure + cortical tension),
MOVEMENT (centroid velocity), cell DIVISION (births) and TRACKING (centroid nearest-neighbour), plus
topology (neighbour count). Writes:
  * quantification.npz  -- per-frame per-cell tables (object arrays; load with allow_pickle=True) + summary
  * analysis.mp4        -- 2x2: 3D force heatmap | 3D stress heatmap | division heatmap | cell tracking

The mechanics core `cell_mechanics(pos, ...)` is torch-differentiable (force = -grad U, set diff=True to
keep the graph), so a downstream loss -- e.g. maximise the outward normal force at the tip, minimise stress
in the tube wall -- can be optimised w.r.t. positions or (through the rollout) parameters."""
import os, sys, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import run_tyssue_round as R
from run_tyssue_round import engine_run
from mesh_ops import face_geometry_3d

torch.set_default_dtype(torch.float64)


def cell_mechanics(pos, es, et, ef, nF, A0, P0, V0f, kA, kP, kV, Lam, Gam, diff=False):
    """Differentiable vertex-model mechanics at one configuration. Returns force[Nv,3], per-cell
    area/perim/vol, pressure (2kV(V0f-v): +ve = below-target volume, pushes OUT), cortical tension, and
    centroids. force = -grad U; diff=True keeps the autograd graph (for optimisation)."""
    x = pos.clone().requires_grad_(True)
    area, perim, cen, vf = face_geometry_3d(x, es, et, ef, nF)
    U = (kA * (area - A0) ** 2 + kP * (perim - P0) ** 2 + 0.5 * Gam * perim ** 2 + kV * (vf - V0f) ** 2).sum() \
        + Lam * (x[et] - x[es]).norm(dim=-1).sum()
    force = -torch.autograd.grad(U, x, create_graph=diff)[0]
    pressure = 2.0 * kV * (V0f - vf)
    tension = 2.0 * kP * (perim - P0) + Gam * perim + Lam
    r = (force if diff else force.detach())
    return r, area.detach(), perim.detach(), vf.detach(), pressure.detach(), tension.detach(), cen.detach()


def _per_cell(vertex_scalar, es, ef, nF):
    """Mean of a per-vertex scalar over each cell's ring vertices."""
    num = torch.zeros(nF).index_add(0, ef, vertex_scalar[es])
    cnt = torch.zeros(nF).index_add(0, ef, torch.ones(es.shape[0]))
    return (num / cnt.clamp(min=1)).numpy()


def _neighbours(es, et, ef, nF):
    """Per-cell neighbour count = distinct adjacent faces (shared edges), via the half-edge twin."""
    Nv = int(max(es.max(), et.max())) + 1
    key = es * Nv + et; twin = et * Nv + es
    order = torch.argsort(key); ks = key[order]
    pos = torch.searchsorted(ks, twin).clamp(max=key.shape[0] - 1)
    tw = torch.where(ks[pos] == twin, ef[order[pos]], ef)        # neighbour face across each half-edge
    deg = torch.zeros(nF).index_add(0, ef, (tw != ef).double())
    return deg.numpy()


def run(folder, nfr=48):
    name = os.path.basename(os.path.normpath(folder))
    print(f"[analyze] {name}: re-running sim ...", flush=True)
    if name in R.PRESETS:
        sim, cfg = R.make(R.PRESETS[name])
    else:
        sim = R.S.load(os.path.join(folder, "spec.yaml")); cfg = None
    se = next(o for o in (cfg or {"operators": []})["operators"] if o["op"] == "cell_mechanics") if cfg else {}
    kA, kP, kV = se.get("K_A", 1.0), se.get("K_P", 1.0), se.get("K_V", 4.0)
    Lam, Gam = se.get("Lambda", 0.2), se.get("Gamma", 0.05)
    Hf, out = engine_run(sim, device="cpu")
    m = Hf.level("vertex")._mesh; hist = m["hist"]
    posf = out["sets"]["vertex"]["pos"]; chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]
    idx = np.unique(np.linspace(0, min(T, len(hist)) - 1, nfr).astype(int))

    Q = {k: [] for k in ("t", "cen", "act", "force", "pressure", "tension", "vel", "neigh", "born", "nF")}
    prev_cen = None
    for t in idx:
        h = hist[int(t)]; nF = int(h["nF"]); Nv = int(h["Nv"])
        es, et, ef = (torch.as_tensor(h[k][: (h["E_srce"].shape[0])], dtype=torch.long) for k in ("E_srce", "E_trgt", "E_face"))
        pos = torch.as_tensor(posf[int(t)][:Nv], dtype=torch.float64)
        A0 = torch.as_tensor(h["A0"][:nF]) if h.get("A0") is not None else torch.zeros(nF)
        P0 = torch.as_tensor(h["P0"][:nF]) if h.get("P0") is not None else torch.zeros(nF)
        V0f = torch.as_tensor(h["V0f"][:nF]) if h.get("V0f") is not None else torch.zeros(nF)
        force, area, perim, vf, pres, tens, cen = cell_mechanics(pos, es, et, ef, nF, A0, P0, V0f, kA, kP, kV, Lam, Gam)
        fmag = _per_cell(force.norm(dim=-1), es, ef, nF)         # per-cell mean vertex force magnitude
        cen = cen.numpy()
        vel = np.zeros(nF)
        if prev_cen is not None:                                # velocity = centroid nearest-neighbour displacement
            d = np.linalg.norm(cen[:, None, :] - prev_cen[None, :, :], axis=2)
            vel = d[np.arange(nF), d.argmin(1)]
        born = np.zeros(nF);
        if Q["nF"]:
            born[Q["nF"][-1]:] = 1.0                             # cells appended since last frame = just divided
        prev_cen = cen
        Q["t"].append(int(t)); Q["cen"].append(cen); Q["act"].append(chemf[int(t)][:nF, 0].copy())
        Q["force"].append(fmag); Q["pressure"].append(pres.numpy()); Q["tension"].append(tens.numpy())
        Q["vel"].append(vel); Q["neigh"].append(_neighbours(es, et, ef, nF)); Q["born"].append(born); Q["nF"].append(nF)

    # --- tracking: sample cells from the FINAL frame (so TUBE cells, born late, are included) and track
    #     them BACKWARD by nearest centroid -> paths back to their origin ---
    cL = Q["cen"][-1]; sel = np.linspace(0, len(cL) - 1, min(90, len(cL))).astype(int)
    bwd = [cL[sel]]
    for k in range(len(Q["cen"]) - 2, -1, -1):
        c = Q["cen"][k]; nxt = bwd[-1]
        nn = np.linalg.norm(c[None, :, :] - nxt[:, None, :], axis=2).argmin(1)
        bwd.append(c[nn])
    tracks = np.array(bwd[::-1])                                 # [frames, n_sel, 3], forward order

    summary = dict(frames=len(idx), cells_start=Q["nF"][0], cells_end=Q["nF"][-1],
                   force_mean=float(np.mean([f.mean() for f in Q["force"]])),
                   pressure_absmean=float(np.mean([np.abs(p).mean() for p in Q["pressure"]])),
                   vel_mean=float(np.mean([v.mean() for v in Q["vel"][1:]])) if len(Q["vel"]) > 1 else 0.0,
                   n_divisions=int(sum(b.sum() for b in Q["born"])))
    np.savez(os.path.join(folder, "quantification.npz"),
             t=np.array(Q["t"]), tracks=tracks, summary=json.dumps(summary),
             **{k: np.array(Q[k], dtype=object) for k in ("cen", "act", "force", "pressure", "tension", "vel", "neigh", "born")})
    print(f"[analyze] {name}: {summary}", flush=True)
    _render(folder, name, Q, tracks, idx)
    return summary


def _axes3d(ax, cen, title):
    R_ = float(np.abs(cen).max()) * 1.05
    ax.set_xlim(-R_, R_); ax.set_ylim(-R_, R_); ax.set_zlim(-R_, R_)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(18, 30); ax.set_axis_off()
    ax.set_title(title, color="white", fontsize=9)


def _scat(ax, cen, val, cmap, title, vlim):
    ax.clear(); ax.set_facecolor("black")
    ax.scatter(cen[:, 0], cen[:, 1], cen[:, 2], c=np.asarray(val), cmap=cmap, s=7, vmin=vlim[0], vmax=vlim[1])
    _axes3d(ax, cen, title)


def _div(ax, cen, born, ncum):
    ax.clear(); ax.set_facecolor("black")
    m = np.asarray(born) > 0.5
    ax.scatter(cen[~m, 0], cen[~m, 1], cen[~m, 2], s=2, c="dimgray", alpha=0.12)   # non-dividing: transparent
    if m.any():
        ax.scatter(cen[m, 0], cen[m, 1], cen[m, 2], s=6, c="red", alpha=0.95)      # dividing: red
    _axes3d(ax, cen, f"cell division ({ncum})")                          # count in the title


def _track(ax, cen, tracks, vel, vthr):
    ax.clear(); ax.set_facecolor("black")
    mig = np.asarray(vel) > vthr                                         # only MIGRATING cells (velocity threshold)
    ax.scatter(cen[~mig, 0], cen[~mig, 1], cen[~mig, 2], s=2, c="dimgray", alpha=0.12)   # slow: transparent
    if mig.any():
        ax.scatter(cen[mig, 0], cen[mig, 1], cen[mig, 2], s=6, c="deepskyblue", alpha=0.95)
    for j in range(tracks.shape[1]):
        ax.plot(tracks[:, j, 0], tracks[:, j, 1], tracks[:, j, 2], "-", lw=0.6, alpha=0.6)
    _axes3d(ax, cen, "cell tracking (migrating)")


def _render(folder, name, Q, tracks, idx):
    flim = (0, float(np.percentile(np.concatenate(Q["force"]), 97)))
    P = float(np.percentile(np.abs(np.concatenate(Q["pressure"])), 97)); plim = (-P, P)
    ncum = np.cumsum([float(b.sum()) for b in Q["born"]]).astype(int)
    vall = np.concatenate([v for v in Q["vel"][1:]]) if len(Q["vel"]) > 1 else np.array([0.0])
    vthr = float(np.percentile(vall, 70))                               # migrating = top-30% velocity
    fig = plt.figure(figsize=(9, 8.4)); fig.patch.set_facecolor("black")
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
    wri = FFMpegWriter(fps=8, metadata={"title": f"{name} analysis"})
    with wri.saving(fig, os.path.join(folder, "analysis.mp4"), dpi=90):
        for k in range(len(Q["cen"])):
            cen = Q["cen"][k]
            _scat(axes[0], cen, Q["force"][k], "inferno", "force  |-grad U|", flim)
            _scat(axes[1], cen, Q["pressure"][k], "coolwarm", "stress  (pressure)", plim)
            _div(axes[2], cen, Q["born"][k], int(ncum[k]))
            _track(axes[3], cen, tracks[:k + 1], Q["vel"][k], vthr)
            wri.grab_frame()
    plt.close(fig); print(f"[analyze] wrote {folder}/analysis.mp4 + quantification.npz", flush=True)


def render_only(folder):
    """Re-render analysis.mp4 from an existing quantification.npz (no sim re-run) -- fast viz iteration."""
    d = np.load(os.path.join(folder, "quantification.npz"), allow_pickle=True)
    Q = {k: list(d[k]) for k in ("cen", "act", "force", "pressure", "tension", "vel", "neigh", "born")}
    Q["t"] = list(d["t"]); Q["nF"] = [len(c) for c in Q["cen"]]
    _render(folder, os.path.basename(os.path.normpath(folder)), Q, d["tracks"], None)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":       # analyze_forces.py render <folder>  (fast re-render)
        render_only(sys.argv[2])
    else:
        folder = sys.argv[1] if len(sys.argv) > 1 else "archive/round_34_900"
        run(folder, int(sys.argv[2]) if len(sys.argv) > 2 else 48)
