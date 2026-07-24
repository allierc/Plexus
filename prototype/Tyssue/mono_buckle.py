#!/usr/bin/env python
"""Sound study of the monolayer BUCKLE (mono_V3b): a flat epithelium with a localized target-volume increase
buckles out of plane. Parameterized + cluster-dispatchable (python mono_buckle.py --only <preset>), with a
DIFFERENTIABLE monolayer force/stress analysis and migration, so each variant tests a physical hypothesis.

Mechanics (monolayer energy, Okuda Eq.3):  U = sum_j [ 1/2 k_v (v-v_eq)^2 + kappa_s s + 1/2 gamma P^2 ] .
  force  = -dU/dx  (autograd)        pressure p = 2 k_v (v_eq - v)  (>0: below-target volume, pushes out)
  tension T = gamma P                migration = per-cell centroid velocity
Each preset writes archive/<preset>/ : quantification.npz, analysis.mp4 (2x2 force|stress|migration|shape),
strip.png, params.json."""
import os, sys, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
from flat_mesh import build_flat_mesh
from tyssue_monolayer import monolayer_geometry_3d, apical_basal_shells
from tyssue_ops3d import face_geometry_3d
from tyssue_topology_ops3d import rings_from_flat_3d

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- physical levers (see monolayer_tube_note.tex): kappa_s/k_v = surface tension vs volume stiffness
# (buckle amplitude), h0 = thickness (emergent bending), gamma = cell rounding, boost = growth magnitude,
# ramp = quasi-static (grow gradually, relax each step) vs instant. spot_r = activated-disc radius.
_BASE = dict(k=15, L=10.0, jitter=0.45, seed=2, h0=0.4, spot_r=2.4, boost=2.8, kappa_s=0.05, k_v=4.0,
             gamma=0.03, iters=650, eta=0.05, smooth_w=0.10, ramp=1, rec=26)
PRESETS = {
    "mb_base":   dict(_BASE),                                   # = mono_V3b baseline
    # H1: lower surface tension -> bigger buckle; higher -> stays flat
    "mb_k02":    dict(_BASE, kappa_s=0.02),
    "mb_k10":    dict(_BASE, kappa_s=0.10),
    "mb_k20":    dict(_BASE, kappa_s=0.20),
    # H2: more growth -> bigger buckle (but more stress)
    "mb_boost20": dict(_BASE, boost=2.0),
    "mb_boost40": dict(_BASE, boost=4.0),
    # H3: thickness -> emergent bending rigidity (thicker = stiffer wall, rounder/straighter)
    "mb_h03":    dict(_BASE, h0=0.3),
    "mb_h06":    dict(_BASE, h0=0.6),
    # H4: volume stiffness
    "mb_kv6":    dict(_BASE, k_v=6.0),
    # H5: QUASI-STATIC ramp (grow gradually) -> lower stress than instant
    "mb_ramp":   dict(_BASE, ramp=20),
    "mb_ramp_k02": dict(_BASE, ramp=20, kappa_s=0.02),
}


def perim(x, es, et, ef, nF):
    return torch.zeros(nF).index_add(0, ef, (x[et] - x[es]).norm(dim=-1))


def mechanics(pos, es, et, ef, nF, hc, V_eq, k_v, kappa_s, gamma):
    """Differentiable monolayer mechanics at one config -> force[Nv,3], v, s, pressure[nF], tension[nF]."""
    x = pos.clone().requires_grad_(True)
    v_f, s_f, _, _ = monolayer_geometry_3d(x, es, et, ef, nF, hc)
    U = (0.5 * k_v * (v_f - V_eq) ** 2).sum() + kappa_s * s_f.sum() + 0.5 * gamma * (perim(x, es, et, ef, nF) ** 2).sum()
    force = -torch.autograd.grad(U, x)[0]
    pressure = 2.0 * k_v * (V_eq - v_f)
    tension = gamma * perim(x, es, et, ef, nF)
    return force.detach(), v_f.detach(), s_f.detach(), pressure.detach(), tension.detach()


def relax(pos0, es, et, ef, nF, hc, V_eq_fn, move, iters, eta, kappa_s, k_v, gamma, smooth_w, rec):
    """Bounded-Euler descent on the monolayer energy; V_eq_fn(it) allows a quasi-static RAMP. Records frames."""
    x = pos0.clone(); Nv = x.shape[0]; ones_e = torch.ones(es.shape[0])
    cap = 0.10 * (x[et] - x[es]).norm(dim=-1).mean(); mm = move[:, None].to(x.dtype)
    frames = []
    for it in range(iters):
        V_eq = V_eq_fn(it)
        xg = x.detach().requires_grad_(True)
        v_f, s_f, _, _ = monolayer_geometry_3d(xg, es, et, ef, nF, hc)
        U = (0.5 * k_v * (v_f - V_eq) ** 2).sum() + kappa_s * s_f.sum() + 0.5 * gamma * (perim(xg, es, et, ef, nF) ** 2).sum()
        g = torch.nan_to_num(torch.autograd.grad(U, xg)[0]); step = -eta * g
        step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
        x = (x + step * mm).detach()
        if smooth_w > 0 and it % 2 == 0:                        # tangential Lloyd smoothing
            nbr = torch.zeros_like(x).index_add(0, es, x[et]); deg = torch.zeros(Nv).index_add(0, es, ones_e).clamp(min=1)
            disp = smooth_w * (nbr / deg[:, None] - x)
            Nf = 0.5 * torch.zeros(nF, 3).index_add(0, ef, torch.cross(x[es], x[et], dim=-1))
            vn = torch.zeros_like(x).index_add(0, es, Nf[ef]); n = vn / (vn.norm(dim=-1, keepdim=True) + 1e-12)
            x = (x + (disp - (disp * n).sum(-1, keepdim=True) * n) * mm).detach()
        if it % rec == 0 or it == iters - 1:
            frames.append(x.clone())
    return frames


def cell_centroids(pos, es, et, ef, nF):
    rings = rings_from_flat_3d(es.numpy(), et.numpy(), ef.numpy(), nF)
    return np.array([pos[r].mean(0).numpy() if r is not None and len(r) else [0, 0, 0] for r in rings])


def do(preset):
    p = PRESETS[preset]; OUT = os.path.join(HERE, "archive", preset); os.makedirs(OUT, exist_ok=True)
    verts, es, et, ef, nF, bmask = build_flat_mesh(k=p["k"], L=p["L"], jitter=p["jitter"], seed=p["seed"])
    es, et, ef = torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef)
    L = p["L"]; hc = torch.full((nF,), p["h0"])
    x0 = torch.as_tensor(verts).clone()
    _, _, cen0, _ = face_geometry_3d(x0, es, et, ef, nF)
    spot = (((cen0[:, 0] - L/2)**2 + (cen0[:, 1] - L/2)**2).sqrt() < p["spot_r"])
    v0, _, _, _ = monolayer_geometry_3d(x0, es, et, ef, nF, hc)
    rv = (x0[:, 0] - L/2)**2 + (x0[:, 1] - L/2)**2
    x0[:, 2] = x0[:, 2] + 0.4 * torch.exp(-rv / 4.0)            # tiny bump -> buckle up

    def V_eq_fn(it):                                            # ramp the boost over the first `ramp` frac of iters
        frac = min(1.0, (it + 1) / max(1, int(p["ramp"] / 100.0 * p["iters"]))) if p["ramp"] > 1 else 1.0
        Veq = v0.detach().clone(); Veq[spot] = v0[spot] * (1.0 + frac * (p["boost"] - 1.0)); return Veq
    move = torch.as_tensor(~bmask)
    frames = relax(x0, es, et, ef, nF, hc, V_eq_fn, move, p["iters"], p["eta"], p["kappa_s"], p["k_v"],
                   p["gamma"], p["smooth_w"], p["rec"])
    V_eq = V_eq_fn(p["iters"] - 1)

    # --- per-frame differentiable mechanics + migration ---
    Q = {k: [] for k in ("cen", "force", "pressure", "tension", "vel", "z")}
    prev = None
    for fr in frames:
        force, v_f, s_f, pres, tens = mechanics(fr, es, et, ef, nF, hc, V_eq, p["k_v"], p["kappa_s"], p["gamma"])
        fmag = (torch.zeros(nF).index_add(0, ef, force[es].norm(dim=-1))
                / torch.zeros(nF).index_add(0, ef, torch.ones(es.shape[0])).clamp(min=1)).numpy()
        cen = cell_centroids(fr, es, et, ef, nF)
        vel = np.zeros(nF)
        if prev is not None:
            vel = np.linalg.norm(cen - prev, axis=1)
        prev = cen
        Q["cen"].append(cen); Q["force"].append(fmag); Q["pressure"].append(pres.numpy())
        Q["tension"].append(tens.numpy()); Q["vel"].append(vel); Q["z"].append(cen[:, 2].copy())
    summ = dict(preset=preset, **{k: p[k] for k in ("kappa_s", "k_v", "h0", "gamma", "boost", "ramp")},
                z_max=float(frames[-1][:, 2].max()), n_cells=nF, spot_cells=int(spot.sum()),
                force_mean=float(np.mean(Q["force"][-1])), pressure_absmean=float(np.abs(Q["pressure"][-1]).mean()),
                pressure_spot=float(np.abs(np.array(Q["pressure"][-1])[spot.numpy()]).mean()))
    np.savez(os.path.join(OUT, "quantification.npz"), summary=json.dumps(summ),
             **{k: np.array(Q[k], dtype=object) for k in Q})
    json.dump(summ, open(os.path.join(OUT, "params.json"), "w"), indent=1)
    _render(OUT, preset, Q)
    print(f"[{preset}] z_max={summ['z_max']:.2f} force={summ['force_mean']:.2f} "
          f"p_spot={summ['pressure_spot']:.2f} p_mean={summ['pressure_absmean']:.2f}", flush=True)
    return summ


def _ax(ax, cen, val, cmap, title, vlim):
    ax.clear(); ax.set_facecolor("black")
    ax.scatter(cen[:, 0], cen[:, 1], cen[:, 2], c=np.asarray(val), cmap=cmap, s=9, vmin=vlim[0], vmax=vlim[1])
    zr = float(np.abs(cen[:, 2]).max()) + 0.3
    ax.set_xlim(2, 8); ax.set_ylim(2, 8); ax.set_zlim(-zr, zr)
    ax.set_box_aspect((1, 1, 0.7)); ax.view_init(22, -70); ax.set_axis_off()
    ax.set_title(title, color="white", fontsize=10)


def _render(OUT, preset, Q):
    flim = (0, float(np.percentile(np.concatenate(Q["force"]), 97)))
    P = float(np.percentile(np.abs(np.concatenate(Q["pressure"])), 97)); plim = (-P, P)
    vlim = (0, float(np.percentile(np.concatenate(Q["vel"][1:]) if len(Q["vel"]) > 1 else [1], 95)) + 1e-9)
    zl = (float(np.min(np.concatenate(Q["z"]))), float(np.max(np.concatenate(Q["z"]))))
    fig = plt.figure(figsize=(9.4, 8.2)); fig.patch.set_facecolor("black")
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
    wri = FFMpegWriter(fps=7)
    with wri.saving(fig, os.path.join(OUT, "analysis.mp4"), dpi=92):
        for k in range(len(Q["cen"])):
            cen = Q["cen"][k]
            _ax(axes[0], cen, Q["force"][k], "inferno", "force  |-grad U|", flim)
            _ax(axes[1], cen, Q["pressure"][k], "coolwarm", "stress  (pressure)", plim)
            _ax(axes[2], cen, Q["vel"][k], "viridis", "migration  (velocity)", vlim)
            _ax(axes[3], cen, Q["z"][k], "YlOrRd", "buckle shape  (height)", zl)
            wri.grab_frame()
    plt.close(fig)
    # strip: last frame's four fields
    figS = plt.figure(figsize=(18, 4.4)); figS.patch.set_facecolor("black")
    cen = Q["cen"][-1]
    for j, (val, cm, t, vl) in enumerate([(Q["force"][-1], "inferno", "force", flim),
                                          (Q["pressure"][-1], "coolwarm", "pressure", plim),
                                          (Q["vel"][-1], "viridis", "migration", vlim),
                                          (Q["z"][-1], "YlOrRd", "height", zl)]):
        _ax(figS.add_subplot(1, 4, j + 1, projection="3d"), cen, val, cm, t, vl)
    figS.savefig(os.path.join(OUT, "strip.png"), dpi=100, facecolor="black"); plt.close(figS)


if __name__ == "__main__":
    args = sys.argv[1:]
    preset = args[args.index("--only") + 1] if "--only" in args else (args[0] if args else "mb_base")
    do(preset)
