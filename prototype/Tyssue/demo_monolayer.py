#!/usr/bin/env python
"""Archive the four monolayer-operator validations as demos (spec + strip.png + movie.mp4), each on the
substrate where it reads clearest:
  V1 geometry      -- SHELL (ball): per-cell 3D volume v=A*h, closed-volume sanity
  V2 bending       -- FLAT epithelium: bend it, apical(outer)>basal(inner) area => surface tension
                      penalises curvature (EMERGENT bending, no K_bend); surface energy climbs with curvature
  V3a rest         -- SHELL: relax at V_eq=rest -> settles to the kappa_s/k_v force balance (smaller, spherical)
  V3b buckle       -- FLAT epithelium: pin the rim, raise V_eq in a central spot -> in-plane growth has
                      nowhere to go but BUCKLE out of plane into a dome (Okuda's tube-initiation mechanism)
Each panel pairs a 3D view (apical shell coloured by the field, faint basal shell) with a CROSS-SECTION
(thickness ticks b_i->a_i) so the monolayer's thickness + apical/basal layers are explicit."""
import os, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
from tyssue_ops3d import build_sphere_mesh, face_geometry_3d
from tyssue_monolayer import monolayer_geometry_3d, _monolayer_energy_core, apical_basal_shells
from flat_mesh import build_flat_mesh

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
KV, KAPPA = 4.0, 0.20


def rings_of(es, et, ef, nF):
    r = [[] for _ in range(nF)]
    for k in range(len(ef)):
        r[int(ef[k])].append(int(es[k]))
    return r


def geom(pos, mesh, hc):
    es, et, ef, nF = mesh["es"], mesh["et"], mesh["ef"], mesh["nF"]
    v, s, A_ap, A_ba = monolayer_geometry_3d(pos, es, et, ef, nF, hc)
    a, b = apical_basal_shells(pos, es, et, ef, nF, hc)
    return v, s, A_ap, A_ba, a.numpy(), b.numpy()


def draw(ax3, axc, pos, mesh, hc, field, cmap, vlim, view, slice_axis, slice_plane, band, title=""):
    es, et, ef, nF = mesh["es"], mesh["et"], mesh["ef"], mesh["nF"]
    _, _, _, _, a, b = geom(pos, mesh, hc)
    rings = mesh["rings"]
    # --- 3D: apical shell coloured by field + faint basal shell -----------------------------------
    polA = [a[r] for r in rings]
    pc = Poly3DCollection(polA, cmap=cmap, edgecolor="k", linewidths=0.15, alpha=0.96)
    pc.set_array(np.asarray(field)); pc.set_clim(*vlim); ax3.add_collection3d(pc)
    polB = [b[r] for r in rings]
    pcb = Poly3DCollection(polB, facecolor=(0, 0, 0, 0), edgecolor=(0.5, 0.5, 0.5, 0.35), linewidths=0.1)
    ax3.add_collection3d(pcb)
    P = np.concatenate([a, b], 0); c = P.mean(0); rad = np.ptp(P - c, axis=0).max() * 0.55
    ax3.set_xlim(c[0]-rad, c[0]+rad); ax3.set_ylim(c[1]-rad, c[1]+rad); ax3.set_zlim(c[2]-rad, c[2]+rad)
    ax3.set_box_aspect((1, 1, 1)); ax3.view_init(*view); ax3.set_axis_off()
    if title:
        ax3.set_title(title, color="white", fontsize=9)
    # --- cross-section: thickness ticks b_i->a_i near the slice plane ------------------------------
    keep = np.where(np.abs(pos[:, slice_axis].numpy() - slice_plane) < band)[0]
    ox = [i for i in range(3) if i != slice_axis]                 # the two in-plane axes to plot (x, z)
    ox = [ox[0], 2] if slice_axis != 2 else [0, 1]
    segs = np.stack([b[keep][:, ox], a[keep][:, ox]], 1)
    from matplotlib.collections import LineCollection
    lc = LineCollection(segs, colors="deepskyblue", linewidths=1.4); axc.add_collection(lc)
    axc.scatter(a[keep][:, ox[0]], a[keep][:, ox[1]], s=4, c="orangered", label="apical", zorder=3)
    axc.scatter(b[keep][:, ox[0]], b[keep][:, ox[1]], s=4, c="royalblue", label="basal", zorder=3)
    allp = np.concatenate([a[keep][:, ox], b[keep][:, ox]], 0)
    if len(allp):
        mid = allp.mean(0); r2 = np.ptp(allp - mid, axis=0).max() * 0.6 + 1e-6
        axc.set_xlim(mid[0]-r2, mid[0]+r2); axc.set_ylim(mid[1]-r2, mid[1]+r2)
    axc.set_aspect("equal"); axc.set_facecolor("black"); axc.set_xticks([]); axc.set_yticks([])


def relax(pos0, mesh, V_eq, iters, move_mask=None, eta=0.05, cap_frac=0.10, kappa=KAPPA, kv=KV, rec=None):
    es, et, ef, nF = mesh["es"], mesh["et"], mesh["ef"], mesh["nF"]
    hc = mesh["hc"]; x = pos0.clone()
    eocc = torch.ones(es.shape[0]); vocc = torch.ones(x.shape[0]); R0t = torch.as_tensor(0.0)
    cap = cap_frac * (x[et] - x[es]).norm(dim=-1).mean()
    mm = None if move_mask is None else move_mask[:, None].to(x.dtype)
    frames = []
    for it in range(iters):
        xg = x.detach().requires_grad_(True)
        E = _monolayer_energy_core(xg, es, et, ef, nF, hc, V_eq, torch.ones(nF), kv, kappa, 0.0, 0.0, R0t, eocc, vocc)
        g = torch.nan_to_num(torch.autograd.grad(E, xg)[0])
        step = -eta * g
        step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
        if mm is not None:
            step = step * mm
        x = (x + step).detach()
        if rec is not None and (it % rec == 0 or it == iters - 1):
            frames.append(x.clone())
    return x, frames


def save(out, mesh, frames, fielder, cmap, vlim, view, slice_axis, slice_plane, band, titles, params, spin=0.0):
    os.makedirs(out, exist_ok=True)
    json.dump(params, open(os.path.join(out, "spec.json"), "w"), indent=1)
    hc = mesh["hc"]
    # strip: 2 rows (3D / cross-section) x 6 columns
    idx = np.linspace(0, len(frames) - 1, 6).round().astype(int)
    figS = plt.figure(figsize=(24, 7.6)); figS.patch.set_facecolor("black")
    for j, i in enumerate(idx):
        ax3 = figS.add_subplot(2, 6, j + 1, projection="3d"); ax3.set_facecolor("black")
        axc = figS.add_subplot(2, 6, 6 + j + 1)
        v = view if spin == 0 else (view[0], view[1] + spin * i)
        draw(ax3, axc, frames[i], mesh, hc, fielder(frames[i]), cmap, vlim, v, slice_axis, slice_plane, band, titles(i))
    figS.subplots_adjust(0.005, 0.005, 0.995, 0.96, wspace=0.03, hspace=0.03)
    figS.savefig(os.path.join(out, "strip.png"), dpi=95, facecolor="black"); plt.close(figS)
    # movie
    figM = plt.figure(figsize=(9.2, 5.0)); figM.patch.set_facecolor("black")
    wri = FFMpegWriter(fps=12)
    with wri.saving(figM, os.path.join(out, "movie.mp4"), dpi=90):
        for i in range(len(frames)):
            figM.clf()
            ax3 = figM.add_subplot(1, 2, 1, projection="3d"); ax3.set_facecolor("black")
            axc = figM.add_subplot(1, 2, 2)
            v = view if spin == 0 else (view[0], view[1] + spin * i)
            draw(ax3, axc, frames[i], mesh, hc, fielder(frames[i]), cmap, vlim, v, slice_axis, slice_plane, band, titles(i))
            wri.grab_frame()
    plt.close(figM)
    print(f"  wrote {out}/  ({len(frames)} frames)")


def mk_mesh(verts, es, et, ef, nF, h0, bmask=None):
    m = dict(es=torch.as_tensor(es), et=torch.as_tensor(et), ef=torch.as_tensor(ef), nF=nF,
             rings=rings_of(es, et, ef, nF), hc=torch.full((nF,), float(h0)), bmask=bmask)
    return m


# ============================ V1: SHELL geometry ============================
def demo_V1():
    R, H0, N = 5.0, 0.4, 220
    verts, es, et, ef, nF = build_sphere_mesh(N, R, jitter=0.15, seed=0)
    mesh = mk_mesh(verts, es, et, ef, nF, H0)
    pos = torch.as_tensor(verts)
    v, s, _, _, _, _ = geom(pos, mesh, mesh["hc"])
    A, _, _, _ = face_geometry_3d(pos, mesh["es"], mesh["et"], mesh["ef"], nF)
    vl = (float(v.min()), float(v.max()))
    frames = [pos] * 24                                          # static -> spin the camera
    save(os.path.join(HERE, "archive", "mono_V1_geometry_shell"), mesh, frames,
         fielder=lambda p: geom(p, mesh, mesh["hc"])[0].numpy(), cmap="viridis", vlim=vl,
         view=(22, 0), slice_axis=1, slice_plane=0.0, band=0.6, spin=15.0,
         titles=lambda i: "V1  cell volume v=A*h   (shell)" if i == 0 else "",
         params=dict(test="V1_geometry", substrate="shell", R=R, h0=H0, n_cells=nF,
                     sum_v=float(v.sum()), A_tot_h0=float(A.sum())*H0, ratio=float(v.sum())/(float(A.sum())*H0),
                     k_v=KV, kappa_s=KAPPA))


# ============================ V2: FLAT bending ============================
def demo_V2():
    verts, es, et, ef, nF, bmask = build_flat_mesh(k=13, L=10.0, jitter=0.5, seed=1)
    H0 = 0.6; mesh = mk_mesh(verts, es, et, ef, nF, H0, bmask)
    x0 = torch.as_tensor(verts); L = 10.0
    xs = (x0[:, 0] - L/2) / (L/2)                                # normalised in-plane coord for the arch
    amps = np.linspace(0.0, 3.2, 22)                             # bend amplitude (arch height)
    frames = [torch.stack([x0[:, 0], x0[:, 1], a * (1 - xs**2)], 1) for a in amps]
    def ratio(p):
        _, _, A_ap, A_ba, _, _ = geom(p, mesh, mesh["hc"]); return (A_ap / A_ba.clamp(min=1e-9)).numpy()
    def energy(p):
        _, s, _, _, _, _ = geom(p, mesh, mesh["hc"]); return float((KAPPA * s).sum())
    E0 = energy(frames[0])
    save(os.path.join(HERE, "archive", "mono_V2_bending_flat"), mesh, frames,
         fielder=ratio, cmap="coolwarm", vlim=(0.9, 1.3),
         view=(8, -80), slice_axis=1, slice_plane=5.0, band=0.9,
         titles=lambda i: ("V2  A_apical/A_basal (flat, bent)   surf-E climbs = bending stiffness" if i == 0 else ""),
         params=dict(test="V2_bending", substrate="flat_epithelium", h0=H0, n_cells=nF,
                     ratio_flat=float(ratio(frames[0]).mean()), ratio_bent=float(ratio(frames[-1]).mean()),
                     surfE_flat=E0, surfE_bent=energy(frames[-1]),
                     note="apical(convex outer) area exceeds basal => kappa_s penalises curvature, EMERGENT bending"))


# ============================ V3a: SHELL rest -> force balance ============================
def demo_V3a():
    R, H0, N = 5.0, 0.4, 220
    verts, es, et, ef, nF = build_sphere_mesh(N, R, jitter=0.15, seed=0)
    mesh = mk_mesh(verts, es, et, ef, nF, H0)
    pos = torch.as_tensor(verts)
    v0, _, _, _, _, _ = geom(pos, mesh, mesh["hc"])
    _, frames = relax(pos, mesh, v0.detach(), iters=260, rec=12)
    vl = (float(v0.min()) * 0.8, float(v0.max()))
    save(os.path.join(HERE, "archive", "mono_V3a_rest_shell"), mesh, frames,
         fielder=lambda p: geom(p, mesh, mesh["hc"])[0].numpy(), cmap="viridis", vlim=vl,
         view=(22, 25), slice_axis=1, slice_plane=0.0, band=0.6,
         titles=lambda i: ("V3a  rest -> kappa_s/k_v force balance (shell settles smaller, stays spherical)" if i == 0 else ""),
         params=dict(test="V3a_rest", substrate="shell", R0=R, h0=H0, n_cells=nF, k_v=KV, kappa_s=KAPPA,
                     R_start=float(pos.norm(dim=1).mean()), R_end=float(frames[-1].norm(dim=1).mean())))


# ============================ V3b: FLAT buckle ============================
def demo_V3b():
    verts, es, et, ef, nF, bmask = build_flat_mesh(k=15, L=10.0, jitter=0.45, seed=2)
    H0 = 0.4; mesh = mk_mesh(verts, es, et, ef, nF, H0, bmask)
    x0 = torch.as_tensor(verts).clone(); L = 10.0
    _, _, cen, _ = face_geometry_3d(x0, mesh["es"], mesh["et"], mesh["ef"], nF)
    rc = ((cen[:, 0] - L/2)**2 + (cen[:, 1] - L/2)**2).sqrt()
    spot = rc < 2.4                                              # central proliferation spot
    BOOST = 3.0
    v0, _, _, _, _, _ = geom(x0, mesh, mesh["hc"])
    V_eq = v0.detach().clone(); V_eq[spot] = v0[spot] * BOOST    # raise target volume in the spot
    move = torch.as_tensor(~bmask)                              # pin the rim (clamped edges)
    rv = (x0[:, 0] - L/2)**2 + (x0[:, 1] - L/2)**2               # central bump biases the buckle upward
    x0[:, 2] = x0[:, 2] + 0.4 * torch.exp(-rv / 4.0)
    # buckling INCREASES area (dome>flat), so it only wins when growth dominates surface tension: use the
    # low-kappa_s regime (Okuda's tubulation regime) -- at high kappa_s the spot grows in-plane and stays flat.
    KB = 0.05
    _, frames = relax(x0, mesh, V_eq, iters=700, move_mask=move, eta=0.05, kappa=KB, rec=28)
    save(os.path.join(HERE, "archive", "mono_V3b_buckle_flat"), mesh, frames,
         fielder=lambda p: spot.numpy().astype(float), cmap="YlOrRd", vlim=(0, 1),
         view=(16, -82), slice_axis=1, slice_plane=5.0, band=0.9,
         titles=lambda i: ("V3b  localized v_eq (red spot) -> BUCKLE out of plane (Okuda tube init)" if i == 0 else ""),
         params=dict(test="V3b_buckle", substrate="flat_epithelium", h0=H0, n_cells=nF, spot_cells=int(spot.sum()),
                     veq_boost=BOOST, kappa_s=KB, k_v=KV, z_max_start=float(frames[0][:, 2].max()),
                     z_max_end=float(frames[-1][:, 2].max()), z_min_end=float(frames[-1][:, 2].min())))


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["V1", "V2", "V3a", "V3b"]
    fns = {"V1": demo_V1, "V2": demo_V2, "V3a": demo_V3a, "V3b": demo_V3b}
    for w in which:
        print(f"[demo {w}]"); fns[w]()
    print("DONE")
