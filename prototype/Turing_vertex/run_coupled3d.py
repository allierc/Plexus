#!/usr/bin/env python
"""run_coupled3d -- Stage 4 in 3D: the FULL coupled Turing x vertex model on a GROWING
MONOLAYER VESICLE (Okuda et al. 2018, Figs 4-5).

The paper's deformation tissues are hollow monolayer cell vesicles (~200 cells enclosing a
lumen), grown by converting each cell's activator concentration into its growth rate. Here
one `cell` set on a fixed buffer carries three independently-integrated blocks (the engine
multi-block extension):

    pos  (coordinate)   <- voronoi_tension_3d   3D shape-energy force (K_V vol + K_S surface)
    chem = [act,inh]    <- graph_diffuse + react  Turing RD on the Voronoi cell graph
    v0   (target vol)   <- growth   activator (mitogen) grows v0 (Hill);  divide_2x at 2x

Two regimes:
  * vesicle_grow    -- activator-INDEPENDENT uniform growth: the vesicle inflates and patterns
                       (Turing on a growing sphere, Fig 4 in 3D);
  * vesicle_tubulate-- activator-DRIVEN growth (mitogen): activator spots grow/divide faster,
                       so the monolayer buckles out of plane -> undulation/tubulation (Fig 5).

Cells coloured by activator (white->red), black cell-delimiting lines; octant cutaway reveals
the lumen; a cross-section shows the monolayer ring.

    python run_coupled3d.py [--only vesicle_grow]
    python run_coupled3d.py --rerender
"""
from __future__ import annotations
import os, sys, argparse, json, tempfile, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators   # noqa: F401
import turing_ops         # noqa: F401  graph_diffuse + react (INTEGRAND="chem")
import coupled_ops        # noqa: F401  growth + divide_2x (block="v0")
import vertex3d_ops       # noqa: F401  tissue_seed_3d + voronoi_graph_3d + voronoi_tension_3d
import shell_ops          # noqa: F401  membrane_bending + voronoi_tension_shell + lumen_pressure
from shell_ops import shell_faces
from vertex3d_ops import cell_shape_3d, cell_prisms_3d
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive")
PANEL = 4.4
CMAP = plt.get_cmap("Reds")


def presets():
    """Figure 4 (3D): Turing patterning on a UNIFORMLY GROWING monolayer vesicle. Uniform
    (activator-independent) growth just inflates the vesicle, so the vertex3d_ops spherical-ghost
    mechanics stays valid (no tubulation) -- the interest is the pattern behaviour on a growing
    domain (spots insert / are maintained; the paper's hysteresis). Rendered as EXTRUDED PRISMS
    (cell_prisms_3d) for clean epithelial paving. gamma sets the patterning time scale.
    (Fig 5 tubulation needs the apical/basal + bending mechanics in shell_ops.py -- deferred.)"""
    BR = {"implementation": "brusselator", "gamma": 2.0, "A": 1.0, "B": 3.0}
    base = dict(lumen=True, R=6.0, dt=0.02, chi=5.0, d_a=0.05, d_h=0.7, react=BR,
                s0=5.3, mu=0.02, vmax=1.0, ratio=2.0, K_V=10.0, K_S=1.0, k_bend=0.35, k_lloyd=0.3,
                lam_ref=0.25, rho_lam=1.0, a_sw=1.0, hill=2.0)                 # UNIFORM growth (activator-independent)
    # Fig 5: activator-driven (mitogen) QUASI-STATIC growth on the apical/basal SHELL mechanics +
    # membrane_bending (prevents single-cell spikes -> coherent folds) + lumen_pressure (area growth
    # buckles instead of inflating). Fast RD (gamma high) so the pattern pre-forms before much growth.
    BR5 = {"implementation": "brusselator", "gamma": 4.0, "A": 1.0, "B": 3.0}
    base5 = dict(lumen=True, R=6.0, dt=0.02, d_a=0.05, react=BR5, s0=5.3, mu=0.015, vmax=0.8,
                 ratio=2.0, K_V=10.0, K_S=1.0, k_bend=0.45, mechanics="shell",
                 lam_ref=0.15, rho_lam=0.03, a_sw=1.6, hill=8.0,                  # quasi-static MITOGEN
                 k_lumen=0.006, v_target_scale=1.1, mu_lumen=0.02)
    # Gray-Scott regimes (archive/shell_{coral,labyrinth,holes}) on the FULL GROWING vesicle coupling
    # (division + surface_lloyd uniform cells + mechanics), like fig4_vesicle. Gray-Scott needs a large
    # dt to develop (F,kk are fixed by the regime, can't be rescaled), so run the whole coupling at
    # dt=0.35 with GENTLE mechanics (small mu, tight vmax clamp) so the vertex model stays stable.
    def GS(F, kk):
        return {"implementation": "gray_scott", "F": F, "kk": kk}
    gs = dict(lumen=True, R=8.0, dt=0.35, norm=False, seed_mode="scatter", seed_frac=0.04,
              d_a=0.08, d_h=0.16, chi=0.28, react=GS(0.058, 0.063), compile=False,   # growing N -> vectorised eager (32x, robust)
              s0=5.3, mu=0.006, vmax=0.6, ratio=2.0, K_V=10.0, K_S=1.0, k_bend=0.2, k_lloyd=0.15,
              lam_ref=0.006, rho_lam=1.0, a_sw=0.25, hill=2.0,                    # UNIFORM growth (real vertex mechanics)
              v0noise=0.3, reset_noise=0.4, retess_every=5, lloyd_every=3)        # multi-rate: topology/5, lloyd/3 ticks
    return [
        dict(base, name="fig4_vesicle",      n0=600, buffer=3000, frames=1000),                  # slow patterning
        dict(base, name="fig4_vesicle_fast", n0=600, buffer=3000, frames=1000, chi=5.0,
             react={"implementation": "brusselator", "gamma": 8.0, "A": 1.0, "B": 3.0}),          # fast patterning -> spots insert
        dict(gs, name="fig4_coral",     n0=1000, buffer=2600, frames=9000, react=GS(0.058, 0.063)),   # coral on growing vesicle
        dict(gs, name="fig4_labyrinth", n0=1000, buffer=2600, frames=9000, react=GS(0.029, 0.054)),   # labyrinth
        dict(gs, name="fig4_holes",     n0=1000, buffer=2600, frames=9000, react=GS(0.039, 0.058)),   # holes
        dict(base5, name="fig5_tube_thick", n0=500, buffer=2400, frames=1600, chi=6.0, d_h=0.8),  # big domains -> thick tubes
        dict(base5, name="fig5_tube_thin",  n0=500, buffer=2400, frames=1600, chi=2.5, d_h=0.5),  # small domains -> thin tubes
    ]


def measure_V0(R, N, lumen):
    W = 2.6 * R
    cfg = {"general": {"name": "seed", "seed": 0, "n_frames": 1, "dt": 0.02, "boundary": "free",
                       "dim": 3, "world": [W, W, W]},
           "sets": {"cell": {"n": N}}, "fields": {},
           "operators": [{"op": "tissue_seed_3d", "at": "cell", "radius": R, "lumen": lumen,
                          "v0": 1.0, "before_frame": 1}], "schedule": ["tissue_seed_3d"]}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    _, out = engine_run(sim, device="cpu")
    p0 = out["sets"]["cell"]["pos"][-1].astype(np.float64)
    vol, _, _, ok = cell_shape_3d(p0, lumen, 0.15 * R)
    return float(np.median(vol[ok > 0]))


def make_spec(p, V0):
    W = 8.0 * p["R"]                                    # generous box; the vesicle grows/deforms into it
    A = p["react"].get("A", 1.0); B = p["react"].get("B", 3.0)
    mech = p.get("mechanics", "ghost")                  # ghost (Fig 4, spherical shells) | shell (Fig 5, apical/basal)
    seed_mode = p.get("seed_mode", "noise")             # noise (Brusselator/GM) | scatter (Gray-Scott)
    thick = 0.7 * V0 ** (1.0 / 3.0)
    ops = [
        {"op": "tissue_seed_3d", "at": "cell", "radius": p["R"], "lumen": p["lumen"], "v0": V0,
         "a_mean": A, "h_mean": B / A, "cnoise": 0.04, "seed_mode": seed_mode,
         "seed_frac": p.get("seed_frac", 0.04), "v0noise": p.get("v0noise", 0.0), "before_frame": 1},
        {"op": "voronoi_graph_3d", "at": "cell", "radius": p["R"], "lumen": p["lumen"]},
        {"op": "graph_diffuse", "at": "cell", "d_a": p["d_a"], "d_h": p["d_h"], "chi": p["chi"],
         "norm": p.get("norm", True)},
        {"op": "react", "at": "cell", **p["react"]},
    ]
    if p.get("static_rd"):                              # RD-only reproduction on the vesicle GEOMETRY (no growth/mechanics)
        return _wrap(p, ops, W)
    ops += [
        {"op": "growth", "at": "cell", "block": "v0", "lam_ref": p["lam_ref"], "rho_lam": p["rho_lam"],
         "a_sw": p["a_sw"], "hill": p["hill"], "cap": p["ratio"]},
        {"op": "divide_2x", "at": "cell", "block": "v0", "ratio": p["ratio"], "offset": 0.12,
         "reset_noise": p.get("reset_noise", 0.0)},
    ]
    if p.get("k_lloyd", 0) > 0:                          # tangential relaxation -> equal-area hexagonal cells
        ops.append({"op": "surface_lloyd", "at": "cell", "k_lloyd": p["k_lloyd"], "vmax": p["vmax"],
                    "every": p.get("lloyd_every", 1)})   # multi-rate: this slow relaxation needn't run every tick
    if mech == "shell":                                 # Fig 5: apical/basal monolayer + lumen pressure
        ops.append({"op": "voronoi_tension_shell", "at": "cell", "s0": p["s0"], "V0": V0,
                    "K_V": p["K_V"], "K_S": p["K_S"], "mu": p["mu"], "thickness": thick, "vmax": p["vmax"]})
        ops.append({"op": "membrane_bending", "at": "cell", "k_bend": p["k_bend"], "vmax": p["vmax"]})
        ops.append({"op": "lumen_pressure", "at": "cell", "k_lumen": p.get("k_lumen", 0.005),
                    "v_target_scale": p.get("v_target_scale", 1.1), "mu": p.get("mu_lumen", 0.02), "vmax": p["vmax"]})
    else:                                               # Fig 4: spherical-ghost Voronoi (vectorised + torch.compiled)
        ops.append({"op": "voronoi_tension_3d", "at": "cell", "s0": p["s0"], "radius": p["R"], "V0": V0,
                    "K_V": p["K_V"], "K_S": p["K_S"], "mu": p["mu"], "lumen": p["lumen"], "vmax": p["vmax"],
                    "compile": p.get("compile", False),      # vectorised eager is robust on growing N
                    "retess_every": p.get("retess_every", 1)})   # multi-rate: cache topology, refresh every K ticks + on division
        ops.append({"op": "membrane_bending", "at": "cell", "k_bend": p["k_bend"], "vmax": p["vmax"]})
    return _wrap(p, ops, W)


def _wrap(p, ops, W):
    cfg = {
        "general": {"name": f"coupled3d_{p['name']}", "seed": 0, "n_frames": p["frames"], "dt": p["dt"],
                    "boundary": "free", "dim": 3, "world": [W, W, W]},
        "sets": {"cell": {"n": p["n0"], "buffer": p["buffer"], "state": {
            "pos":  {"width": 3, "integration": "second_order_coordinate", "boundary": "free"},
            "vel":  {"width": 3, "integration": "second_order_rate", "boundary": "free"},
            "chem": {"width": 2, "integration": "first_order", "boundary": "free"},
            "v0":   {"width": 1, "integration": "first_order", "boundary": "free"}}}},
        "fields": {},
        "operators": ops,
        "schedule": [o["op"] for o in ops],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


# --------------------------------------------------------------------------- #
#  Render: EXTRUDED PRISM monolayer coloured by activator (clean epithelial paving)
# --------------------------------------------------------------------------- #
def _prism_slice(prism_faces, avals, axis, c0):
    """Cross-section polygons: each cell's prism cut by the plane coord[axis]=c0, + activator.
    Slices every prism face (fan-triangulated) -> the monolayer reads as a ring of wall segments."""
    other = [a for a in (0, 1, 2) if a != axis]
    out = []
    for faces, a in zip(prism_faces, avals):
        if faces is None:
            continue
        pts = []
        for poly in faces:
            for t in range(1, len(poly) - 1):
                tri = np.array([poly[0], poly[t], poly[t + 1]]); w = tri[:, axis]
                for u, v in ((0, 1), (1, 2), (2, 0)):
                    if (w[u] - c0) * (w[v] - c0) < 0:
                        tt = (c0 - w[u]) / (w[v] - w[u]); pts.append((tri[u] + tt * (tri[v] - tri[u]))[other])
        if len(pts) >= 3:
            pts = np.array(pts); ce = pts.mean(0)
            out.append((pts[np.argsort(np.arctan2(pts[:, 1] - ce[1], pts[:, 0] - ce[0]))], a))
    return out


def render(pos_t, act_t, occ_t, outdir, name, lumen, R, thickness=0.8, face_mode="prism",
           seconds=11.0, movie_frames=80):
    os.makedirs(outdir, exist_ok=True)
    T = pos_t.shape[0]
    lf = occ_t[-1] > 0
    box = float(np.nanmax([np.ptp(pos_t[-1][lf][:, k]) for k in range(3)])) * 1.1 + 1.0
    aall = act_t[occ_t > 0]
    norm = mcolors.Normalize(float(np.percentile(aall, 2)), float(np.percentile(aall, 98)))

    def frame(t):
        """-> (cell_polys, avals, cen, ncell); cell_polys[k] = list of polygon faces (or None)."""
        live = occ_t[t] > 0
        pl = pos_t[t][live].astype(np.float64); al = act_t[t][live]; cen = pl.mean(0)
        if face_mode == "shell":                          # deformed monolayer: apical/basal Voronoi cells
            faces, avals, _ = shell_faces(pl, al, thickness)
            return [list(f) for f in faces], avals, cen, int(live.sum())
        # near-spherical: clean prisms of the ACTUAL cell positions (uniformity comes from the
        # surface_lloyd operator in the simulation, not from relaxing the render).
        pf, apical, vol, surf, ok = cell_prisms_3d(pl, thickness, center=cen, relax=0)
        return pf, al, cen, int(live.sum())

    def poly3d(ax, cell_polys, avals, cen):
        ax.clear(); ax.set_facecolor("black")
        tris, cols = [], []
        for polys, a in zip(cell_polys, avals):
            if polys is None:
                continue
            rc = CMAP(norm(a)); rgba = (rc[0], rc[1], rc[2], 0.95)
            for f in polys:
                tris.append(f); cols.append(rgba)
        ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors=(0, 0, 0, 0.45), linewidths=0.12))
        ax.set_xlim(cen[0] - box / 2, cen[0] + box / 2); ax.set_ylim(cen[1] - box / 2, cen[1] + box / 2)
        ax.set_zlim(cen[2] - box / 2, cen[2] + box / 2); ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()

    def label(ax, txt):
        ax.text2D(0.02, 0.98, txt, transform=ax.transAxes, color="white", fontsize=7, va="top", family="monospace")

    # strip: CLOSED external views @0 / 50% / 100% (front, front, back) + cross-section @100%
    # (the cross-section reveals the lumen -- no artificial cutaway hole)
    picks = [0, T // 2, T - 1]
    views = [(22, 35), (22, 35), (22, 215)]
    tags = ["external", "external", "external back"]
    sfig = plt.figure(figsize=(4 * PANEL, PANEL)); sfig.patch.set_facecolor("black")
    last = None
    for i, t in enumerate(picks):
        cp, av, cen, ncell = frame(t)
        if t == T - 1:
            last = (cp, av, cen)
        ax = sfig.add_subplot(1, 4, i + 1, projection="3d"); ax.set_facecolor("black")
        poly3d(ax, cp, av, cen); ax.view_init(*views[i])
        label(ax, f"{name}\n{tags[i]} {int(100*t/(T-1))}%\nN={ncell}  activator(red)")
    # cross-section (last frame): the monolayer reads as an activator-coloured ring around the lumen
    pf, av, cen = last
    axc = sfig.add_subplot(1, 4, 4); axc.set_facecolor("black")
    for poly, a in _prism_slice(pf, av, 2, float(cen[2])):
        axc.fill(poly[:, 0], poly[:, 1], facecolor=CMAP(norm(a)), alpha=0.95, edgecolor="black", lw=0.5)
    hb = box * 0.5
    axc.set_xlim(cen[0] - hb, cen[0] + hb); axc.set_ylim(cen[1] - hb, cen[1] + hb)
    axc.set_aspect("equal"); axc.axis("off")
    sfig.subplots_adjust(0.005, 0.005, 0.995, 0.995, wspace=0.02)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)

    # movie: rotating CLOSED external view (no cutaway)
    midx = np.linspace(0, T - 1, min(T, movie_frames)).astype(int)
    fig = plt.figure(figsize=(PANEL, PANEL)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(projection="3d"); ax.set_facecolor("black"); fig.subplots_adjust(0, 0, 1, 1)
    w = FFMpegWriter(fps=max(1, round(len(midx) / seconds)), metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=110):
        for k, t in enumerate(midx):
            cp, av, cen, _ = frame(t)
            poly3d(ax, cp, av, cen); ax.view_init(20, 35 + 360.0 * k / len(midx))
            w.grab_frame()
    plt.close(fig)


def diagnostics(pos_t, act_t, occ_t):
    n0, nT = int((occ_t[0] > 0).sum()), int((occ_t[-1] > 0).sum())
    L = occ_t[-1] > 0; c = pos_t[-1][L].mean(0); rad = np.linalg.norm(pos_t[-1][L] - c, axis=1)
    aT = act_t[-1][L]
    return dict(n_start=n0, n_end=nT, grew=bool(nT > n0 * 1.15),
                roughness=round(float(rad.std() / max(rad.mean(), 1e-9)), 3),
                v_std=round(float(aT.std()), 3), patterned=bool(aT.std() > 0.1),
                nan=bool(np.isnan(pos_t[-1]).any()))


def run_all(only=None):
    for p in presets():
        if only and p["name"] not in only:
            continue
        odir = os.path.join(OUT, p["name"]); os.makedirs(odir, exist_ok=True)
        print(f"[coupled3d] {p['name']}: n0={p['n0']} lam_ref={p['lam_ref']} rho_lam={p['rho_lam']}", flush=True)
        rec = {k: p[k] for k in ("n0", "buffer", "frames", "chi", "lam_ref", "rho_lam", "ratio")}
        try:
            V0 = measure_V0(p["R"], p["n0"], p["lumen"]); rec["V0"] = round(V0, 3)
            sim, cfg = make_spec(p, V0)
            yaml.safe_dump(cfg, open(os.path.join(odir, "spec.yaml"), "w"), sort_keys=False)
            _, out = engine_run(sim, device="cpu")
            cell = out["sets"]["cell"]
            pos = cell["pos"]; act = cell["state"]["chem"][..., 0]; occ = cell["occ"]
            T = pos.shape[0]; keep = np.linspace(0, T - 1, min(T, 120)).astype(int)
            pos_d, act_d, occ_d = pos[keep].astype("float32"), act[keep].astype("float32"), occ[keep].astype("float32")
            diag = diagnostics(pos_d, act_d, occ_d); rec.update(diag)
            thickness = 0.7 * V0 ** (1.0 / 3.0)               # monolayer height ~ cell size
            fmode = "shell" if p.get("mechanics") == "shell" else "prism"   # deformed -> apical/basal faces
            np.savez_compressed(os.path.join(odir, "traj.npz"), pos=pos_d, act=act_d, occ=occ_d,
                                lumen=p["lumen"], R=p["R"], name=p["name"], thickness=thickness, face_mode=fmode)
            render(pos_d, act_d, occ_d, odir, p["name"], p["lumen"], p["R"], thickness=thickness, face_mode=fmode)
            print(f"     -> N {diag['n_start']}->{diag['n_end']} roughness={diag['roughness']} "
                  f"patterned={diag['patterned']}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


def rerender(only=None):
    for p in presets():
        if only and p["name"] not in only:
            continue
        odir = os.path.join(OUT, p["name"]); tf = os.path.join(odir, "traj.npz")
        if not os.path.exists(tf):
            print(f"[rerender] {p['name']}: no traj.npz"); continue
        d = np.load(tf)
        thick = float(d["thickness"]) if "thickness" in d else 0.8
        fmode = str(d["face_mode"]) if "face_mode" in d else "prism"
        render(d["pos"], d["act"], d["occ"], odir, str(d["name"]), bool(d["lumen"]), float(d["R"]),
               thickness=thick, face_mode=fmode)
        print(f"[rerender] {p['name']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerender", action="store_true"); ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.rerender:
        rerender(a.only)
    else:
        run_all(a.only)


if __name__ == "__main__":
    main()
