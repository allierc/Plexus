#!/usr/bin/env python
"""run_tyssue2d -- Stage 1 of the Tyssue AVM prototype: force-balanced 2D vertex mechanics
and the p0 rigidity transition, as a TRUE vertex model (half-edge mesh + explicit vertices),
the sibling of Turing_vertex's Self-Propelled-Voronoi run_vertex2d.py.

A disordered honeycomb (jittered cell centres -> Voronoi -> half-edge mesh) relaxes to force
balance under the AVM shape energy E = sum_f K_A(A-A0)^2 + K_P(P-P0)^2 (tyssue_ops.py):
seed_mesh -> shape_energy (inner gradient-descent to residual). We sweep the target shape
index p0: below p0*~3.81 the tissue jams (finite residual energy, cells stay compact); above
it the cells can satisfy A0 and P0 at once (residual energy -> 0, elongated cells). The
relaxed <q> and residual energy vs p0 is the rigidity-transition signature (Bi et al. 2015).

    python run_tyssue2d.py            # run the p0 sweep, archive each + transition curve
    python run_tyssue2d.py --montage
    python run_tyssue2d.py --rerender
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
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators   # noqa: F401
import tyssue_ops         # noqa: F401  registers seed_mesh + shape_energy
from tyssue_ops import build_honeycomb, face_polygons
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_specfmt import write_spec

OUT = os.path.join(HERE, "archive")
NX, NY, A, BORDER, JITTER, SEED = 16, 20, 1.0, 1, 0.35, 0
FRAMES, RELAX_ITERS, ETA = 140, 8, 0.08     # inner steps/frame: relaxation visible over the first ~third, fully converged by the end


def presets():
    #     name          p0     (denser grid to pin the rigidity transition)
    return [("solid_3p60", 3.60), ("solid_3p70", 3.70), ("solid_3p75", 3.75),
            ("crit_3p78", 3.78), ("crit_3p81", 3.81), ("crit_3p84", 3.84),
            ("fluid_3p90", 3.90), ("fluid_4p00", 4.00), ("fluid_4p10", 4.10)]


def _mesh_dict():
    """Deterministic mesh (topology fixed in Stage 1) for rendering + diagnostics."""
    verts, es, et, ef, fc, pin, a0 = build_honeycomb(NX, NY, A, BORDER, JITTER, SEED)
    nF = int(ef.max()) + 1
    return dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=verts.shape[0], a0=a0,
                pin=pin, verts0=verts)


def make_spec(name, p0, mesh):
    Nv = mesh["Nv"]
    cfg = {
        "general": {"name": f"tyssue2d_{name}", "seed": SEED, "n_frames": FRAMES, "dt": 1.0,
                    "boundary": "free", "dim": 2, "world": [float(NX * A + 4), float(NY * A + 4)]},
        "sets": {"vertex": {"n": Nv}},
        "fields": {},
        "operators": [
            {"op": "seed_mesh", "at": "vertex", "nx": NX, "ny": NY, "a": A, "border": BORDER,
             "jitter": JITTER, "p0": p0, "seed": SEED, "before_frame": 1},
            {"op": "shape_energy", "at": "vertex", "p0": p0, "K_A": 1.0, "K_P": 1.0,
             "mu": 1.0, "dt": 1.0, "relax_iters": RELAX_ITERS, "eta": ETA},
        ],
        "schedule": ["seed_mesh", "shape_energy"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _frame(pos_np, mesh, p0):
    m = dict(E_srce=mesh["E_srce"], E_trgt=mesh["E_trgt"], E_face=mesh["E_face"], nF=mesh["nF"])
    polys, area, perim, shape = face_polygons(pos_np, m)
    a0 = mesh["a0"]; P0 = p0 * np.sqrt(a0)
    energy = (area - a0) ** 2 + (perim - P0) ** 2
    return polys, area, perim, shape, energy


def render(pos_traj, mesh, outdir, name, p0, seconds=8.0, max_frames=120):
    os.makedirs(outdir, exist_ok=True)
    T = pos_traj.shape[0]
    idx = list(range(0, T, max(1, -(-T // max_frames))))
    fps = max(1, round(len(idx) / seconds))
    PANEL = 4.4
    # crop to the INTERIOR (drop the ragged pinned border) so the tissue fills the panel
    pin = mesh.get("pin"); free = (~pin) if pin is not None else np.ones(mesh["verts0"].shape[0], bool)
    Wx = mesh["verts0"][free, 0]; Wy = mesh["verts0"][free, 1]
    cx, cy = 0.5 * (Wx.min() + Wx.max()), 0.5 * (Wy.min() + Wy.max())
    half = 0.46 * max(Wx.max() - Wx.min(), Wy.max() - Wy.min())          # square, tight
    xlim = (cx - half, cx + half); ylim = (cy - half, cy + half)
    norm = TwoSlopeNorm(vcenter=3.81, vmin=3.70, vmax=max(3.95, p0 + 0.1))

    def draw(ax, t):
        ax.clear(); ax.set_facecolor("black")
        polys, _, _, shape, _ = _frame(pos_traj[t], mesh, p0)
        pc = PolyCollection(polys, array=shape, cmap="coolwarm", norm=norm,
                            edgecolors=(1, 1, 1, 0.35), linewidths=0.5)
        ax.add_collection(pc)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
        ax.text(0.02, 0.98, f"{name}\nt={t} ({int(100*t/max(T-1,1))}%)\np0={p0}\n"
                            f"shape index  blue<3.81<red",
                transform=ax.transAxes, color="white", fontsize=7, va="top", family="monospace")

    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.15, 0.5, 1.0)]
    sfig, sax = plt.subplots(1, len(picks), figsize=(len(picks) * PANEL, PANEL)); sfig.patch.set_facecolor("black")
    for a, t in zip(sax, picks):
        draw(a, t)
    sfig.subplots_adjust(0.005, 0.005, 0.995, 0.995, wspace=0.03)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)

    fig, ax = plt.subplots(figsize=(PANEL, PANEL)); fig.patch.set_facecolor("black"); fig.subplots_adjust(0, 0, 1, 1)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=120):
        for t in idx:
            draw(ax, t); w.grab_frame()
    plt.close(fig)


def diagnostics(pos_traj, mesh, p0):
    interior = ~np.isin(np.arange(mesh["nF"]), _border_faces(mesh))
    def stats(pos):
        _, area, perim, shape, energy = _frame(pos, mesh, p0)
        return (float(np.mean(np.abs(area[interior] - mesh["a0"]))),
                float(np.nanmean(shape[interior])), float(np.mean(energy[interior])))
    ae0, q0, e0 = stats(pos_traj[0]); aeT, qT, eT = stats(pos_traj[-1])
    return dict(p0=p0, area_err_start=round(ae0, 4), area_err_end=round(aeT, 4),
                shape_start=round(q0, 4), shape_relaxed=round(qT, 4),
                energy_start=round(e0, 5), energy_relaxed=round(eT, 6),
                relaxed=bool(eT < e0 * 0.5))


def _border_faces(mesh):
    """Faces touching a pinned vertex (they can't fully relax -> excluded from bulk stats)."""
    pin = mesh["pin"]; ef = mesh["E_face"]; es = mesh["E_srce"]
    touch = np.unique(ef[pin[es]])
    return touch


def run_all(only=None):
    mesh = _mesh_dict()
    recs = []
    for name, p0 in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        print(f"[tyssue2d] {name}: p0={p0}  (Nv={mesh['Nv']}, F={mesh['nF']})", flush=True)
        rec = {"name": name, "p0": p0, "Nv": mesh["Nv"], "nF": mesh["nF"]}
        try:
            sim, cfg = make_spec(name, p0, mesh)
            write_spec(cfg, os.path.join(odir, "spec.yaml"))
            _, out = engine_run(sim, device="cpu")
            pos = out["sets"]["vertex"]["pos"][:, :mesh["Nv"], :]        # [T,Nv,2]
            diag = diagnostics(pos, mesh, p0); rec.update(diag)
            keep = np.linspace(0, pos.shape[0] - 1, min(pos.shape[0], 120)).astype(int)
            np.savez_compressed(os.path.join(odir, "traj.npz"), pos=pos[keep].astype("float32"),
                                p0=p0, es=mesh["E_srce"], et=mesh["E_trgt"], ef=mesh["E_face"],
                                nF=mesh["nF"], a0=mesh["a0"], verts0=mesh["verts0"], pin=mesh["pin"])
            render(pos[keep], mesh, odir, name, p0)
            print(f"           -> relaxed={diag['relaxed']}  E {diag['energy_start']}->{diag['energy_relaxed']}"
                  f"  <q> {diag['shape_start']}->{diag['shape_relaxed']}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)
        recs.append(rec)
    _transition_curve(recs)


def rerender(only=None):
    for name, p0 in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); tf = os.path.join(odir, "traj.npz")
        if not os.path.exists(tf):
            print(f"[rerender] {name}: no traj.npz"); continue
        d = np.load(tf)
        mesh = dict(E_srce=d["es"], E_trgt=d["et"], E_face=d["ef"], nF=int(d["nF"]),
                    Nv=d["pos"].shape[1], a0=float(d["a0"]), verts0=d["verts0"], pin=d["pin"])
        render(d["pos"], mesh, odir, name, float(d["p0"]))
        print(f"[rerender] {name}", flush=True)


def _transition_curve(recs=None):
    if recs is None:
        recs = [json.load(open(os.path.join(OUT, n, "diag.json"))) for n, _ in presets()
                if os.path.exists(os.path.join(OUT, n, "diag.json"))]
    recs = [r for r in recs if "energy_relaxed" in r]
    if not recs:
        return
    recs.sort(key=lambda r: r["p0"])
    p0s = [r["p0"] for r in recs]
    E = [r["energy_relaxed"] for r in recs]; q = [r["shape_relaxed"] for r in recs]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6)); fig.patch.set_facecolor("black")
    for a in ax:
        a.set_facecolor("black"); a.tick_params(colors="white")
        for s in a.spines.values():
            s.set_color("white")
        a.axvline(3.81, color="0.6", ls="--", lw=1)
    ax[0].plot(p0s, E, "o-", color="tab:cyan"); ax[0].set_yscale("log")
    ax[0].set_xlabel("target shape index p0", color="white"); ax[0].set_ylabel("residual energy / cell", color="white")
    ax[1].plot(p0s, q, "o-", color="tab:orange"); ax[1].plot(p0s, p0s, ":", color="0.5")
    ax[1].set_xlabel("target shape index p0", color="white"); ax[1].set_ylabel("relaxed <q> = P/sqrt(A)", color="white")
    ax[0].set_title("rigidity transition (AVM)", color="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "_transition_tyssue2d.png"), dpi=120, facecolor="black"); plt.close(fig)
    print(f"[tyssue2d] transition curve -> {OUT}/_transition_tyssue2d.png")


def montage():
    from PIL import Image
    names = [p[0] for p in presets()]
    strips = [(n, Image.open(os.path.join(OUT, n, "strip.png")).convert("RGB"))
              for n in names if os.path.exists(os.path.join(OUT, n, "strip.png"))]
    if strips:
        w = max(i.width for _, i in strips); h = sum(i.height for _, i in strips)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for _, im in strips:
            sheet.paste(im, (0, y)); y += im.height
        sheet.save(os.path.join(OUT, "_montage_tyssue2d.png"))
    _transition_curve()
    print(f"[tyssue2d] montage -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--rerender", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.montage:
        montage()
    elif a.rerender:
        rerender(a.only); montage()
    else:
        run_all(a.only); montage()


if __name__ == "__main__":
    main()
