"""Exhaustive microswimmer test suite (the analogue of prototype/water).

Every test is the SAME registry + engine over a different spec dict; each writes a
gif/montage and an intent metric. Five families, mirroring the paper's figures:

  flow_*        sessile / motile-body / motile-lab analytic flow fields (streamlines)
  pe_*          Peclet sweep 0..200 -> boundary-layer thinning + Sherwood rises
  cover_*       mouth-coverage sweep 5..100% -> Sherwood vs feeding area (Fig. 4)
  place_*       mouth placement front/side/rear relative to the feeding current
  swim_*        a motile cell swimming through the bath (lab frame, depleted wake)

Sherwood number Sh = uptake_rate(Pe) / uptake_rate(Pe=0): the dimensionless
feeding flux, advective enhancement over pure diffusion.

    PYTHONPATH=../../src python run_swim.py [family ...]      # default: all

Reference: Liu, Costello & Kanso, Nat Commun 16, 4154 (2025),
doi:10.1038/s41467-025-59413-x.
"""
from __future__ import annotations

import os, sys, copy, warnings, json
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swimmer_engine as E
from scenario_schema import Scenario, OpSpec, Selector
import render_swim as RS

DEV = "cuda"
try:
    import torch
    if not torch.cuda.is_available():
        DEV = "cpu"
except Exception:
    DEV = "cpu"


def spec(name, lifestyle="sessile", feeding=0.5, pe=100.0, frame="body",
         mouth_off=0.0, world=1.0, R=0.14, nframes=120, res=160, swim_gain=0.0,
         diffusion=0.06):
    """Build a validated Scenario dict programmatically (a base spec + overrides)."""
    ops = [
        {"op": "squirmer_flow", "at": "organism", "to": "flow", "frame": frame},
        {"op": "slip", "at": "surface_node"},
        {"op": "absorb", "at": "surface_node[type=mouth]", "from": "chemical", "rate": 1.0},
    ]
    sched = ["squirmer_flow", "slip", "absorb", "chemical.diffuse"]
    org_t = {lifestyle: {"fraction": 1.0, "lifestyle": lifestyle, "feeding_area": feeding,
                         "mouth_offset": mouth_off}}
    if swim_gain > 0:
        ops.insert(2, {"op": "swim", "at": f"organism[type={lifestyle}]", "gain": swim_gain})
        sched.insert(2, "swim")
    raw = {
        "name": name, "seed": 0, "n_frames": nframes, "record_every": 2, "dt": 1.0,
        "world": world,
        "sets": {
            "organism": {"n": 1, "radius": R, "n_mode": 40, "axis": 0.0, "types": org_t},
            "surface_node": {"parent": "organism", "per_parent": 240, "role": "membrane"},
        },
        "fields": {
            "flow": {"kind": "flow", "res": res, "couples_to": "organism"},
            "chemical": {"kind": "chemical", "res": res, "couples_to": "surface_node",
                         "diffusion": diffusion, "peclet": pe, "c_inf": 1.0},
        },
        "operators": ops, "schedule": sched,
    }
    return _validate(raw)


def _validate(raw):
    """Run the spec through the same typed validator the engine uses, then return a
    Scenario. Proves the microswimmer specs satisfy the contract."""
    operators = []
    from plexus.models import registry
    for o in raw["operators"]:
        registry.get_operator(o["op"])                      # raises if unregistered
        operators.append(OpSpec(op=o["op"], on=Selector.parse(o["at"]),
                                to=o.get("to"), frm=o.get("from"),
                                params={k: v for k, v in o.items()
                                        if k not in ("op", "at", "to", "from")}))
    return Scenario(name=raw["name"], seed=raw["seed"], n_frames=raw["n_frames"],
                    dt=raw["dt"], record_every=raw["record_every"], sets=raw["sets"],
                    fields=raw["fields"], operators=operators, schedule=raw["schedule"],
                    world=raw["world"])


def uptake_rate(out):
    """Steady-state uptake rate = late-time slope of cumulative uptake."""
    ut = out["uptake_t"]; h = len(ut) // 2
    return (ut[-1] - ut[h]) / max(len(ut) - h, 1)


def sherwood(lifestyle="sessile", feeding=0.5, pe=100.0, mouth_off=0.0, diffusion=0.06):
    """Sh = rate(Pe)/rate(0): advective feeding enhancement over pure diffusion."""
    base = dict(lifestyle=lifestyle, feeding=feeding, mouth_off=mouth_off, diffusion=diffusion)
    r_adv = uptake_rate(E.run(spec("_sh_adv", pe=pe, **base), DEV))
    r_dif = uptake_rate(E.run(spec("_sh_dif", pe=0.0, **base), DEV))
    return r_adv / max(r_dif, 1e-9), r_adv, r_dif


# --------------------------------------------------------------------------- #
def fam_flow(log):
    for nm, life, frame in [("flow_sessile", "sessile", "body"),
                            ("flow_motile_body", "motile", "body"),
                            ("flow_motile_lab", "motile", "lab")]:
        out = E.run(spec(nm, lifestyle=life, frame=frame), DEV)
        _save_flow(nm, out)
        log.append(f"{nm:18s} max|u|={np.hypot(*out['flow'].transpose(2,0,1)).max():.3f}")


def _save_flow(nm, out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    u = out["flow"]; x, y, W = RS._cell_axes(out)
    U = u[..., 0].T; V = u[..., 1].T; spd = np.hypot(U, V)
    cx = out["org_pos"][-1][0]; R = out["radius"]
    fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor("white")
    ax.streamplot(x, y, U, V, density=1.5, color=RS.BLUE, linewidth=0.7, arrowsize=0.7)
    ax.add_patch(Circle((cx[0], cx[1]), R, fc="0.85", ec="k", lw=1.2, zorder=3))
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 1.02, nm, transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{nm}.png", dpi=110); plt.close(fig)


def fam_pe(log):
    rows = []
    for pe in [0, 10, 30, 60, 120, 200]:
        out = E.run(spec(f"pe_{pe:03d}", pe=float(pe)), DEV)
        RS.render_conc.__wrapped__ if False else None
        _conc_gif(f"pe_{pe:03d}", out)
        rows.append((pe, uptake_rate(out), float(out["chem"].min())))
    r0 = rows[0][1]
    for pe, r, cmin in rows:
        log.append(f"pe_{pe:03d}  Sh={r/max(r0,1e-9):.3f}  rate={r:7.2f}  c_min={cmin:.3f}")
    _curve([r[0] for r in rows], [r[1] / max(r0, 1e-9) for r in rows],
           "Peclet number", "Sherwood number Sh", "sweep_peclet.png")


def fam_cover(log):
    rows = []
    for f in [0.05, 0.15, 0.30, 0.50, 0.70, 1.0]:
        sh, r_adv, r_dif = sherwood(feeding=f, pe=120.0)
        out = E.run(spec(f"cover_{int(f*100):03d}", feeding=f, pe=120.0), DEV)
        _conc_gif(f"cover_{int(f*100):03d}", out)
        rows.append((f, sh, r_adv))
        log.append(f"cover_{int(f*100):03d}  feeding={f:.2f}  Sh={sh:.3f}  rate={r_adv:7.2f}")
    _curve([r[0] * 100 for r in rows], [r[1] for r in rows],
           "mouth coverage (%)", "Sherwood number Sh", "sweep_coverage.png")
    _curve([r[0] * 100 for r in rows], [r[2] for r in rows],
           "mouth coverage (%)", "uptake rate", "sweep_coverage_rate.png")


def fam_place(log):
    import math
    for nm, off in [("place_front", 0.0), ("place_side", math.pi / 2), ("place_rear", math.pi)]:
        sh, r_adv, _ = sherwood(mouth_off=off, pe=120.0)
        out = E.run(spec(nm, mouth_off=off, pe=120.0), DEV)
        _conc_gif(nm, out)
        log.append(f"{nm:12s}  off={off:.2f}  Sh={sh:.3f}  rate={r_adv:7.2f}")


def fam_swim(log):
    for nm, g in [("swim_slow", 0.006), ("swim_fast", 0.014)]:
        out = E.run(spec(nm, lifestyle="motile", frame="lab", world=1.6, R=0.12,
                         nframes=200, res=176, pe=120.0, swim_gain=g, diffusion=0.05), DEV)
        _conc_gif(nm, out)
        disp = float(np.linalg.norm(out["org_pos"][-1][0] - out["org_pos"][0][0]))
        log.append(f"{nm:12s}  gain={g}  displacement={disp:.3f}  uptake={out['uptake']:.0f}")


# --- shared rendering helpers ---------------------------------------------- #
def _conc_gif(nm, out):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle
    from PIL import Image
    c = out["chem"]; x, y, W = RS._cell_axes(out); T = c.shape[0]
    snp = out["sn_pos"]; mouth = out["sn_type"] == 0; op = out["org_pos"]; R = out["radius"]
    fig, ax = plt.subplots(figsize=(5.2 * max(W, 1), 5.2)); fig.patch.set_facecolor("white")
    im = ax.imshow(c[0].T, origin="lower", extent=[0, W, 0, 1], cmap=RS.CMAP, vmin=0, vmax=1)
    body = Circle((op[0][0][0], op[0][0][1]), R, fc="0.85", ec="k", lw=1.0, zorder=3); ax.add_patch(body)
    cil = ax.scatter(snp[0][~mouth, 0], snp[0][~mouth, 1], s=3, c="0.4", zorder=4)
    mth = ax.scatter(snp[0][mouth, 0], snp[0][mouth, 1], s=8, c="#d62728", zorder=5)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        im.set_data(c[fr].T); body.center = (op[fr][0][0], op[fr][0][1])
        cil.set_offsets(snp[fr][~mouth]); mth.set_offsets(snp[fr][mouth])
        ax.set_title(f"{nm}  {fr}/{T-1}  uptake={out['uptake_t'][fr]:.0f}", fontsize=9)
        return [im, cil, mth, body]
    FuncAnimation(fig, upd, frames=T, blit=False).save(f"{nm}.gif", writer=PillowWriter(fps=20))
    plt.close(fig)
    g = Image.open(f"{nm}.gif"); fr = []
    try:
        while True:
            fr.append(g.copy().convert("RGB")); g.seek(g.tell() + 1)
    except EOFError:
        pass
    idx = [0, int(T * .2), int(T * .45), int(T * .7), T - 1]; sel = [fr[i] for i in idx]
    w, h = sel[0].size; m = Image.new("RGB", (w * len(sel), h), "white")
    for k, im2 in enumerate(sel):
        m.paste(im2, (k * w, 0))
    m.save(f"{nm}_montage.png")


def _curve(xs, ys, xlabel, ylabel, fname):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, ys, "o-", color=RS.BLUE, lw=2, ms=6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=120); plt.close(fig)


FAMILIES = {"flow": fam_flow, "pe": fam_pe, "cover": fam_cover, "place": fam_place, "swim": fam_swim}

if __name__ == "__main__":
    want = sys.argv[1:] or list(FAMILIES)
    log = []
    for fam in want:
        print(f"=== {fam} ===", flush=True)
        FAMILIES[fam](log)
        for line in log[-8:]:
            print("  " + line, flush=True)
    with open("results.md", "w") as fh:
        fh.write("# Microswimmer suite results\n\n")
        fh.write("Sh = uptake_rate(Pe) / uptake_rate(Pe=0)  (advective feeding enhancement)\n\n")
        fh.write("```\n" + "\n".join(log) + "\n```\n")
    print("\n[done] results.md written", flush=True)
