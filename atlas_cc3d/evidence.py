"""evidence -- one reference run per CompuCell3D mechanism, each beside its ablation.

Writes `log/atlas_cc3d/<mechanism>/` in the same shape as `log/atlas_jax/<name>/` -- `strip.png`,
`movie.mp4`, `metrics.png`, `metrics.json`, `_provenance.json` -- so the two campaigns can be read
side by side.

WHAT THESE ARE, AND ARE NOT. In the jax-morph atlas an evidence folder holds a **Plexus** run --
our operators, our engine -- scored against the reference. Nothing here is a Plexus run: Phases 1-4
have not started. Producing the same folder shape while implying otherwise would be the most
misleading thing this campaign could do, so every artefact says REFERENCE on its face.

EVERY MECHANISM IS RUN TWICE: once as itself, once with the one parameter that makes it a
mechanism switched off. A single run shows that something happened; only the pair shows that this
mechanism caused it. The first atlas learned that the hard way -- in its ablated loop run every
operator still "acted" and the acted ledger was unchanged while the coupling was gone.

WHY THE LATTICE IS THE PICTURE. A Potts cell is a set of lattice sites, not a point with a radius;
drawing centroids would discard the thing that makes this target worth doing.

    python evidence.py --all
    python evidence.py --name chemotaxis --steps 3000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

import demos

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
LOG = os.path.join(PLEXUS, "log", "atlas_cc3d")
ENV = os.environ.get("CC3D_ENV", "/workspace/.conda_envs/cc3d-oracle")
PY = os.path.join(ENV, "bin", "python")
RUN_SCRIPT = os.path.join(ENV, "lib", "python3.12", "site-packages", "cc3d", "run_script.py")

BG = "black"
C_ON, C_OFF = "#4FA3FF", "#FF6B6B"          # mechanism on / ablated: two distinct sources
TYPE_RGB = {0: (0.0, 0.0, 0.0), 1: (0.31, 0.64, 1.00), 2: (1.00, 0.42, 0.42),
            3: (1.00, 0.82, 0.40), 4: (0.61, 0.42, 0.87)}

# Records the trajectory: the cell-type lattice plus per-cell aggregates every `stride` MCS. The
# derived observables are computed in the renderer with numpy -- a python double loop over the
# lattice inside the steppable made the run several times slower for no extra information.
RECORDER = '''
import os, json
import numpy as np
from cc3d.core.PySteppables import *

class RecordSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.frames, self.ids, self.series = [], [], []

    def step(self, mcs):
        dim = self.dim
        lat = np.zeros((dim.x, dim.y), dtype=np.uint8)
        ids = np.zeros((dim.x, dim.y), dtype=np.uint16)
        for x in range(dim.x):
            for y in range(dim.y):
                c = self.cell_field[x, y, 0]
                if c is not None:
                    lat[x, y] = int(c.type)
                    ids[x, y] = int(c.id)
        self.frames.append(lat)
        self.ids.append(ids)
        vols = [float(c.volume) for c in self.cell_list]
        srfs = [float(c.surface) for c in self.cell_list]
        xs = [float(c.xCOM) for c in self.cell_list]
        self.series.append({"mcs": int(mcs), "n": len(vols),
                            "volume_mean": float(np.mean(vols)) if vols else 0.0,
                            "surface_mean": float(np.mean(srfs)) if srfs else 0.0,
                            "com_x_mean": float(np.mean(xs)) if xs else 0.0})

    def finish(self):
        out = os.environ["CC3D_OUT"]
        np.savez_compressed(os.path.join(out, "frames.npz"), lattice=np.stack(self.frames),
                            ids=np.stack(self.ids))
        json.dump({"series": self.series}, open(os.path.join(out, "series.json"), "w"), indent=1)
'''

MAIN_TMPL = '''
import os
from cc3d import CompuCellSetup
from recSteppables import RecordSteppable
{extra_import}
CompuCellSetup.register_steppable(steppable=RecordSteppable(
    frequency=int(os.environ.get("CC3D_STRIDE", "50"))))
{extra_register}
CompuCellSetup.run()
'''

PROJECT = '''<Simulation version="4.10.0">
   <XMLScript Type="XMLScript">Simulation/model.xml</XMLScript>
   <PythonScript Type="PythonScript">Simulation/main.py</PythonScript>
</Simulation>
'''


def run_one(name, arm, seed, steps, dim, stride):
    """One arm (`on` or `off`) of one mechanism."""
    control = (arm == "off")
    out = os.path.join(LOG, name, arm)
    proj = os.path.join(HERE, "_oracle", "_evidence", f"{name}_{arm}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(proj, "Simulation"), exist_ok=True)

    step_src = demos.build_steppable(name, control)
    open(os.path.join(proj, "model.cc3d"), "w").write(PROJECT)
    open(os.path.join(proj, "Simulation", "recSteppables.py"), "w").write(RECORDER)
    if step_src:
        open(os.path.join(proj, "Simulation", "extraSteppables.py"), "w").write(step_src)
    open(os.path.join(proj, "Simulation", "main.py"), "w").write(MAIN_TMPL.format(
        extra_import="from extraSteppables import GrowDivide" if step_src else "",
        extra_register=("CompuCellSetup.register_steppable(steppable=GrowDivide(frequency=1))"
                        if step_src else "")))

    gen = os.path.join(proj, "_gen.py")
    open(gen, "w").write(demos.build_source(name, control))
    r = subprocess.run([PY, "-u", gen, str(seed), str(steps), str(dim)],
                       capture_output=True, text=True, timeout=600)
    if "<CompuCell3D" not in r.stdout:
        raise RuntimeError(f"{name}/{arm}: CC3DML generation failed\n{r.stderr[-1500:]}")
    open(os.path.join(proj, "Simulation", "model.xml"), "w").write(r.stdout)

    frames = os.path.join(out, "frames.npz")
    if os.path.exists(frames):
        os.remove(frames)                    # never read a stale run as if it were this one
    env = dict(os.environ, CC3D_OUT=out, CC3D_STRIDE=str(stride))
    rr = subprocess.run([PY, "-u", RUN_SCRIPT, "-i", os.path.join(proj, "model.cc3d"),
                         "-o", os.path.join(HERE, "_oracle", "_out", f"{name}_{arm}"),
                         f"--current-dir={proj}"],
                        capture_output=True, text=True, timeout=5400, env=env)
    if not os.path.exists(frames):
        raise RuntimeError(f"{name}/{arm}: no frames\n{rr.stdout[-1500:]}\n{rr.stderr[-1500:]}")
    return out


def derived(out):
    """Observables computed from the lattice in numpy: heterotypic boundary length."""
    z = np.load(os.path.join(out, "frames.npz"))
    lat, ids = z["lattice"], (z["ids"] if "ids" in z.files else None)
    ser = json.load(open(os.path.join(out, "series.json")))["series"]
    for t, s in enumerate(ser):
        a = lat[t]
        het = 0
        u, v = a[1:, :], a[:-1, :]
        het += int(np.sum((u != v) & (u > 0) & (v > 0)))
        u, v = a[:, 1:], a[:, :-1]
        het += int(np.sum((u != v) & (u > 0) & (v > 0)))
        s["heterotypic_boundary"] = het
    return lat, ser, ids


def rgb(lat, ids=None):
    """Type colour, with cell borders drawn from the ID lattice.

    Colouring by type alone renders a one-type proliferating tissue as a solid block -- the cells,
    which are the entire subject of a Potts model, become invisible. Borders make the tessellation
    legible and cost nothing.
    """
    img = np.zeros(lat.shape + (3,), np.float32)
    for t, c in TYPE_RGB.items():
        img[lat == t] = c
    if ids is not None:
        b = np.zeros(lat.shape, bool)
        b[1:, :] |= ids[1:, :] != ids[:-1, :]
        b[:, 1:] |= ids[:, 1:] != ids[:, :-1]
        img[b & (lat > 0)] *= 0.45          # darken the seam, keep the type hue readable
    return img


def _pct(a, b):
    return None if not a else round(100.0 * (b - a) / abs(a), 1)


def render(name, seed, steps, dim, stride):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    sys.path.insert(0, os.path.join(PLEXUS, "src"))
    from plexus.plot import _ffmpeg

    d = demos.DEMOS[name]
    base = os.path.join(LOG, name)
    lat_on, ser_on, ids_on = derived(os.path.join(base, "on"))
    lat_off, ser_off, ids_off = derived(os.path.join(base, "off"))
    T = min(len(ser_on), len(ser_off))
    key, n_panels = d["headline"], 6
    picks = np.linspace(0, T - 1, n_panels).astype(int)

    # ---- strip: mechanism on (top) and ablated (bottom), matched frames ---------------------- #
    fig, axes = plt.subplots(2, n_panels, figsize=(2.35 * n_panels, 5.5), facecolor=BG)
    for row, (lat, ser, ids, lab, col) in enumerate(
            ((lat_on, ser_on, ids_on, "mechanism ON", C_ON),
             (lat_off, ser_off, ids_off, "ABLATED", C_OFF))):
        for j, t in enumerate(picks):
            ax = axes[row, j]
            ax.imshow(np.transpose(rgb(lat[t], None if ids is None else ids[t]), (1, 0, 2)),
                      origin="lower", interpolation="nearest")
            ax.set_facecolor(BG), ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#444444")
            ax.text(0.03, 0.97, f"MCS {ser[t]['mcs']}", transform=ax.transAxes, color="white",
                    fontsize=9, va="top", ha="left")
            if j == 0:
                ax.text(0.03, 0.05, lab, transform=ax.transAxes, color=col, fontsize=11,
                        va="bottom", ha="left", weight="bold")
    fig.text(0.005, 0.985,
             f"CompuCell3D · {name} — REFERENCE run (not a Plexus run) · {d['desc']}",
             color="white", fontsize=11, va="top", ha="left")
    fig.text(0.005, 0.955, f"ablation: {d['control_desc']}",
             color="#AAAAAA", fontsize=9.5, va="top", ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(base, "strip.png"), dpi=135, facecolor=BG)
    plt.close(fig)

    # ---- metrics: the headline observable, both arms ----------------------------------------- #
    mcs = [s["mcs"] for s in ser_on[:T]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3), facecolor=BG)
    a1.plot(mcs, [s[key] for s in ser_on[:T]], color=C_ON, lw=2.4, label="mechanism ON")
    a1.plot(mcs, [s[key] for s in ser_off[:T]], color=C_OFF, lw=2.4, label="ablated")
    a1.text(0.03, 0.96, d["ylabel"], transform=a1.transAxes, color="white", fontsize=11,
            va="top", ha="left")
    a2.plot(mcs, [s["n"] for s in ser_on[:T]], color=C_ON, lw=2.0, label="ON")
    a2.plot(mcs, [s["n"] for s in ser_off[:T]], color=C_OFF, lw=2.0, label="ablated")
    a2.text(0.03, 0.96, "live cells (sanity: both arms really ran)", transform=a2.transAxes,
            color="white", fontsize=11, va="top", ha="left")
    for ax in (a1, a2):
        ax.set_facecolor(BG), ax.set_xlabel("MCS", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#444444")
        leg = ax.legend(loc="center right", fontsize=9, facecolor=BG, edgecolor="#444444")
        for t in leg.get_texts():
            t.set_color("white")
    fig.tight_layout()
    fig.savefig(os.path.join(base, "metrics.png"), dpi=135, facecolor=BG)
    plt.close(fig)

    # ---- movie: the mechanism running --------------------------------------------------------- #
    ff = _ffmpeg()
    if ff is not None:
        matplotlib.rcParams["animation.ffmpeg_path"] = ff
        fig, ax = plt.subplots(figsize=(5.2, 5.4), facecolor=BG)
        ax.set_facecolor(BG), ax.set_xticks([]), ax.set_yticks([])
        im = ax.imshow(np.transpose(rgb(lat_on[0], None if ids_on is None else ids_on[0]),
                                    (1, 0, 2)), origin="lower", interpolation="nearest")
        lab = ax.text(0.03, 0.97, "", transform=ax.transAxes, color="white", fontsize=11,
                      va="top", ha="left")
        fig.tight_layout()
        w = FFMpegWriter(fps=10, codec="h264", bitrate=2600)
        with w.saving(fig, os.path.join(base, "movie.mp4"), dpi=130):
            for t in range(len(ser_on)):
                im.set_data(np.transpose(rgb(lat_on[t], None if ids_on is None else ids_on[t]),
                                          (1, 0, 2)))
                lab.set_text(f"{name}  MCS {ser_on[t]['mcs']}   n={ser_on[t]['n']}")
                w.grab_frame(facecolor=BG)
        plt.close(fig)

    # ---- the ablation, as a number ------------------------------------------------------------ #
    on0, on1 = ser_on[0][key], ser_on[T - 1][key]
    of0, of1 = ser_off[0][key], ser_off[T - 1][key]
    verdict = {"mechanism": name, "observable": key, "description": d["desc"],
               "ablation": d["control_desc"],
               "on": {"first": on0, "last": on1, "change_pct": _pct(on0, on1)},
               "off": {"first": of0, "last": of1, "change_pct": _pct(of0, of1)},
               "params": d["params"], "control_params": d["control"],
               "seed": seed, "steps": steps, "dim": dim, "stride": stride,
               "cc3d_version": "4.10.0", "source": "REFERENCE (CompuCell3D)",
               "note": "not a Plexus run -- Phases 1-4 have not started"}
    json.dump({"verdict": verdict, "on": ser_on, "off": ser_off},
              open(os.path.join(base, "metrics.json"), "w"), indent=1)
    json.dump(verdict, open(os.path.join(base, "_provenance.json"), "w"), indent=1)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, choices=sorted(demos.DEMOS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--dim", type=int, default=70)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args()

    names = sorted(demos.DEMOS) if a.all else ([a.name] if a.name else [])
    if not names:
        ap.error("give --name or --all")
    rows, failed = [], []
    for n in names:
        d = demos.DEMOS[n]
        steps, dim = d.get("steps", a.steps), d.get("dim", a.dim)
        try:
            if not a.render_only:
                for arm in ("on", "off"):
                    run_one(n, arm, a.seed, steps, dim, a.stride)
            v = render(n, a.seed, steps, dim, a.stride)
        except Exception as e:
            failed.append((n, f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"  FAIL {n}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        rows.append(v)
        print(f"  {n:<20} {v['observable']:<22} ON {v['on']['first']:.4g}->{v['on']['last']:.4g}"
              f" ({v['on']['change_pct']:+}%)   ABLATED {v['off']['first']:.4g}->"
              f"{v['off']['last']:.4g} ({v['off']['change_pct']:+}%)", flush=True)
    if rows:
        json.dump(rows, open(os.path.join(LOG, "_ablations.json"), "w"), indent=1)
    for n, why in failed:
        print(f"  NOT PRODUCED  {n}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
