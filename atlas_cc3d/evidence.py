"""evidence -- render a CompuCell3D reference run into the atlas's standard evidence folder.

Same shape as `log/atlas_jax/<name>/` so the two campaigns can be read side by side:
`strip.png`, `movie.mp4`, `metrics.png`, `metrics.json`, `_provenance.json`.

WHAT THESE ARTEFACTS ARE, and are not. In the jax-morph atlas an evidence folder holds a
**Plexus** run -- our operators, our engine -- scored against the reference. Nothing here is a
Plexus run yet: Phases 1-4 have not started, and pretending otherwise by producing the same folder
shape would be the most misleading thing this campaign could do. These are the **reference**
runs, produced to prove the oracle is real, readable and reproducible, and to show what the
observable even looks like before anyone tries to reproduce it.

WHY THE LATTICE IS THE PICTURE. jax-morph's strip was a scatter of cells at their true radius,
because a cell there IS a point with a radius. A Potts cell is a *set of lattice sites*; drawing
its centroid would throw away the thing that makes this target interesting and would quietly
suggest the two frameworks share a representation. So the strip and movie render the cell-type
field over the lattice, which is what the model actually computes.

    python evidence.py --name cell_sorting --steps 4000 --dim 80
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
LOG = os.path.join(PLEXUS, "log", "atlas_cc3d")
ENV = os.environ.get("CC3D_ENV", "/workspace/.conda_envs/cc3d-oracle")
PY = os.path.join(ENV, "bin", "python")
RUN_SCRIPT = os.path.join(ENV, "lib", "python3.12", "site-packages", "cc3d", "run_script.py")

BG = "black"
# two cell types, two distinct SOURCES of colour -- red/blue, per the campaign's plot convention.
# Medium is left black so the cluster reads against the lattice.
TYPE_RGB = {0: (0.0, 0.0, 0.0), 1: (0.31, 0.64, 1.00), 2: (1.00, 0.42, 0.42)}

# Records the whole trajectory, not just the end: the cell-type lattice every `stride` MCS plus
# the per-cell table, so a strip, a movie and a metric series all come from ONE run.
STEPPABLE = '''
import os, json
import numpy as np
from cc3d.core.PySteppables import *

class RecordSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.frames, self.series = [], []

    def step(self, mcs):
        dim = self.dim
        lat = np.zeros((dim.x, dim.y), dtype=np.uint8)
        for x in range(dim.x):
            for y in range(dim.y):
                c = self.cell_field[x, y, 0]
                lat[x, y] = 0 if c is None else int(c.type)
        self.frames.append(lat)
        vols = [float(c.volume) for c in self.cell_list]
        srfs = [float(c.surface) for c in self.cell_list]
        # boundary length between the two types: the observable cell sorting is ABOUT
        het = 0
        for x in range(dim.x - 1):
            for y in range(dim.y - 1):
                a = self.cell_field[x, y, 0]
                for nb in (self.cell_field[x + 1, y, 0], self.cell_field[x, y + 1, 0]):
                    ta = 0 if a is None else int(a.type)
                    tb = 0 if nb is None else int(nb.type)
                    if ta and tb and ta != tb:
                        het += 1
        self.series.append({"mcs": int(mcs), "n": len(vols),
                            "volume_mean": float(np.mean(vols)) if vols else 0.0,
                            "surface_mean": float(np.mean(srfs)) if srfs else 0.0,
                            "heterotypic_boundary": int(het)})

    def finish(self):
        out = os.environ["CC3D_OUT"]
        np.savez_compressed(out + "/frames.npz", lattice=np.stack(self.frames))
        rows = sorted((int(c.id), int(c.type), float(c.volume), float(c.surface),
                       round(float(c.xCOM), 6), round(float(c.yCOM), 6)) for c in self.cell_list)
        json.dump({"series": self.series, "final_cells": rows,
                   "columns": ["id", "type", "volume", "surface", "xCOM", "yCOM"]},
                  open(out + "/metrics.json", "w"), indent=1)
'''

MAIN = '''
from cc3d import CompuCellSetup
from recSteppables import RecordSteppable
import os
CompuCellSetup.register_steppable(steppable=RecordSteppable(
    frequency=int(os.environ.get("CC3D_STRIDE", "50"))))
CompuCellSetup.run()
'''

PROJECT = '''<Simulation version="4.10.0">
   <XMLScript Type="XMLScript">Simulation/model.xml</XMLScript>
   <PythonScript Type="PythonScript">Simulation/main.py</PythonScript>
</Simulation>
'''

SPECS = '''
import warnings, sys; warnings.filterwarnings("ignore")
from cc3d.core.PyCoreSpecs import (PottsCore, CellTypePlugin, VolumePlugin, ContactPlugin,
                                   BlobInitializer, CenterOfMassPlugin, SurfaceTrackerPlugin)
seed, steps, dim = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
e_cc = float(sys.argv[4])   # Condensing-Condensing contact energy; the sorting knob
potts = PottsCore(dim_x=dim, dim_y=dim, dim_z=1, steps=steps, fluctuation_amplitude=10.0,
                  neighbor_order=2, random_seed=seed)
ct = CellTypePlugin("Condensing", "NonCondensing")
vol = VolumePlugin()
vol.param_new("Condensing", target_volume=25, lambda_volume=2.0)
vol.param_new("NonCondensing", target_volume=25, lambda_volume=2.0)
con = ContactPlugin(neighbor_order=2)
con.param_new("Medium", "Condensing", 16); con.param_new("Medium", "NonCondensing", 16)
con.param_new("Condensing", "Condensing", e_cc); con.param_new("NonCondensing", "NonCondensing", 11)
con.param_new("Condensing", "NonCondensing", 11)
com = CenterOfMassPlugin()
# without this, `cell.surface` silently reads 0 for every cell -- a flat zero line that looks
# like a measurement. Perimeter is a real Potts observable; make it real rather than plot a fake.
srf = SurfaceTrackerPlugin()
blob = BlobInitializer()
blob.region_new(width=5, radius=dim // 3, center=(dim // 2, dim // 2, 0),
                cell_types=("Condensing", "NonCondensing"))
body = "\\n".join(s.xml.getCC3DXMLElementString() for s in (potts, ct, vol, con, com, srf, blob))
sys.stdout.write('<CompuCell3D Revision="0" Version="4.10.0">\\n' + body + '\\n</CompuCell3D>\\n')
'''


def run(name, seed, steps, dim, stride, e_cc):
    out = os.path.join(LOG, name)
    proj = os.path.join(HERE, "_oracle", "_evidence", name)
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(proj, "Simulation"), exist_ok=True)
    open(os.path.join(proj, "model.cc3d"), "w").write(PROJECT)
    open(os.path.join(proj, "Simulation", "recSteppables.py"), "w").write(STEPPABLE)
    open(os.path.join(proj, "Simulation", "main.py"), "w").write(MAIN)
    gen = os.path.join(proj, "_gen.py")
    open(gen, "w").write(SPECS)
    r = subprocess.run([PY, "-u", gen, str(seed), str(steps), str(dim), str(e_cc)],
                       capture_output=True, text=True, timeout=600)
    if "<CompuCell3D" not in r.stdout:
        raise RuntimeError(f"CC3DML generation failed:\n{r.stderr[-1000:]}")
    open(os.path.join(proj, "Simulation", "model.xml"), "w").write(r.stdout)

    env = dict(os.environ, CC3D_OUT=out, CC3D_STRIDE=str(stride))
    rr = subprocess.run([PY, "-u", RUN_SCRIPT, "-i", os.path.join(proj, "model.cc3d"),
                         "-o", os.path.join(HERE, "_oracle", "_out", name),
                         f"--current-dir={proj}"],
                        capture_output=True, text=True, timeout=3600, env=env)
    if not os.path.exists(os.path.join(out, "frames.npz")):
        raise RuntimeError(f"no frames written\n{rr.stdout[-1200:]}\n{rr.stderr[-1200:]}")
    json.dump({"model": "cell_sorting", "seed": seed, "steps": steps, "dim": dim,
               "stride": stride, "contact_CC": e_cc, "cc3d_version": "4.10.0",
               "source": "REFERENCE (CompuCell3D)",
               "note": "not a Plexus run -- Phases 1-4 have not started"},
              open(os.path.join(out, "_provenance.json"), "w"), indent=1)
    return out


def rgb(lat):
    img = np.zeros(lat.shape + (3,), np.float32)
    for t, c in TYPE_RGB.items():
        img[lat == t] = c
    return img


def render(out, n_panels=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    sys.path.insert(0, os.path.join(PLEXUS, "src"))
    from plexus.plot import _ffmpeg

    lat = np.load(os.path.join(out, "frames.npz"))["lattice"]
    met = json.load(open(os.path.join(out, "metrics.json")))
    ser = met["series"]
    T = lat.shape[0]

    # ---- strip: one scale, panels evenly across the run ---------------------------------- #
    picks = np.linspace(0, T - 1, n_panels).astype(int)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.4 * n_panels, 2.8), facecolor=BG)
    for ax, t in zip(np.atleast_1d(axes), picks):
        ax.imshow(np.transpose(rgb(lat[t]), (1, 0, 2)), origin="lower", interpolation="nearest")
        ax.set_facecolor(BG), ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#444444")
        ax.text(0.03, 0.97, f"MCS {ser[t]['mcs']}", transform=ax.transAxes, color="white",
                fontsize=9, va="top", ha="left")
    prov = json.load(open(os.path.join(out, "_provenance.json")))
    tag = ("ABLATION: equal contact energies, no sorting expected"
           if prov.get("contact_CC", 2.0) >= 11 else "differential adhesion -> sorting")
    fig.text(0.005, 0.985,
             f"CompuCell3D cell sorting — REFERENCE run (not a Plexus run) · seed "
             f"{prov['seed']} · {tag}",
             color="white", fontsize=11, va="top", ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "strip.png"), dpi=140, facecolor=BG)
    plt.close(fig)

    # ---- metrics: what cell sorting IS, measured ------------------------------------------ #
    mcs = [s["mcs"] for s in ser]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.2), facecolor=BG)
    a1.plot(mcs, [s["heterotypic_boundary"] for s in ser], color="#FFD166", lw=2.2)
    a1.text(0.03, 0.96, "heterotypic boundary length\n(the observable sorting minimises)",
            transform=a1.transAxes, color="white", fontsize=11, va="top", ha="left")
    a2.plot(mcs, [s["volume_mean"] for s in ser], color="#4FA3FF", lw=2.2, label="volume")
    a2.plot(mcs, [s["surface_mean"] for s in ser], color="#FF6B6B", lw=2.2, label="surface")
    a2.text(0.03, 0.96, "mean cell volume and surface", transform=a2.transAxes,
            color="white", fontsize=11, va="top", ha="left")
    leg = a2.legend(loc="center right", fontsize=9, facecolor=BG, edgecolor="#444444")
    for t in leg.get_texts():
        t.set_color("white")
    for ax in (a1, a2):
        ax.set_facecolor(BG), ax.set_xlabel("MCS", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#444444")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "metrics.png"), dpi=135, facecolor=BG)
    plt.close(fig)

    # ---- movie ----------------------------------------------------------------------------- #
    ff = _ffmpeg()
    if ff is None:
        print("  (no ffmpeg -- skipping movie)")
        return
    matplotlib.rcParams["animation.ffmpeg_path"] = ff
    fig, ax = plt.subplots(figsize=(5.2, 5.4), facecolor=BG)
    ax.set_facecolor(BG), ax.set_xticks([]), ax.set_yticks([])
    im = ax.imshow(np.transpose(rgb(lat[0]), (1, 0, 2)), origin="lower", interpolation="nearest")
    lab = ax.text(0.03, 0.97, "", transform=ax.transAxes, color="white", fontsize=11,
                  va="top", ha="left")
    fig.tight_layout()
    writer = FFMpegWriter(fps=10, codec="h264", bitrate=2600)
    with writer.saving(fig, os.path.join(out, "movie.mp4"), dpi=130):
        for t in range(T):
            im.set_data(np.transpose(rgb(lat[t]), (1, 0, 2)))
            lab.set_text(f"MCS {ser[t]['mcs']}   n={ser[t]['n']}")
            writer.grab_frame(facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cell_sorting")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--dim", type=int, default=80)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--e-cc", type=float, default=2.0,
                    help="Condensing-Condensing contact energy. 2 sorts; 11 (= the heterotypic "
                         "energy) is the ABLATION: no adhesion difference, so no sorting.")
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args()
    out = os.path.join(LOG, a.name)
    if not a.render_only:
        out = run(a.name, a.seed, a.steps, a.dim, a.stride, a.e_cc)
        print(f"[evidence] ran -> {os.path.relpath(out, PLEXUS)}")
    render(out)
    print(f"[evidence] {sorted(os.listdir(out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
