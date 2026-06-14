"""Scenario archive (Q1): turn a scenario into a reproducible, browsable record.

For each scenario it: validates -> runs -> measures generic metrics ->
renders a first/last thumbnail (and optional gif) -> writes
  scenarios/archive/<name>/{scenario.yaml, metrics.json, thumb.png[, run.gif]}
and refreshes scenarios/GALLERY.md (an index of everything ever archived).

The archive key is the scenario `name`; re-archiving overwrites that entry, so
the gallery is always the current truth. Metrics + the exact spec + seed make
each entry reproducible.

    python archive.py scenarios/*.yaml            # archive many
    python archive.py scenarios/sort.yaml --gif   # also render a gif
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scenario_schema import load
import engine2

ARCHIVE = os.path.join(os.path.dirname(__file__), "scenarios", "archive")
GALLERY = os.path.join(os.path.dirname(__file__), "scenarios", "GALLERY.md")
PALETTE = ["#e8483c", "#2f6fdb", "#2ca02c", "#ff7f0e"]


def _aspect(pp_fr, par, idx):
    out = []
    for c in idx:
        p = pp_fr[par == c]
        if len(p) < 3:
            continue
        w = np.linalg.eigvalsh(np.cov((p - p.mean(0)).T))
        out.append(float(np.sqrt(max(w) / max(min(w), 1e-12))))
    return float(np.mean(out)) if out else 1.0


def _nnd(p):
    if len(p) < 2:
        return 0.0
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def metrics(a):
    cp, pp, par = a["cell_pos"], a["particle_pos"], a["parent"]
    t, names = a["cell_type"], a["type_names"]
    m = {"finite": bool(np.isfinite(pp).all()), "frames": int(cp.shape[0]),
         "n_cells": int(cp.shape[1]), "n_particles": int(pp.shape[1]), "types": {}}
    step = np.linalg.norm(np.diff(cp, axis=0), axis=2)        # [T-1, Nc]
    for tid, nm in enumerate(names):
        sel = t == tid
        idx = np.where(sel)[0]
        m["types"][nm] = {
            "path": float(step[:, sel].sum(0).mean()),
            "aspect_0": _aspect(pp[0], par, idx), "aspect_T": _aspect(pp[-1], par, idx),
            "nnd_0": _nnd(cp[0][sel]), "nnd_T": _nnd(cp[-1][sel]),
        }
    return m


def thumb(a, path):
    cp, pp, par = a["cell_pos"], a["particle_pos"], a["parent"]
    pt = a["cell_type"][par]
    fld = a["field"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    vmax = max(fld.max(), 1e-6)
    for j, fr in enumerate([0, -1]):
        ax[j].imshow(fld[fr].T, origin="lower", extent=[0, 1, 0, 1],
                     cmap="Greens", vmin=0, vmax=vmax, alpha=0.8)
        for tid in range(int(pt.max()) + 1):
            mk = pt == tid
            ax[j].scatter(pp[fr][mk, 0], pp[fr][mk, 1], s=0.6, c=PALETTE[tid % 4])
        ax[j].set_xticks([]); ax[j].set_yticks([])
        ax[j].set_title("start" if fr == 0 else "end", fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=95); plt.close(fig)


def archive_one(scenario_path, device, make_gif=False):
    sc = load(scenario_path)
    _, a = engine2.run(sc, "/tmp/arch.zarr", device=device)
    d = os.path.join(ARCHIVE, sc.name)
    os.makedirs(d, exist_ok=True)
    shutil.copy(scenario_path, os.path.join(d, "scenario.yaml"))
    m = metrics(a); m["seed"] = sc.seed
    json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=2)
    thumb(a, os.path.join(d, "thumb.png"))
    if make_gif:
        import zarr
        zarr.save_group("/tmp/arch.zarr")  # already written by run
        os.system(f"PYTHONPATH={os.environ.get('PYTHONPATH','')} {sys.executable} "
                  f"{os.path.join(os.path.dirname(__file__),'viz2.py')} /tmp/arch.zarr "
                  f"{os.path.join(d,'run.gif')} --dot 0.5 >/dev/null 2>&1")
    return sc.name, m


def refresh_gallery():
    rows = []
    for name in sorted(os.listdir(ARCHIVE)):
        mp = os.path.join(ARCHIVE, name, "metrics.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        # headline = the type that moved/changed most
        tinfo = "; ".join(
            f"{nm}: path {d['path']:.2f}, aspect {d['aspect_0']:.2f}->{d['aspect_T']:.2f}, "
            f"nnd {d['nnd_0']:.3f}->{d['nnd_T']:.3f}"
            for nm, d in m["types"].items())
        gif = f" · [gif](archive/{name}/run.gif)" if os.path.exists(os.path.join(ARCHIVE, name, "run.gif")) else ""
        rows.append(f"### {name}\n\n![{name}](archive/{name}/thumb.png)\n\n"
                    f"{m['n_cells']} cells × {m['n_particles']//max(m['n_cells'],1)} particles, "
                    f"{m['frames']} frames, finite={m['finite']}. "
                    f"[spec](archive/{name}/scenario.yaml){gif}\n\n- {tinfo}\n")
    header = ("# Scenario gallery\n\nEvery archived scenario: thumbnail (start→end), "
              "exact spec, seed, and generic metrics (path = mean per-cell distance "
              "travelled; aspect = cell elongation; nnd = nearest-neighbour distance / "
              "clustering). Regenerate with `python archive.py scenarios/*.yaml`.\n\n")
    open(GALLERY, "w").write(header + "\n".join(rows))


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    make_gif = "--gif" in sys.argv
    device = "cpu" if "--cpu" in sys.argv else ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ARCHIVE, exist_ok=True)
    for p in paths:
        name, m = archive_one(p, device, make_gif)
        print(f"[archived] {name}: finite={m['finite']}, "
              + ", ".join(f"{nm} path {d['path']:.2f}" for nm, d in m["types"].items()))
    refresh_gallery()
    print(f"[gallery] {GALLERY}")


if __name__ == "__main__":
    main()
