#!/usr/bin/env python
"""test_02_ecm_block -- the third operator test: a block of ECM fibres, dropped.

    python test_02_ecm_block.py [--frames N] [--device cuda:0]   ->  log/okuda_ECM/02_ecm_block/

WHY DROP IT. The matrix has only ever been seen doing one thing in this prototype -- being pushed
outward by a growing ball -- and that loads it in one direction, slowly, everywhere at once. Nothing in
those runs would reveal whether its fibres carry stress the way a fibrous material should, because
there was no moment at which the load arrived. A block falling under gravity and hitting a floor is the
cheapest test that has one: the impact is a front that travels up through the material, the fibres
nearest the floor take it first, and the block either rebounds (it is elastic and the stress is
transient) or slumps (it is not).

WHAT IS BEING TESTED, in Plexus2 terms: `seed_ecm` (a Seed operator laying fibres of a given length,
alignment and density), the MLS-MPM cycle as four operators around one field, `gravity` as a
parent-level body force, and `ecm_stress` as the readout. No tissue, no membrane, no adhesion -- so any
stress in the picture is the matrix's own.

WHAT IS MEASURED: the block's centre of height against time (the bounce), its kinetic energy (how much
of the drop comes back), and the stress distribution through the impact -- mean and p99 of the same
von Mises measure the spheroid runs colour by, so the numbers are comparable with them.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "prototype", "Tyssue"),
          os.path.join(_ROOT, "prototype", "eye")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import ListedColormap

import ecm_ops                                        # noqa: F401  registers seed_ecm / ecm_stress
import ecm_render as RD
import ecm_spec as ES

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

LOG = os.path.join(_ROOT, "log", "okuda_ECM")
# STROKE WIDTHS AS MODULE STATE, so a re-render can thin them without touching the
# drawing routines the run itself used.
_LW3 = [0.7]
_LW2 = [1.1]
CMAP = ListedColormap(ES.STRESS_COLORS)


def build(name, frames, n_particles=90000, n_grid=64, dt=3.2e-3, sub=2.0e-4, youngs=120.0,
          n_fibres=1500, fibre_len=0.09, align=0.0, g=2.5, drag=0.05, wall_damp=0.7,
          stress_scale=2.0e-5, cube=(0.30, 0.28, 0.30, 0.70, 0.60, 0.70), hole_r=0.07, seed=0):
    """The spec, as a plain dict -- sets, one field, operators, schedule.

    SLOW, AND CLOSE TO THE FLOOR, BECAUSE THE SUBJECT IS THE STRESS AND NOT THE FALL. Dropped from
    0.75 at g = 16 the block crosses the box in a dozen frames and the impact is over before it can be
    watched; the fibres go from black to saturated between two frames. Here the cube starts with its
    underside at 0.28 -- about a quarter of the box above the floor -- and gravity is 2.5, so the fall
    takes a third of the run and the front travelling up through the fibres takes the rest.
    AND THE STRESS IS MEANT TO STAY. A stiff, undamped block rings and the load washes out; the matrix
    is a gel that holds what it is given, so E = 120 with drag 0.05 and `wall_damp` 0.7 -- the stress
    builds as the block settles onto the floor and remains in the fibres that carry it.

    THE NUMBERS ARE OTHERWISE THE STOCK MPM DEMO'S, so this cube falls the way every other cube in Plexus falls:
    `n_grid` 64, `dt` 3.2e-3 over a 2e-4 substep, gravity 16 along -y (the operator's default axis),
    `drag` 0.05 and `wall_damp` 0.7. What differs is the material: a `block:` type-fill would give a
    cube of homogeneous points, and this is a cube of FIBRES, seeded by `seed_ecm` into the same box.

    AND IT HAS A HOLE, which nothing in this run fills. The cavity is where the spheroid goes in the
    run after this one, and it has to be empty from frame 0 -- matrix seeded where the ball will be is
    in contact before the clock starts.
    """
    types = {f"s{i}": {"fraction": 1.0 / len(ES.STRESS_COLORS), "youngs": float(youngs)}
             for i in range(len(ES.STRESS_COLORS))}
    return {
        "general": {"name": name, "seed": int(seed), "n_frames": int(frames), "dt": float(dt),
                    "boundary": "wall", "dim": 3, "world": [1.0, 1.0, 1.0]},
        "sets": {
            "cell": {"n": 1, "start": [[0.5 * (cube[0] + cube[3]), 0.5 * (cube[1] + cube[4]),
                                        0.5 * (cube[2] + cube[5])]], "types": types},
            "mpm_particle": {"parent": "cell", "per_parent": int(n_particles), "radius": 0.48,
                             "density": 1.0, "types": types},
        },
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": int(n_grid)}},
        "operators": [
            {"op": "aggregate", "at": "cell"},
            {"op": "seed_ecm", "at": "mpm_particle",
             "centre": [0.5 * (cube[0] + cube[3]), 0.5 * (cube[1] + cube[4]),
                        0.5 * (cube[2] + cube[5])],
             "block": list(cube), "cavity_r": float(hole_r), "cavity_h": float(hole_r), "axis": 1,
             "cavity_sphere": True,
             "margin": 0.02, "n_fibres": int(n_fibres), "fibre_len": float(fibre_len),
             "align": float(align), "align_dir": [1.0, 0.0, 0.0], "seed": int(seed)},
            # THE ONLY DRIVE. `gravity` is a parent-level body force: `mpm_scatter` reads the cell's
            # accumulated delta as a_ext, so one number moves every particle and nothing else is
            # imposed anywhere in the run.
            {"op": "gravity", "at": "cell", "g": float(g)},
            {"op": "ecm_stress", "at": "mpm_particle", "scale": float(stress_scale),
             "bands": len(ES.STRESS_COLORS), "measure": "vol"},
            {"op": "mpm_strain", "at": "mpm_particle"},
            {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid", "drag": float(drag),
             "a_max": 300.0},
            {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": float(wall_damp)},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid",
             "wall_damp": float(wall_damp), "wall_contact": 0.04, "vmax": 1.0e9},
        ],
        "schedule": ["aggregate", "seed_ecm", "gravity", "ecm_stress",
                     {"substep_dt": float(sub),
                      "steps": ["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"]}],
        "plotting": {"background": "black", "up_axis": 1, "box_frame": True},
    }


UP = 1          # gravity's own axis: `gravity` defaults to -y, and the stock MPM demos are y-up


def measure(P, band, vm):
    """The bounce, and the stress that produced it."""
    zc = P[:, :, UP].mean(axis=1)
    z_lo = np.percentile(P[:, :, UP], 2, axis=1)
    v = np.diff(zc, prepend=zc[0])
    ke = 0.5 * (np.diff(P, axis=0, prepend=P[:1]) ** 2).sum(axis=2).mean(axis=1)
    s_mean = np.asarray([float(np.mean(a)) for a in vm]) if vm is not None else np.zeros(len(P))
    s_p99 = np.asarray([float(np.percentile(a, 99)) for a in vm]) if vm is not None \
        else np.zeros(len(P))
    hot = np.asarray([float((np.asarray(b) > 0).mean()) for b in band])
    i_land = int(np.argmin(z_lo))
    return dict(frames=int(len(P)), z_centre=zc.tolist(), z_floor=z_lo.tolist(),
                kinetic=ke.tolist(), stress_mean=s_mean.tolist(), stress_p99=s_p99.tolist(),
                strained_frac=hot.tolist(),
                z_start=float(zc[0]), z_min=float(zc.min()), z_end=float(zc[-1]),
                rebound=float((zc[-1] - zc.min()) / max(zc[0] - zc.min(), 1e-9)),
                frame_lowest=i_land, stress_peak=float(s_p99.max()),
                frame_stress_peak=int(np.argmax(s_p99)))


def plot(m, out):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), facecolor="white")
    t = np.arange(m["frames"])
    axes[0].plot(t, m["z_centre"], color="#4aa3ff", lw=1.6, label="centre")
    axes[0].plot(t, m["z_floor"], color="#2b6cb0", lw=1.0, ls="--", label="underside")
    axes[0].set_ylabel("height (box units)"); axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title(f"drops {m['z_start']:.3f} $\\to$ {m['z_min']:.3f}, "
                      f"rebound {100*m['rebound']:.0f}%", fontsize=9)
    axes[1].plot(t, m["stress_p99"], color="#e0452b", lw=1.6, label="p99")
    axes[1].plot(t, m["stress_mean"], color="#e08a2e", lw=1.2, label="mean")
    axes[1].axvline(m["frame_lowest"], color="#999", lw=0.8, ls=":")
    axes[1].set_ylabel("von Mises stress"); axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_title(f"peak {m['stress_peak']:.3g} at frame {m['frame_stress_peak']}", fontsize=9)
    axes[2].plot(t, m["strained_frac"], color="#1f8a5c", lw=1.6)
    axes[2].set_ylabel("fraction of fibres above band 0")
    axes[2].set_title("how much of the block the front reached", fontsize=9)
    for a in axes:
        a.set_xlabel("frame"); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=140, facecolor="white"); plt.close(fig)


def _fibre_lines(P_t, band_t, per, max_fibres=700):
    """The particles of one fibre, in order, as a polyline -- which is what makes a fibre visible.

    `seed_ecm` lays each strand as `per` consecutive particles along one direction, so particle i
    belongs to fibre i // per. Drawing them as dots throws that away: 90,000 points in a cube read as
    fog, and the thing the matrix is supposed to be -- an intricate network of strands -- is exactly
    the structure a scatter plot cannot show. Thinned to `max_fibres` strands for legibility, and the
    thinning is stated rather than silent.
    """
    nf = P_t.shape[0] // per
    step = max(1, nf // max_fibres)
    idx = np.arange(0, nf, step)
    segs = P_t[: nf * per].reshape(nf, per, 3)[idx]
    cols = band_t[: nf * per].reshape(nf, per)[idx].mean(axis=1)
    return segs, cols, len(idx), nf


def _scatter3d(ax, P_t, band_t, per, title=None):
    """The cube in 3D, y up, fibres as strands coloured by the stress they carry."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    ax.clear(); ax.set_facecolor("black"); ax.axis("off")
    segs, cols, drawn, nf = _fibre_lines(P_t, band_t, per)
    # y is up: matplotlib's 3D axis always draws its third coordinate vertically, so the physical
    # y is passed there and z sideways. Swapping the data is honest; relabelling the axis is not.
    segs = segs[:, :, [0, 2, 1]]
    lc = Line3DCollection(list(segs), linewidths=_LW3[0], alpha=0.9)
    lc.set_array(np.clip(cols, 0, 7)); lc.set_cmap(CMAP); lc.set_clim(0, 7)
    ax.add_collection3d(lc)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=16, azim=-62)
    return drawn, nf


def _section(ax, P_t, band_t, per, slab=0.06):
    """A slab through the middle, x across and y up, again as strands."""
    from matplotlib.collections import LineCollection
    ax.clear(); ax.set_facecolor("black"); ax.axis("off")
    segs, cols, _, _ = _fibre_lines(P_t, band_t, per, max_fibres=100000)
    mid = np.abs(segs[:, :, 2].mean(axis=1) - 0.5) < slab
    seg2 = segs[mid][:, :, [0, 1]]
    lc = LineCollection(list(seg2), linewidths=_LW2[0], alpha=0.95)
    lc.set_array(np.clip(cols[mid], 0, 7)); lc.set_cmap(CMAP); lc.set_clim(0, 7)
    ax.add_collection(lc)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")


def render(P, band, d, name, per, fps=15, n_frames=150, n_col=8, lw3=0.7, lw2=1.1):
    keep = np.unique(np.round(np.linspace(0, len(P) - 1, min(n_frames, len(P)))).astype(int))
    fig = plt.figure(figsize=(11.0, 5.6), facecolor="black")
    a3 = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
    a2 = fig.add_subplot(1, 2, 2, facecolor="black")
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.02)
    wri = FFMpegWriter(fps=fps, metadata={"title": name})
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for t in keep:
            drawn, nf = _scatter3d(a3, P[t], band[t], per)
            _section(a2, P[t], band[t], per)
            a3.text2D(0.02, 0.96, f"{name}   frame {t}\n{nf} fibres of {per} particles "
                                  f"({drawn} drawn)", transform=a3.transAxes, color="white",
                      fontsize=11, va="top")
            a2.text(0.02, 0.98, "a slab through the middle", transform=a2.transAxes, color="white",
                    fontsize=11, va="top")
            wri.grab_frame()
    _scatter3d(a3, P[keep[-1]], band[keep[-1]], per); _section(a2, P[keep[-1]], band[keep[-1]], per)
    fig.savefig(os.path.join(d, "3d.png"), dpi=110, facecolor="black")
    plt.close(fig)

    # section.mp4: the cut alone, big
    fig2 = plt.figure(figsize=(5.6, 5.6), facecolor="black")
    b2 = fig2.add_subplot(111, facecolor="black")
    fig2.subplots_adjust(0, 0, 1, 1)
    wri2 = FFMpegWriter(fps=fps, metadata={"title": name + " section"})
    with wri2.saving(fig2, os.path.join(d, "section.mp4"), dpi=100):
        for t in keep:
            _section(b2, P[t], band[t], per)
            b2.text(0.02, 0.98, f"{name}   frame {t}\na slab through the middle, coloured by stress",
                    transform=b2.transAxes, color="white", fontsize=11, va="top")
            wri2.grab_frame()
    plt.close(fig2)

    # strip.png, always
    idx = np.unique(np.round(np.linspace(0, len(P) - 1, n_col)).astype(int))
    figs = plt.figure(figsize=(3.4 * len(idx), 7.0), facecolor="black")
    for i, t in enumerate(idx):
        ax = figs.add_subplot(2, len(idx), i + 1, projection="3d", facecolor="black",
                              computed_zorder=False)
        _scatter3d(ax, P[t], band[t], per)
        ax.text2D(0.04, 0.95, f"frame {t}", transform=ax.transAxes, color="white", fontsize=12,
                  va="top")
        axs = figs.add_subplot(2, len(idx), len(idx) + i + 1, facecolor="black")
        _section(axs, P[t], band[t], per)
    figs.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.02, hspace=0.02)
    figs.savefig(os.path.join(d, "strip.png"), dpi=95, facecolor="black")
    plt.close(figs)


def bands_from_vm(vm, pct=99.0):
    """Colour from the RECORDED stress, scaled after the fact -- not from a guess made before the run.

    `ecm_stress` bands at run time against `stress_scale`, and that number has to be chosen before
    anything has fallen. Set at 2e-5 against a run whose p99 peaks at 0.64, every particle after the
    impact pins at band 7 and the movie is a solid orange block: the LUT saturates and the front, which
    is the only thing the run is about, is invisible. The raw von Mises is in `traj.npz`, so the scale
    is taken from the run itself -- one p99 over every frame, fixed for the whole movie, which is the
    same convention `run_ecm.autoscale` uses.
    """
    v = np.concatenate([np.asarray(a).ravel() for a in vm[:: max(1, len(vm) // 40)]])
    sc = float(np.percentile(v[np.isfinite(v)], pct)) or 1.0
    return [np.clip(np.asarray(a, np.float32) / sc, 0, 1).__mul__(7).round().astype(np.uint8)
            for a in vm], sc


def main():
    import plexus.operators                                          # noqa: F401
    from plexus import schema
    from plexus.engine import run as engine_run

    frames = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 360
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0"
    name = (sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else "02_ecm_block")
    if "--fps" in sys.argv:
        globals()["_FPS"] = int(sys.argv[sys.argv.index("--fps") + 1])
    if "--lw" in sys.argv:
        f = float(sys.argv[sys.argv.index("--lw") + 1]); _LW3[0] = 0.7 * f; _LW2[0] = 1.1 * f
    d = os.path.join(LOG, name)
    os.makedirs(d, exist_ok=True)
    spec = build(name, frames)
    path = os.path.join(d, "spec.yaml")
    if "--render-only" in sys.argv:
        z = np.load(os.path.join(d, "traj.npz"))
        P = np.asarray(z["pos"], np.float32)
        band = [np.asarray(b) for b in z["stress"]]
        vm = [np.asarray(v, np.float32) for v in z["vm"]] if z["vm"].size else None
        band, sc = bands_from_vm(vm) if vm else (band, None)
        print(f"[{name}] re-render only; stress full-scale {sc:.4g}", flush=True)
        m = measure(P, band, vm)
        json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=1)
        plot(m, os.path.join(d, "stress.png"))
        import yaml as _y
        sp = _y.safe_load(open(os.path.join(d, "spec.yaml")))
        per = max(1, int(sp["sets"]["mpm_particle"]["per_parent"])
                  // int(next(o for o in sp["operators"] if o["op"] == "seed_ecm")["n_fibres"]))
        render(P, band, d, name, per=per, fps=globals().get("_FPS", 15))
        return
    yaml.safe_dump(spec, open(path, "w"), sort_keys=False)

    ecm_ops.STRESS_HISTORY.clear(); ecm_ops.STRESS_RAW.clear()
    H, out = engine_run(schema.load(path), device=dev)
    P = np.asarray(out["sets"]["mpm_particle"]["pos"], np.float32)
    band = [np.asarray(b) for b in ecm_ops.STRESS_HISTORY] or [np.zeros(P.shape[1], np.uint8)] * len(P)
    vm = [np.asarray(v, np.float32) for v in ecm_ops.STRESS_RAW] or None
    n = min(len(P), len(band))
    P, band = P[:n], band[:n]
    vm = vm[:n] if vm else None
    np.savez_compressed(os.path.join(d, "traj.npz"), pos=P,
                        stress=np.asarray(band, np.uint8),
                        vm=np.asarray(vm, np.float16) if vm else np.zeros((0,), np.float16))
    if vm:
        band, sc = bands_from_vm(vm)
        print(f"[{name}] stress colour full-scale {sc:.4g} (p99 over the run, from traj.npz)",
              flush=True)
    m = measure(P, band, vm)
    json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    plot(m, os.path.join(d, "stress.png"))
    render(P, band, d, name, per=max(1, spec["sets"]["mpm_particle"]["per_parent"]
                                     // spec["operators"][1]["n_fibres"]))
    print(f"[{name}] {P.shape[1]} particles, centre {m['z_start']:.3f} -> {m['z_min']:.3f} -> "
          f"{m['z_end']:.3f} (rebound {100*m['rebound']:.0f}%), stress p99 peak {m['stress_peak']:.3g} "
          f"at frame {m['frame_stress_peak']} -> {d}", flush=True)


if __name__ == "__main__":
    main()
