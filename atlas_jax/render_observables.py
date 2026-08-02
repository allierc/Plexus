"""render_observables -- a movie of the thing the mechanism actually changes.

`render_movies.py` gave every evidence folder a `movie.mp4`. Six of them show a point cloud
sitting perfectly still, because `plot_dataset` renders POSITIONS and these six mechanisms move
nothing: a gene network changes `gene`, a growth law changes `radius`, a stress sensor writes
`stress` and is explicitly required not to move a cell. Those movies are not wrong, they are
*empty* -- which is the failure this campaign keeps catching one level up: an artefact that looks
deliberate and carries no information.

`trajectory.npz` cannot help; it stores `pos`/`occ` and nothing else. `simulation.zarr` stores the
full state, so the observable is already on disk for every run ever done. Nothing is re-simulated
here either.

TWO KINDS OF RUN, AND THEY GET DIFFERENT ARTEFACTS -- deciding which is the whole job:

  * the state VARIES IN TIME (gene networks, saturating growth) -> `observable.mp4`, cells
    coloured by the observable, beside the per-cell traces with a time cursor. A real movie.
  * the state is CONSTANT IN TIME (free_screened_diffusion's steady-state chemical, virial
    stress's still-life sensor) -> `observable.png`, one frame coloured by the spatial profile.
    A movie of these would be a held frame pretending to be evidence; the stillness is the
    result, and the artefact should say so rather than animate it.

    python render_observables.py
    python render_observables.py --only gene_network_mwc
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
LOG_DIR = os.path.join(PLEXUS, "log", "atlas_jax")
DATA_DIR = os.path.join(PLEXUS, "graphs_data", "atlas_jax")

SKIP_FIELDS = {"vel", "pos", "occ"}          # kinematics, already in the position movie
BG, FG = "black", "white"


def read_run(name):
    """(pos [T,N,2], occ [T,N], {field: [T,N,C]}) straight out of the zarr. No simulation."""
    import zarr
    z = zarr.open(os.path.join(DATA_DIR, name, "simulation.zarr"), mode="r")
    for sname, g in z.groups():
        arrs, grps = dict(g.arrays()), dict(g.groups())
        if "pos" not in arrs:
            continue
        pos, occ = np.asarray(arrs["pos"][:]), np.asarray(arrs["occ"][:])
        fields = {}
        if "state" in grps:
            for k, v in grps["state"].arrays():
                if k in SKIP_FIELDS:
                    continue
                a = np.asarray(v[:], np.float64)
                if a.ndim == 2:
                    a = a[..., None]
                fields[k] = a
        return sname, pos, occ, fields
    raise ValueError(f"{name}: no set with positions in the zarr")


def varies_in_time(a, occ):
    """Does this field change over frames, on the cells that are actually alive?"""
    live = occ.any(0)
    if not live.any():
        return False
    return float(a[:, live].std(axis=0).max()) > 1e-9


def _scatter(ax, p, c, s, vmin, vmax, cmap, world):
    ax.set_facecolor(BG)
    sc = ax.scatter(p[:, 0], p[:, 1], c=c, s=s, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_xlim(world[0]), ax.set_ylim(world[1])
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#444444")
    return sc


def frame_window(pos, occ):
    """One spatial scale for every frame, from every live cell -- never per-frame autoscale."""
    live = occ.astype(bool)
    p = pos[live] if live.any() else pos.reshape(-1, pos.shape[-1])
    c = p.mean(0)
    r = max(1e-6, np.abs(p - c).max() * 1.20)
    return (c[0] - r, c[0] + r), (c[1] - r, c[1] + r)


def render_movie(name, sname, pos, occ, field, arr, radius, out_mp4):
    """Cells coloured by the observable + every cell's trace, with a time cursor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    # ffmpeg ships INSIDE the conda env, not on PATH -- reuse the library's own locator rather
    # than silently degrading to a gif nobody would notice was a gif.
    sys.path.insert(0, os.path.join(PLEXUS, "src"))
    from plexus.plot import _ffmpeg
    ff = _ffmpeg()
    if ff is None:
        raise RuntimeError("no ffmpeg -- refusing to write a movie by another name")
    matplotlib.rcParams["animation.ffmpeg_path"] = ff

    T, N, C = arr.shape
    live = occ.any(0)
    vmin, vmax = float(arr[:, live].min()), float(arr[:, live].max())
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12
    world = frame_window(pos, occ)
    # marker area from radius where the run has one, so a growth law is visible as growth
    if radius is not None:
        rr = radius[..., 0]
        s_all = 40.0 + 900.0 * (rr / max(1e-9, float(rr[:, live].max()))) ** 2
    else:
        s_all = np.full((T, N), 160.0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=BG,
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})
    sc = _scatter(axL, pos[0][occ[0].astype(bool)],
                  arr[0, occ[0].astype(bool), 0], s_all[0][occ[0].astype(bool)],
                  vmin, vmax, "viridis", world)
    cb = fig.colorbar(sc, ax=axL, fraction=0.046, pad=0.02)
    cb.ax.tick_params(colors=FG, labelsize=8)
    cb.outline.set_edgecolor("#444444")

    axR.set_facecolor(BG)
    idx = np.where(live)[0]
    for i in idx:
        for c in range(C):
            axR.plot(np.arange(T), arr[:, i, c], lw=1.0, alpha=0.85,
                     color=plt.get_cmap("viridis")((c + 0.5) / C))
    cursor = axR.axvline(0, color="#FF4F4F", lw=1.4)
    axR.set_xlim(0, T - 1), axR.set_ylim(vmin - 0.05 * (vmax - vmin), vmax + 0.05 * (vmax - vmin))
    axR.tick_params(colors=FG, labelsize=8)
    for sp in axR.spines.values():
        sp.set_color("#444444")
    lab = fig.text(0.005, 0.985, "", color=FG, fontsize=11, va="top", ha="left")
    fig.text(0.005, 0.03, f"{name} · {sname}.{field}   {len(idx)} cells · {C} channel(s)",
             color="#AAAAAA", fontsize=9, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    writer = FFMpegWriter(fps=8, codec="h264", bitrate=2400)
    with writer.saving(fig, out_mp4, dpi=130):
        for t in range(T):
            m = occ[t].astype(bool)
            sc.set_offsets(pos[t][m])
            sc.set_array(arr[t, m, 0])
            sc.set_sizes(s_all[t][m])
            cursor.set_xdata([t, t])
            lab.set_text(f"t={t}   n={int(m.sum())}   {field}[0] "
                         f"{arr[t, m, 0].min():.3g} … {arr[t, m, 0].max():.3g}")
            writer.grab_frame(facecolor=BG)
    plt.close(fig)
    return out_mp4


def render_still(name, sname, pos, occ, fields, out_png):
    """One panel per spatial observable. The stillness IS the result -- say it, don't animate it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k, a in fields.items() if float(np.ptp(a[0, occ[0].astype(bool)])) > 1e-9]
    if not keys:
        keys = sorted(fields)
    world = frame_window(pos, occ)
    m = occ[0].astype(bool)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2), facecolor=BG,
                             squeeze=False)
    for ax, k in zip(axes[0], keys):
        a = fields[k][0, m, 0]
        sc = _scatter(ax, pos[0][m], a, 200.0, float(a.min()), float(a.max()), "magma", world)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(colors=FG, labelsize=8)
        cb.outline.set_edgecolor("#444444")
        ax.text(0.03, 0.97, f"{k}\n{a.min():.4g} … {a.max():.4g}", transform=ax.transAxes,
                color=FG, fontsize=11, va="top", ha="left")
    fig.text(0.01, 0.02, f"{name} · {sname} · identical across all {pos.shape[0]} frames\n"
                         f"still life by construction — not a stalled run",
             color="#AAAAAA", fontsize=8, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out_png, dpi=140, facecolor=BG)
    plt.close(fig)
    return out_png, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    names = [n for n in sorted(os.listdir(LOG_DIR))
             if os.path.isdir(os.path.join(DATA_DIR, n, "simulation.zarr"))
             and (not a.only or n in a.only)]

    movies, stills, skipped, failed = [], [], [], []
    for name in names:
        try:
            sname, pos, occ, fields = read_run(name)
            # RELATIVE, not absolute. An operator that is required to move nothing still leaves
            # ~1e-6 of float32 round-off on coordinates of order 10, and an absolute 1e-6 bar
            # read that as motion -- sending the one mechanism with the clearest time-varying
            # observable to the "already covered" pile.
            motion = float(pos[:, occ.any(0)].std(axis=0).max())
            if motion > 1e-5 * max(1.0, float(np.abs(pos).max())):
                skipped.append(name)          # it moves -- the position movie already shows it
                continue
            live_var = {k: v for k, v in fields.items() if varies_in_time(v, occ)}
            out = os.path.join(LOG_DIR, name)
            if live_var:
                field = max(live_var, key=lambda k: float(live_var[k][:, occ.any(0)].std()))
                radius = fields.get("radius")
                p = render_movie(name, sname, pos, occ, field, live_var[field], radius,
                                 os.path.join(out, "observable.mp4"))
                movies.append((name, field, os.path.getsize(p) / 1e6))
                print(f"  movie  {name:<30} {sname}.{field}  {os.path.getsize(p)/1e6:.1f} MB")
            elif fields:
                p, drawn = render_still(name, sname, pos, occ, fields,
                                        os.path.join(out, "observable.png"))
                stills.append((name, drawn))
                print(f"  still  {name:<30} {', '.join(drawn)}   "
                      f"(uniform, not drawn: {', '.join(sorted(set(fields) - set(drawn))) or '-'})")
            else:
                # occupancy-only run (death): the position movie already shows cells vanishing
                skipped.append(name)
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  FAIL   {name:<30} {type(e).__name__}: {e}")

    print(f"\n[observables] {len(movies)} movie(s), {len(stills)} still(s), "
          f"{len(skipped)} already shown by the position movie, {len(failed)} failed")
    for n, why in failed:
        print(f"  FAILED  {n}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
