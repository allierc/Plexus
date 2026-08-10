#!/usr/bin/env python
"""run_ecm -- run one ECM experiment and leave the movie, the strip and the numbers behind.

    python run_ecm.py 25_epi_ecm_E40 --frames 320 --device cuda:0     # stand-in sphere
    python epi_sweep.py --device cuda:0                               # the real epithelium, x5

Everything lands in `log/okuda_ECM/<name>/`: `movie.mp4`, `strip.png`, `spec_run.yaml` and
`metrics.json`. The spec is written beside the result because a movie without the spec that made
it is an anecdote -- and the sweep varies stiffness, cavity shape and fibre architecture, so "which
run was this" is a question that gets asked of every frame.

WHAT THE NUMBERS ARE FOR. The movie shows the stress front; `metrics.json` says whether it was
real, and now says it FRAME BY FRAME rather than once at the end:

  contact_frame   the first frame any matrix particle is inside the tissue's apical surface. Before
                  it, nothing this experiment is about has happened; a run whose contact frame is 0
                  was seeded wrong -- the tissue was already overlapping the matrix when the clock
                  started, so "first contact" is not an event it can report.
  strained_frac   the fraction of matrix carrying |J-1| above the colour floor, PER FRAME. It
                  should be ~0 before contact and grow after it. This used to be a single number
                  read off `node_type`, which the recorder saves only once (its final value); the
                  per-frame series comes from the stress history the operator keeps, so the
                  propagation is now measured and not only watched.
  front_r95       the 95th-percentile radius of the strained particles, per frame: WHERE the front
                  is. Together with contact_frame this is the propagation as two curves, and a run
                  whose front never leaves the tissue surface is a null however bright it looks.
  front_cheb95    the same percentile in CHEBYSHEV distance -- distance to the nearest wall of the
                  cubic domain rather than to its centre. `front_reaches_wall` is read off this,
                  and once the front touches a wall the run's later frames describe the box.
  max_disp        the furthest any particle has moved from where it was seeded. Distinguishes a
                  matrix being pushed from a matrix falling apart.

`remeasure(out_dir)` recomputes all of it from `traj.npz`, and `rerender(out_dir)` redraws from the
same file: neither costs a re-simulation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "prototype", "Tyssue"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

LOG = os.path.join(ROOT, "log", "okuda_ECM")

# How close to the surface counts as touching it, in box units. One grid cell at n_grid=48 is 0.021, so
# this is a fifth of a cell -- inside the width the MPM kernel itself smears a boundary over.
CONTACT_SKIN = 0.004


# --------------------------------------------------------------------------- the numbers
def surface_radius(pos, M, centre):
    """Per-particle tissue-surface radius, by the SAME angular lookup the operator uses.

    Duplicated in numpy rather than approximated by a median, because `contact_frame` has to be the
    same event the physics saw. The previous version compared each particle's distance against the
    MEDIAN of the surface map, so a particle sitting beyond a bulge counted as inside and one
    tucked into a dimple counted as outside -- and with a disc cavity thinner than the tissue's
    starting radius that reported contact at frame 0 for every run in the folder.
    """
    nth, nph = M.shape
    d = pos - centre
    r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    u = d / r[:, None]
    th = np.arccos(np.clip(u[:, 2], -1, 1))
    ph = np.arctan2(u[:, 1], u[:, 0]) % (2 * math.pi)
    it = np.clip((th / math.pi * nth).astype(int), 0, nth - 1)
    ip = np.clip((ph / (2 * math.pi) * nph).astype(int), 0, nph - 1)
    return r, M[it, ip]


def measure(out, spec, stress=None, seeded=None):
    s = out["sets"]["mpm_particle"]
    pos = np.asarray(s["pos"])                      # [T, N, 3]
    T = pos.shape[0]
    op = next(o for o in spec["operators"] if o["op"] == "cell_to_ecm")
    c = np.asarray(op["centre"], float)

    contact, tissue_r, strained, front, cheb = None, [], [], [], []
    if op.get("implementation") == "replay":
        z = np.load(op["surface"])
        S = np.asarray(z["smap"], float) * float(op.get("scale", 1.0))
    else:
        S = None
    for t in range(T):
        if S is not None:
            # CLAMPED, NOT WRAPPED. `np.resize` was used to match the surface's frame count to the
            # recorder's, and the recorder keeps one frame more -- so the LAST entry wrapped round
            # to the FIRST, and every run in this folder reported `ball_r_final` equal to its
            # INITIAL radius (0.116 for a tissue that reached 0.397).
            M = S[min(t, S.shape[0] - 1)]
            r, R = surface_radius(pos[t], M, c)
            tissue_r.append(float(np.median(M)))
            # CONTACT IS THE SURFACE REACHING A PARTICLE, not a particle being found INSIDE it. The
            # strict test was right while penetration was the only evidence of contact, and it broke the
            # moment `cell_exclude_3d` started preventing penetration: a run with a working
            # non-penetration constraint has almost nothing inside the tissue ever, so the strict test
            # reported first contact at frame 197 for a tissue that had been pressing since frame 35.
            # A metric that fails when the physics improves is measuring the artefact.
            if contact is None and (r - R < CONTACT_SKIN).any():
                contact = t
        else:
            rr = min(op["r_max"], op["r0"] + op["growth"] * t)
            tissue_r.append(rr)
            r = np.linalg.norm(pos[t] - c, axis=1)
            if contact is None and (r < rr).any():
                contact = t
        if stress is not None and t < len(stress):
            b = np.asarray(stress[t])
            hot = b > 0
            strained.append(float(hot.mean()))
            front.append(float(np.percentile(r[hot], 95)) if hot.any() else 0.0)
            # "HAS THE FRONT REACHED THE WALL" IS NOT A RADIUS QUESTION. The domain is a CUBE: its
            # faces are at 0.5 from the centre but its corners are at 0.866, so a radial 95th
            # percentile passes 0.45 while the front is still nowhere near a face -- it just found a
            # corner. Chebyshev distance is the distance to the nearest face, which is the thing
            # being asked about. A run whose front reaches the wall is also a run whose later frames
            # describe the BOX rather than the matrix, so this number is a validity flag.
            cheb.append(float(np.percentile(np.abs(pos[t] - c).max(1)[hot], 95))
                        if hot.any() else 0.0)

    start = pos[0] if seeded is None else seeded
    disp = np.linalg.norm(pos - start[None], axis=2)
    q = lambda a, f: (float(np.asarray(a)[int(f * (len(a) - 1))]) if len(a) else None)
    return {"frames": int(T), "n_particles": int(pos.shape[1]),
            "contact_frame": contact,
            "tissue_r_start": tissue_r[0], "tissue_r_final": tissue_r[-1],
            "strained_frac_end": (strained[-1] if strained else None),
            "strained_frac_at_contact": (strained[contact] if strained and contact is not None
                                         and contact < len(strained) else None),
            "strained_frac_q25": q(strained, 0.25), "strained_frac_q50": q(strained, 0.50),
            "strained_frac_q75": q(strained, 0.75),
            "front_r95_end": (front[-1] if front else None),
            "front_r95_q50": q(front, 0.50),
            "front_cheb95_end": (cheb[-1] if cheb else None),
            "front_reaches_wall": (next((t for t, f in enumerate(cheb) if f > 0.45), None)
                                   if cheb else None),
            "max_disp": float(disp.max()), "med_disp_final": float(np.median(disp[-1])),
            "strained_frac": [round(v, 5) for v in strained],
            "front_r95": [round(v, 5) for v in front],
            "front_cheb95": [round(v, 5) for v in cheb],
            "exploded": bool(np.isnan(pos).any() or float(np.abs(pos).max()) > 5.0)}


def remeasure(out_dir):
    """Recompute `metrics.json` from `traj.npz` -- so a corrected metric never costs a re-run.

    The same reason `traj.npz` exists at all: a definition that turns out to be wrong (a radial
    threshold standing in for a distance to a wall, a wrap-around index) should be a two-second fix
    applied to every finished run, not a reason to keep a number nobody trusts.
    """
    spec = yaml.safe_load(open(os.path.join(out_dir, "spec_run.yaml")))
    # A RUN SUBMITTED TO THE CLUSTER RECORDS CLUSTER PATHS. The same NFS export is mounted at
    # /workspace in the devcontainer and at /groups/saalfeld/home/allierc/Graph on gpu_l4, so a spec
    # written there names files this side cannot open -- re-rendering 69 locally died on a surface map
    # that exists, under its other name. The mapping is `cluster.MAP`; it is read from there rather
    # than restated, so there is one place it can be wrong.
    try:
        sys.path.insert(0, "/workspace/Plexus/discovery_okuda")
        from cluster import MAP as _MAP
    except Exception:
        _MAP = ("/workspace", "/groups/saalfeld/home/allierc/Graph")
    def _here(v):
        return _MAP[0] + v[len(_MAP[1]):] if isinstance(v, str) and v.startswith(_MAP[1]) else v
    for _o in spec.get("operators", []):
        for _k in ("surface", "load", "gate", "map"):
            if _k in _o:
                _o[_k] = _here(_o[_k])
    z = np.load(os.path.join(out_dir, "traj.npz"))
    out = {"sets": {"mpm_particle": {"pos": np.asarray(z["pos"])}}}
    m = measure(out, spec, stress=list(np.asarray(z["stress"])))
    old = os.path.join(out_dir, "metrics.json")
    prev = json.load(open(old)) if os.path.exists(old) else {}
    for k in ("wall_s", "name", "varied"):                       # provenance the trajectory lost
        if k in prev:
            m[k] = prev[k]
    json.dump(m, open(old, "w"), indent=1)
    return m


# --------------------------------------------------------------------------- the pictures
def rerender(out_dir, **kw):
    """Redraw a finished run from `traj.npz` -- no GPU, no re-simulation."""
    spec = yaml.safe_load(open(os.path.join(out_dir, "spec_run.yaml")))
    # A RUN SUBMITTED TO THE CLUSTER RECORDS CLUSTER PATHS. The same NFS export is mounted at
    # /workspace in the devcontainer and at /groups/saalfeld/home/allierc/Graph on gpu_l4, so a spec
    # written there names files this side cannot open -- re-rendering 69 locally died on a surface map
    # that exists, under its other name. The mapping is `cluster.MAP`; it is read from there rather
    # than restated, so there is one place it can be wrong.
    try:
        sys.path.insert(0, "/workspace/Plexus/discovery_okuda")
        from cluster import MAP as _MAP
    except Exception:
        _MAP = ("/workspace", "/groups/saalfeld/home/allierc/Graph")
    def _here(v):
        return _MAP[0] + v[len(_MAP[1]):] if isinstance(v, str) and v.startswith(_MAP[1]) else v
    for _o in spec.get("operators", []):
        for _k in ("surface", "load", "gate", "map"):
            if _k in _o:
                _o[_k] = _here(_o[_k])
    z = np.load(os.path.join(out_dir, "traj.npz"))
    import ecm_ops
    ecm_ops.STRESS_HISTORY[:] = list(np.asarray(z["stress"]))
    ecm_ops.STRESS_RAW[:] = list(np.asarray(z["vm"])) if "vm" in z.files else []
    ecm_ops.BALL_RADIUS[:] = list(np.asarray(z["radius"], float))
    out = {"sets": {"mpm_particle": {"pos": np.asarray(z["pos"])}}}
    if "mpos" in z.files:
        import membrane_ops
        membrane_ops.MEMBRANE_STRAIN[:] = (list(np.asarray(z["mstrain"]))
                                           if "mstrain" in z.files else [])
        out["sets"]["basement_membrane_particle"] = {"pos": np.asarray(z["mpos"])}
        if "malive" in z.files:
            membrane_ops.MEMBRANE_ALIVE = np.asarray(z["malive"], bool)
        if "bond_frames" in z.files:
            membrane_ops.BOND_SNAPSHOTS = [
                (int(f), np.asarray(z["bond_i"][a:b]), np.asarray(z["bond_j"][a:b]),
                 np.asarray(z["bond_s"][a:b], float))
                for f, a, b in zip(z["bond_frames"], z["bond_off"][:-1], z["bond_off"][1:])]
    if "bpos" in z.files:
        import block_ops
        block_ops.BLOCK_STRESS[:] = list(np.asarray(z["bstress"]))
        block_ops.BLOCK_RAW[:] = list(np.asarray(z["bvm"])) if "bvm" in z.files else []
        out["sets"]["mpm_block"] = {"pos": np.asarray(z["bpos"])}
    # `fps` lives in the spec, not in render()'s signature, but a re-render is exactly when you want to
    # change it without touching the archived spec on disk.
    if "fps" in kw:
        spec.setdefault("plotting", {})["fps"] = int(kw.pop("fps"))
    render(os.path.basename(out_dir.rstrip("/")), out, spec, out_dir, **kw)


def render_sphere(name, pos, hist, spec, out_dir, cmap, ax_i, centre, T,
                  n_strip=8, movie=True, movie_frames=80):
    """The stand-in-sphere runs (`sweep.py` 01-20): the matrix, plus the sphere as a wireframe.

    Three rows -- matrix from the side, matrix with the near octant cut away, and the mid-plane
    section -- in BOX units, since there is no tissue frame to map into. The sphere is drawn as a
    wireframe of the radius the operator actually used that frame, so it reads unmistakably as a
    prescribed boundary and not as a cell ball.
    """
    import matplotlib.pyplot as plt
    import ecm_ops
    import ecm_render as RD
    radii = ecm_ops.BALL_RADIUS
    plane = [i for i in range(3) if i != ax_i]
    keep = np.unique(np.linspace(0, T - 1, min(movie_frames, T)).astype(int))
    L = 0.5
    uu, vv = np.mgrid[0:2 * np.pi:24j, 0:np.pi:13j]

    def panels(fig, n_col, i, t):
        q = pos[t] - centre
        band = np.asarray(hist[t]) if hist and t < len(hist) else np.zeros(pos.shape[1], np.uint8)
        r = radii[t] if t < len(radii) else 0.0
        for row, cut in enumerate([False, True]):
            ax = fig.add_subplot(3, n_col, row * n_col + i + 1, projection="3d",
                                 computed_zorder=False, facecolor="black")
            RD.draw_3d(ax, None, None, q, band, cmap, RD.CAM_SIDE, L, tissue=False, cutaway=cut)
            if row == 0 and r > 0:
                ax.plot_wireframe(r * np.cos(uu) * np.sin(vv), r * np.sin(uu) * np.sin(vv),
                                  r * np.cos(vv), color="#39d0ff", lw=0.5, alpha=0.8, zorder=5)
                ax.text2D(0.03, 0.95, f"frame {t}   prescribed r={r:.3f}   "
                                      f"strained {float((band > 0).mean()) * 100:.0f}%",
                          transform=ax.transAxes, color="white", fontsize=11, va="top")
        axc = fig.add_subplot(3, n_col, 2 * n_col + i + 1, facecolor="black")
        sl = np.abs(q[:, ax_i]) < 0.06
        RD._matrix_scatter(axc, q[sl][:, plane], band[sl], cmap, zorder=0, alpha=0.95,
                           s_rest=3.4, s_hot=7.0, three_d=False)
        if r > 0:
            axc.add_patch(plt.Circle((0, 0), r, fill=False, ec="#39d0ff", lw=1.1, alpha=0.9))
        axc.set_xlim(-L, L); axc.set_ylim(-L, L); axc.set_aspect("equal"); axc.axis("off")

    idx = np.unique(np.linspace(0, T - 1, n_strip).astype(int))
    fig = plt.figure(figsize=(4.4 * len(idx), 13.5), facecolor="black")
    for i, t in enumerate(idx):
        panels(fig, len(idx), i, int(t))
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995, wspace=0.02, hspace=0.02)
    fig.savefig(os.path.join(out_dir, "strip.png"), dpi=100, facecolor="black")
    plt.close(fig)
    print(f"[{name}] strip.png ({len(idx)} columns, prescribed-sphere layout)", flush=True)
    if not movie:
        return
    from matplotlib.animation import FFMpegWriter
    import matplotlib
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    figm = plt.figure(figsize=(5.4, 13.5), facecolor="black")
    wri = FFMpegWriter(fps=int(spec.get("plotting", {}).get("fps", 10)), metadata={"title": name})
    with wri.saving(figm, os.path.join(out_dir, "movie.mp4"), dpi=95):
        for t in keep:
            figm.clear(); figm.patch.set_facecolor("black")
            panels(figm, 1, 0, int(t))
            figm.subplots_adjust(0, 0, 1, 1, 0.0, 0.02)
            wri.grab_frame()
    plt.close(figm)
    print(f"[{name}] movie.mp4 ({len(keep)} frames, prescribed-sphere layout)", flush=True)


def autoscale(raw, pct=99.0, sample=12):
    """One colour full-scale for the WHOLE run, taken from the run's own distribution.

    NOT PER FRAME. Rescaling each frame to its own maximum makes a growing load and a static one look
    identical -- the same defect `run_box` fixes for the camera and `ecm_stress` warns about for the
    palette. But a HAND-PICKED fixed scale has the opposite failure, and runs 47/48 hit it: 0.008
    resolved the front at frame 200 and left 76% of the matrix saturated at frame 400. Taking a high
    percentile over frames sampled across the run keeps one scale, chosen by the data, so the top band
    means "the most stressed material this run produced" instead of "whatever I guessed".
    """
    if not len(raw):
        return None
    idx = np.unique(np.linspace(0, len(raw) - 1, min(sample, len(raw))).astype(int))
    v = np.concatenate([np.asarray(raw[i], np.float32).ravel() for i in idx])
    v = v[np.isfinite(v)]
    return float(np.percentile(v, pct)) if v.size else None


def render(name, out, spec, out_dir, n_strip=8, movie_frames=None, movie=True, fps=None,
           strip_only=False, frame_limit=None, stress_scale=None, block_scale=None,
           membrane_scale=None):
    """The okuda artefact pair, with the matrix added: a 4-row strip and a 2-camera movie.

    ROWS OF THE STRIP, in the order `log/okuda/cellfix_B_new/strip.png` has them, plus one:

        1  3D side   epithelium (white, green where just divided) inside the stressed matrix
        2  3D top    the same tissue from elev 88 -- what row 1 foreshortens
        3  3D side   THE MATRIX ALONE, near octant cut away. Row 1 shows where the tissue is
                     relative to the front; row 3 shows the front with nothing in front of it. The
                     reference strip uses this row for the structural cell classes, which say
                     nothing here -- the tissue is a sphere in every frame by construction, since
                     the coupling is one-way -- so the slot goes to the thing that does vary.
        4  section   the monolayer in cross-section, in the plane of the cavity axis, with the
                     matrix sliced in the same plane

    Everything is drawn in TISSUE units: the mesh is not rescaled, the matrix is mapped into the
    tissue's frame by (p - centre)/scale. The alternative -- rescaling the tissue into the unit MPM
    box -- is what `tissue.py` refuses to do to the mechanics, and doing it in the renderer instead
    would mean the picture and the simulation disagreed about how big a cell is.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import ecm_ops
    import ecm_spec as ES
    import ecm_render as RD

    pos = np.asarray(out["sets"]["mpm_particle"]["pos"])            # [T, N, 3]
    hist = ecm_ops.STRESS_HISTORY
    T = min(pos.shape[0], len(hist)) if hist else pos.shape[0]
    if frame_limit:
        T = min(T, int(frame_limit))
    if not hist:
        print(f"[{name}] no stress history -- the matrix cannot be stress-coloured", flush=True)
    cmap = ListedColormap(ES.STRESS_COLORS)
    seed_op = next(o for o in spec["operators"] if o["op"] == "seed_ecm")
    ax_i = int(seed_op["axis"])
    axis_dir = np.eye(3)[ax_i]                                      # the cavity's pinched axis
    op = next((o for o in spec["operators"] if o["op"] == "cell_to_ecm"), None)
    centre = np.asarray(op["centre"], float)

    if op is None or op.get("implementation") != "replay" or "surface" not in op:
        # THE STAND-IN SPHERE HAS NO CELLS, so it gets a renderer that does not pretend to have any.
        # `sweep.py`'s runs 01-20 are prescribed spheres and are still worth drawing; what is not
        # acceptable is drawing a proxy -- centroids, a dot cloud, a shaded ball -- in the slot where
        # an epithelium belongs, which is exactly how runs 21-23 came to show cyan dots.
        print(f"[{name}] `implementation: {op and op.get('implementation')}` -- a PRESCRIBED "
              f"SPHERE, not a tissue. Drawing the matrix and the sphere's analytic surface; there "
              f"are no cells in this run and none are drawn.", flush=True)
        return render_sphere(name, pos, hist, spec, out_dir, cmap, ax_i, centre, T,
                             n_strip=n_strip, movie=movie)
    scale = float(op.get("scale", 1.0))
    Tis = RD.load_tissue(op["surface"], scale)
    meshes = [(t, m) for t, m in Tis["meshes"] if t < T]
    plate = Tis.get("plate_gap")                        # rigid-block half-gap in TISSUE units, or None

    # THE BASEMENT MEMBRANE, if this run has one: a third MPM set with its own bond-strain history.
    import membrane_ops
    mem_pos = (np.asarray(out["sets"]["basement_membrane_particle"]["pos"])
               if "basement_membrane_particle" in out.get("sets", {}) else None)
    mem_hist = membrane_ops.MEMBRANE_STRAIN if mem_pos is not None else None
    # `render` has no Hierarchy in scope, so `run` and `rerender` both publish the mask here first.
    # (`a or b` on arrays raises, so this is a plain None check, not a fallback expression.)
    snaps = getattr(membrane_ops, "BOND_SNAPSHOTS", None) or []
    snap_f = np.asarray([q[0] for q in snaps]) if snaps else None

    def bonds_of(t):
        """The crosslink network recorded NEAREST this frame, or None. Snapshots are periodic, so the
        network drawn on a frame between two of them is the older one -- correct to within the snapshot
        interval, and far better than drawing no edges at all."""
        if snap_f is None or not len(snap_f):
            return None, None
        k = int(np.argmin(np.abs(snap_f - t)))
        return (snaps[k][1], snaps[k][2]), snaps[k][3]

    mem_alive = getattr(membrane_ops, "MEMBRANE_ALIVE", None) if mem_pos is not None else None
    if mem_alive is not None:
        mem_alive = np.asarray(mem_alive.detach().cpu() if hasattr(mem_alive, "detach")
                               else mem_alive, bool)
        print(f"[{name}] {int(mem_alive.sum())} of {mem_alive.size} membrane particles secreted; "
              f"the reserve is not drawn", flush=True)
    if mem_pos is not None:
        print(f"[{name}] basement membrane: {mem_pos.shape[1]} particles, "
              f"{len(mem_hist or [])} strain frames", flush=True)

    # THE ELASTIC BLOCK, if this run has one: a second MPM set, its own strain history, its own ramp.
    import block_ops
    bpos = np.asarray(out["sets"]["mpm_block"]["pos"]) if "mpm_block" in out.get("sets", {}) else None
    bhist = block_ops.BLOCK_STRESS if bpos is not None else None
    if bpos is not None:
        print(f"[{name}] elastic block: {bpos.shape[1]} particles, "
              f"{len(bhist or [])} strain frames", flush=True)

    # ---- ONE camera, computed once, held for every frame and every panel.
    Lbox_box = 0.5 / max(scale, 1e-12)                  # the MPM box half-width, in tissue units
    L3 = min(Lbox_box, Tis["Lbox"] * 1.60)
    L2 = L3 * 1.15                                      # the 2D section: 3D axes shrink content
    slab = 0.055 / max(scale, 1e-12)                    # section slab half-thickness, tissue units
    print(f"[{name}] camera: FIXED Lbox={L3:.2f} tissue units for all {len(meshes)} drawn frames "
          f"(tissue extent {Tis['Lbox']:.2f}, MPM box {Lbox_box:.2f}); scale={scale:.5f}",
          flush=True)

    def q_of(t):
        """The matrix in tissue coordinates."""
        return (pos[t] - centre) / max(scale, 1e-12)

    def mem_of(t):
        """The basement membrane in tissue coordinates, with its bond strain -- or None.

        MASKED TO WHAT HAS ACTUALLY BEEN SECRETED. With `reserve = 8` the set is nine times the size of
        the sheet and the unsecreted eight ninths sit at the tissue CENTRE with mass 0. Drawn unmasked
        they are a bright blob in the middle of the lumen, which is how run 66's first frames rendered:
        a membrane apparently collapsed inside the tissue. They are not membrane yet.
        """
        if mem_pos is None:
            return None
        qm = (mem_pos[min(t, mem_pos.shape[0] - 1)] - centre) / max(scale, 1e-12)
        sm = (np.asarray(mem_hist[t], np.float32) / max(mem_sc or 1.0, 1e-12)
              if (mem_hist and t < len(mem_hist)) else np.zeros(qm.shape[0], np.float32))
        if mem_alive is not None:
            a = mem_alive if mem_alive.ndim == 1 else mem_alive[min(t, mem_alive.shape[0] - 1)]
            if a.shape[0] == qm.shape[0]:
                qm, sm = qm[a], sm[a]
        # ...AND THAT MASK IS THE FINAL ONE, applied to every frame. A node alive at the END was still
        # parked at the tissue centre at frame 0, so the early frames drew tens of thousands of nodes
        # sitting in the lumen -- the green blob in the middle of the spheroid. Aliveness is per-frame
        # and the position says it exactly: unsecreted material is AT the centre, so anything at a small
        # fraction of the sheet radius is not membrane yet. Done here rather than by storing a per-frame
        # mask so that runs already on disk are fixed on re-render.
        # THE TEST IS "IS IT IN THE BOX", not "is its radius a decent fraction of the median". The
        # median heuristic assumed the reserve waits at the tissue CENTRE; it now waits OUTSIDE the box,
        # where it inflates the median instead of deflating it -- so early frames, which are almost all
        # reserve, pushed the threshold above the live sheet and hid the membrane entirely. It reappeared
        # only once secretion had released enough material to drag the median back down, about 70% of the
        # way through the run. Position still says it exactly, just not by that statistic: dormant
        # material is either at the centre or outside the walls, and live material is neither.
        Pb = mem_pos[min(t, mem_pos.shape[0] - 1)]
        rb = np.linalg.norm(Pb - centre, axis=1)
        live_now = np.all((Pb > 0.0) & (Pb < 1.0), axis=1) & (rb > 0.02)
        if mem_alive is not None and a.shape[0] == live_now.shape[0]:
            live_now = live_now[a] if qm.shape[0] != live_now.shape[0] else live_now
        if live_now.shape[0] == qm.shape[0]:
            qm, sm = qm[live_now], sm[live_now]
        return qm, sm

    def blk_of(t):
        """The block in tissue coordinates, with its own band -- or None if there is no block."""
        if bpos is None:
            return None
        if braw and t < len(braw):
            b = (np.clip(np.asarray(braw[t], np.float32) / max(bsc, 1e-12), 0, 1)
                 * 7).round().astype(np.uint8)
        else:
            b = (np.asarray(bhist[t]) if bhist and t < len(bhist)
                 else np.zeros(bpos.shape[1], np.uint8))
        return ((bpos[min(t, bpos.shape[0] - 1)] - centre) / max(scale, 1e-12), b)

    # THE RAW SCALAR WINS WHERE IT EXISTS. `stress` in traj.npz is already banded and clipped; `vm` is
    # the number itself, so the scale below is a rendering decision and re-deciding it costs nothing.
    raw = ecm_ops.STRESS_RAW
    sc = stress_scale or autoscale(raw)
    braw = block_ops.BLOCK_RAW if bpos is not None else []
    bsc = block_scale or autoscale(braw)
    if sc:
        # A GUARD, so an unstable run can never again be ranked best on its membrane numbers alone.
        # `k_adh = 8e4` blew the matrix up by three orders of magnitude while LOOKING like the value that
        # finally held the sheet out; the tell was here, in a scale nobody was reading.
        if float(sc) > 100.0:
            print(f"[{name}] *** UNSTABLE: ECM stress p99 = {float(sc):.4g}. Healthy runs sit at 2-8. "
                  f"The membrane numbers from this run are NOT comparable -- a blown-up matrix can hold "
                  f"a sheet out geometrically while tearing it.", flush=True)
        print(f"[{name}] stress colour full-scale {sc:.5g} (p99 over the run)"
              + (f"; block {bsc:.5g}" if bsc else ""), flush=True)

    # TWO MATERIALS, TWO FIXED SCALES. The interstitial ECM and the basement membrane carry different
    # quantities (von Mises stress in MPM units against crosslink strain, dimensionless) that differ by
    # orders of magnitude, so one shared full-scale would render whichever is smaller as uniformly
    # unstrained. They also get different ramps -- inferno for the ECM, green-to-amber for the membrane --
    # so the two are never confused for one field.
    #
    # AND BOTH ARE FIXED OVER THE RUN. The membrane was being normalised PER FRAME, which is the defect
    # `autoscale` exists to avoid: rescaling every frame to its own p99 makes a sheet whose strain is
    # climbing look exactly like one at equilibrium, and the whole question about this membrane is
    # whether strain accumulates faster than turnover removes it.
    mem_sc = membrane_scale or autoscale(mem_hist) if mem_pos is not None else None
    if mem_sc:
        print(f"[{name}] membrane colour full-scale {mem_sc:.5g} strain (p99 over the run), "
              f"separate from the ECM's {sc:.5g}", flush=True)

    def band_of(t):
        if raw and t < len(raw):
            v = np.asarray(raw[t], np.float32) / max(sc, 1e-12)
            return (np.clip(v, 0, 1) * 7).round().astype(np.uint8)
        return (np.asarray(hist[t]) if hist and t < len(hist)
                else np.zeros(pos.shape[1], np.uint8))

    # ------------------------------------------------------------------ strip
    idx = [meshes[int(round(f * (len(meshes) - 1)))] for f in np.linspace(0, 1, n_strip)]
    fig = plt.figure(figsize=(4.4 * n_strip, 18.0), facecolor="black")
    for i, (t, mt) in enumerate(idx):
        vp, q, band, blk = mt["pos"], q_of(t), band_of(t), blk_of(t)
        div, brk = RD.divided_mask(mt), RD.broken_mask(mt, vp, name)
        for row, (cam, tissue, cut) in enumerate([(RD.CAM_SIDE, True, False),
                                                  (RD.CAM_TOP, True, False),
                                                  (RD.CAM_SIDE, False, True)]):
            ax = fig.add_subplot(4, n_strip, row * n_strip + i + 1, projection="3d",
                                 computed_zorder=False, facecolor="black")
            RD.draw_3d(ax, mt, vp, q, band, cmap, cam, L3, div=div, brk=brk, tissue=tissue,
                       cutaway=cut, plate_gap=plate, blk=blk, mem=mem_of(t))
            if row == 0:
                ax.text2D(0.03, 0.95, f"frame {t}   {int(mt['nF'])} cells   "
                                      f"strained {float((band > 0).mean()) * 100:.0f}%",
                          transform=ax.transAxes, color="white", fontsize=11, va="top")
        axc = fig.add_subplot(4, n_strip, 3 * n_strip + i + 1, facecolor="black")
        RD.draw_cross(axc, mt, vp, q, band, cmap, L2, axis_dir, slab, plate_gap=plate, blk=blk,
                      mem=mem_of(t))
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995, wspace=0.02, hspace=0.02)
    fig.savefig(os.path.join(out_dir, "strip.png"), dpi=100, facecolor="black")
    plt.close(fig)
    print(f"[{name}] strip.png ({n_strip} columns of {len(meshes)} drawn frames)", flush=True)
    if strip_only:
        return
    # NOT `or not movie`. `3d.png` is drawn from the movie figure, so returning here on movie=False
    # skipped the still as well -- and "give me the end state without waiting for the mp4" is exactly
    # the movie=False case. The still is written below, then the loop is skipped.

    # ------------------------------------------------------------------ movie
    # THE MOVIE DRAWS THE FRAMES THE MESH WAS KEPT FOR. Choosing its own would leave most panels
    # with no tissue to draw, and a movie of a matrix with the epithelium missing from four frames
    # in five is the defect this replaced, one frame at a time.
    from matplotlib.animation import FFMpegWriter
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    keep = meshes if movie_frames is None else [
        meshes[int(round(f * (len(meshes) - 1)))]
        for f in np.linspace(0, 1, min(movie_frames, len(meshes)))]

    # TWO FULL PANELS: the 3D view on the left, the cross-section on the right. The previous layout
    # had two 3D cameras plus a section inset in the bottom-right corner -- and the inset was the panel
    # carrying the monolayer, the lumen and the front's timing, squeezed into a fifth of the width. The
    # top-down camera exists to catch a protrusion pointing at the side camera; the tissue here is a
    # sphere or an ovoid by construction, so it was the panel with the least to say.
    # THREE PANELS when there is a basement membrane or per-junction myosin to show. Both live in a
    # shell a few percent of the tissue radius thick: at the whole-tissue framing the junction network is
    # thinner than the line used to draw a cell and the membrane is a one-dot rim, so neither is readable
    # in the panels the other results need. The zoom is the only place a colour scale can honestly be
    # spent on them.
    # 2x2 WHEN THERE IS A BASEMENT MEMBRANE OR PER-JUNCTION MYOSIN. Top row is the whole tissue -- 3D
    # and cross-section -- and the bottom row is the SAME two views zoomed onto a patch of the surface,
    # which is the only scale at which a sheet a few percent of the radius thick and a junction network
    # thinner than a cell outline can be read. The bottom-right is the one that shows the layering the
    # biology is about: lumen, epithelium, basement membrane, stroma, outward in that order.
    # A FIXED MYOSIN SCALE, like the other two. The junction panel was normalising to its own p98 every
    # frame, so a sheet-wide drift in myosin -- exactly what `beta` produces -- rendered as no change at
    # all, while the membrane beside it sat on a fixed scale. Two panels, two conventions, is a way to
    # publish a null.
    _myo = np.concatenate([np.asarray(m["myo"], float).ravel() for _, m in meshes if "myo" in m]) \
        if any("myo" in m for _, m in meshes) else None
    myo_sc = float(np.percentile(_myo, 98)) if _myo is not None and _myo.size else None
    if myo_sc:
        print(f"[{name}] myosin colour full-scale {myo_sc:.4g} (p98 over the run, fixed)", flush=True)
    zoom = (mem_pos is not None) or any("myo" in m for _, m in meshes)
    if zoom:
        figm = plt.figure(figsize=(11.0, 11.0), facecolor="black")
        axs = figm.add_subplot(2, 2, 1, projection="3d", computed_zorder=False, facecolor="black")
        axc2 = figm.add_subplot(2, 2, 2, facecolor="black")
        axz = figm.add_subplot(2, 2, 3, projection="3d", computed_zorder=False,
                               facecolor="black")
        axzc = figm.add_subplot(2, 2, 4, projection="3d", computed_zorder=False,
                                facecolor="black")
        # A 2D ZOOM INSET IN EACH BOTTOM PANEL. The 3D views show each entity whole; at whole-tissue
        # framing neither sheet nor network resolves to more than a few pixels, so the inset carries
        # the detail and the panel around it carries the context. FIGURE-level axes, not `inset_axes`
        # on the 3D parents: an inset of a 3D axis inherits its projection machinery and came out empty.
        inz = figm.add_axes([0.335, 0.035, 0.155, 0.155], facecolor="black", zorder=20)
        inzc = figm.add_axes([0.818, 0.035, 0.155, 0.155], facecolor="black", zorder=20)
    else:
        figm = plt.figure(figsize=(11.0, 5.5), facecolor="black")
        axs = figm.add_subplot(1, 2, 1, projection="3d", computed_zorder=False, facecolor="black")
        axc2 = figm.add_subplot(1, 2, 2, facecolor="black")
        axz = axzc = None
    figm.subplots_adjust(0, 0, 1, 1, wspace=0.02, hspace=0.02)
    fps = int(fps or spec.get("plotting", {}).get("fps", 10))
    wri = FFMpegWriter(fps=fps, metadata={"title": name})
    t0 = time.time()
    # ONE FRAME FIRST, AS A STILL. The movie takes minutes to write and the end state is what anyone
    # looks at first, so `3d.png` is the LAST frame in the same 2x2 layout, saved before the loop starts.
    # Same code path, so the still cannot drift from the movie.
    def _draw_frame(t, mt):
            vp, q, band, blk = mt["pos"], q_of(t), band_of(t), blk_of(t)
            div, brk = RD.divided_mask(mt), RD.broken_mask(mt, vp, name)
            # FRAMED TO MATCH THE SECTION BESIDE IT. `L3` is sized for the MPM cube, and measured off
            # a rendered frame the spheroid came out ~180px against ~250px in the 2D section, so the two
            # top panels could not be read against each other. 0.72 closes that ratio; the outer matrix
            # is clipped, which costs nothing -- it is a diffuse cloud and the section shows its extent.
            RD.draw_3d(axs, mt, vp, q, band, cmap, RD.CAM_SIDE, 0.72 * L3, div=div, brk=brk,
                       plate_gap=plate, blk=blk, mem=mem_of(t))
            RD.draw_cross(axc2, mt, vp, q, band, cmap, L2, axis_dir, slab, dot_scale=0.85,
                          plate_gap=plate, blk=blk, mem=mem_of(t))
            if axz is not None:
                mq = ms = None
                if mem_pos is not None:
                    mq = (mem_pos[min(t, mem_pos.shape[0] - 1)] - centre) / max(scale, 1e-12)
                    if mem_hist and t < len(mem_hist):
                        ms = np.asarray(mem_hist[t], np.float32)
                # BOTTOM-LEFT IS THE JUNCTIONS, AND NOTHING ELSE. With the membrane drawn over it the
                # network was invisible: 30k dots sit in front of a line mesh whose spacing is comparable
                # to the dot size, so the sheet simply occluded the thing the panel exists to show. The
                # membrane keeps the bottom-right, where the section shows it in its layer.
                # ONE ENTITY PER PANEL, IDENTICALLY FRAMED. The cutaway still drew the interstitial
                # matrix around the sheet, so "the membrane alone" was the membrane inside a cloud. Here
                # it is the membrane and nothing else, cut and boxed exactly as the network on its right,
                # which is what makes the two panels comparable frame by frame.
                #
                # AND THE SAME FIXED BOX AS THE TOP ROW. This was `1.12 * p98(|vp|)` -- recomputed every
                # frame, so the box grew with the tissue and the spheroid held a constant apparent size
                # while the top row showed it growing. Two rows, two camera conventions, and the growth
                # visible in one and invisible in the other. A camera that tracks its subject hides the
                # one thing the run is about.
                Lt = 0.72 * L3
                # UNMASKED positions here, with the mask passed separately: bond indices are in the
                # FULL particle space, so indexing an already-masked array would connect the wrong nodes.
                _bij, _bs = bonds_of(t)
                _raw = (mem_pos[min(t, mem_pos.shape[0] - 1)] - centre) / max(scale, 1e-12)
                _al = None
                if mem_alive is not None:
                    _al = mem_alive if mem_alive.ndim == 1 else mem_alive[min(t, mem_alive.shape[0] - 1)]
                # same per-frame correction as `mem_of`, and it has to be the SAME RULE -- this was a
                # second copy of the median heuristic, so fixing one left the bottom-left panel drawing
                # no membrane at all while the strip and the zoom drew it correctly.
                _Pb = mem_pos[min(t, mem_pos.shape[0] - 1)]
                _now = (np.all((_Pb > 0.0) & (_Pb < 1.0), axis=1)
                        & (np.linalg.norm(_Pb - centre, axis=1) > 0.02))
                _al = _now if _al is None else (_al & _now)
                _rs = (np.asarray(mem_hist[t], np.float32) / max(mem_sc or 1.0, 1e-12)
                       if (mem_hist and t < len(mem_hist)) else None)
                RD.draw_membrane_3d(axz, _raw, _rs, RD.CAM_SIDE, Lt, mem_hi=1.0,
                                    alive=_al, bonds=_bij, bond_s=_bs)
                # BOTTOM-RIGHT IS THE JUNCTION NETWORK, on its own. The membrane is not drawn here at
                # all: 30k dots sit in front of a line mesh of comparable spacing and simply occlude it,
                # so the two entities get a panel each rather than one panel showing neither well.
                # FRAMED ON THE TISSUE, NOT ON THE MATRIX BOX. `L3` is sized for the MPM cube, which is
                # half again the tissue extent, so the network rendered as a small ball in a large frame.
                RD.draw_junctions_3d(axzc, mt, vp, RD.CAM_SIDE, Lt, myo_hi=myo_sc)
                # the insets: the membrane patch under the cutaway, the junction patch under the network
                # r_ref = the SAME fixed extent the panels above use. The insets were the last place
                # still sizing themselves off the current radius each frame, so the two rows had the
                # camera convention this file already fixed once for the panels -- growth visible in the
                # panel and invisible in the inset sitting on top of it.
                # `mem_s=_rs`, THE SAME ARRAY AND THE SAME SCALE the panel above it uses. Passing None
                # left the inset on a flat colour while the panel it magnifies was coloured by strain,
                # so the two disagreed about what they were showing.
                RD.draw_zoom(inz, mt, vp, mem_q=_raw, mem_s=_rs, name=name, frac=0.22, r_ref=Lt,
                             # mem_hi=1.0 as well: _rs is ALREADY divided by the run-wide scale, so
                             # letting the inset fall back to percentile(mem_s, 99) normalises it a
                             # second time -- same colormap, different stretch, green beside orange
                             mem_hi=1.0,
                             junctions=False, bonds=_bij, bond_s=_bs)
                RD.draw_zoom(inzc, mt, vp, mem_q=None, mem_s=None, name=name, frac=0.16, lw=2.4,
                             r_ref=Lt, myo_hi=myo_sc)
                for _a in (inz, inzc):
                    for _sp in _a.spines.values():
                        _sp.set_color("#666"); _sp.set_visible(True)
                    _a.set_xticks([]); _a.set_yticks([])
            # ONE LABEL, ON THE 3D PANEL. `_draw`/`_cross_screen` both call ax.clear(), which drops
            # any label, so it is re-stamped every frame. The camera elevation and the cut plane used
            # to be printed too and are not any more: they are fixed for the whole run and recorded in
            # the spec beside the movie, so they were spending the frame's only text on constants.
            axs.text2D(0.02, 0.96, f"{name}   frame {t}   {int(mt['nF'])} cells   "
                                   f"strained {float((band > 0).mean()) * 100:.0f}%",
                       transform=axs.transAxes, color="white", fontsize=9)


    if keep:
        _t, _mt = keep[-1]
        _draw_frame(_t, _mt)
        figm.savefig(os.path.join(out_dir, "3d.png"), dpi=140, facecolor="black")
        print(f"[{name}] 3d.png (end point, frame {_t})", flush=True)
    if not movie:
        plt.close(figm)
        return

    with wri.saving(figm, os.path.join(out_dir, "movie.mp4"), dpi=95):
        for k, (t, mt) in enumerate(keep):
            _draw_frame(t, mt)
            wri.grab_frame()
            if k == 0:
                print(f"[{name}] first movie frame in {time.time()-t0:.1f}s "
                      f"({len(keep)} to draw)", flush=True)
    plt.close(figm)
    print(f"[{name}] movie.mp4 ({len(keep)} frames @ {fps} fps, {time.time()-t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- the run
def run(name, spec, device="cuda:0", movie=True, keep_traj=True, render_kw=None):
    import plexus.operators                                    # noqa: F401  register the stock ops
    import ecm_ops
    # THE HISTORY IS PER RUN. A module-level list survives between runs in a sweep, so a second
    # run would render the first run's stress on its own particles -- silently, and looking
    # entirely plausible.
    ecm_ops.STRESS_HISTORY.clear()
    ecm_ops.BALL_RADIUS.clear()
    ecm_ops.PRESSURE_HISTORY.clear()
    ecm_ops.STRESS_RAW.clear()
    import block_ops
    block_ops.BLOCK_STRESS.clear()
    block_ops.BLOCK_RAW.clear()
    import plexus.schema as S
    from plexus.engine import run as engine_run

    out_dir = os.path.join(LOG, name)
    os.makedirs(out_dir, exist_ok=True)
    spec_path = os.path.join(out_dir, "spec_run.yaml")
    with open(spec_path, "w") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)

    t0 = time.time()
    sim = S.load(spec_path)
    H, out = engine_run(sim, device=device)
    # publish the secreted/reserve split for `render`, which has no Hierarchy in scope
    import membrane_ops as _mo
    _mo.MEMBRANE_ALIVE = getattr(H, "membrane_alive", None)
    wall = time.time() - t0

    # THE TRAJECTORY, KEPT -- so a re-render never costs a re-simulation. Asked to redraw a run
    # with the cells visible, the only honest answer was "that means running the 402 frames again",
    # because the movie was the only surviving record of where anything was.
    if keep_traj:
        try:
            import block_ops
            extra = {}
            if ecm_ops.STRESS_RAW:
                extra["vm"] = np.asarray(ecm_ops.STRESS_RAW, np.float16)
            # THE MEMBRANE TOO, or a membrane run cannot be redrawn -- which is the whole point of
            # keeping a trajectory. Without it a re-render would silently produce the same movie with the
            # basement membrane simply missing, and missing reads as "there wasn't one".
            import membrane_ops
            if "basement_membrane_particle" in out.get("sets", {}):
                # STRIDED. At 390k nodes the full record is 390k x 3 x 4 B x 403 frames = 1.9 GB for
                # `mpos` alone, and nothing reads every frame -- the movie draws 200 and the analysis
                # reads the last. The stride keeps the first and LAST frames exactly, which are the two
                # the gap measurements use.
                _mp = np.asarray(out["sets"]["basement_membrane_particle"]["pos"], np.float32)
                _st = max(1, int(np.ceil(_mp.shape[0] * _mp.shape[1] * 12 / 400e6)))
                if _st > 1:
                    _keep = np.unique(np.r_[np.arange(0, _mp.shape[0], _st), _mp.shape[0] - 1])
                    _mp = _mp[_keep]
                    extra["mpos_frames"] = _keep.astype(np.int32)
                    print(f"[{name}] mpos strided 1-in-{_st} ({len(_keep)} of "
                          f"{out['sets']['basement_membrane_particle']['pos'].shape[0]} frames, "
                          f"first and last kept)", flush=True)
                extra["mpos"] = _mp
                # THE INTEGRIN SET, when the fibres are MPM material rather than a force with a target.
                # Without it a re-render draws a sheet held by nothing visible, and the one measurement
                # the design is for -- what the fibre's length actually does -- cannot be made after.
                #
                # AND AT THIS INDENTATION, not one level out. Added a level out, it ended the membrane
                # block early and swallowed the two saves below it, so `mstrain` and `malive` were
                # written only for runs that HAD integrins: 153's section movie died on
                # `KeyError: mstrain` and its strain column read `nan` for a run whose log says
                # "403 strain frames".
                if "integrin_particle" in out.get("sets", {}):
                    _ip = np.asarray(out["sets"]["integrin_particle"]["pos"], np.float32)
                    _st2 = max(1, int(np.ceil(_ip.shape[0] * _ip.shape[1] * 12 / 400e6)))
                    if _st2 > 1:
                        _k2 = np.unique(np.r_[np.arange(0, _ip.shape[0], _st2), _ip.shape[0] - 1])
                        _ip = _ip[_k2]
                        extra["ipos_frames"] = _k2.astype(np.int32)
                    extra["ipos"] = _ip
                    print(f"[{name}] integrin fibres: {_ip.shape[1]} particles recorded", flush=True)
                if membrane_ops.MEMBRANE_STRAIN:
                    extra["mstrain"] = np.asarray(membrane_ops.MEMBRANE_STRAIN, np.float16)
                # WHICH PARTICLES ARE MEMBRANE AND WHICH ARE UNSECRETED RESERVE. With `reserve = 8`,
                # eight ninths of the set starts parked at the tissue centre with mass 0, so any
                # measurement or render that treats the set as one body reports a membrane collapsed
                # into the lumen. Anything reading this file must mask on it.
                al = getattr(H, "membrane_alive", None)
                if al is not None:
                    extra["malive"] = np.asarray(al.detach().cpu(), bool)
                # THE HOOP MAP, in the shape the growth gate reads. `ecm_growth_gate_3d` takes a
                # (frames, ntheta, nphi) pressure map; the sheet's hoop tension is recorded per latitude
                # band, so it is broadcast across longitude -- which is exact here, because a corset
                # built from an axis-aligned anisotropy IS axisymmetric. Without this file the corset
                # cannot reach the epithelium at all: runs 96-99 recorded the tension and had nowhere to
                # send it, so they came back identical.
                if getattr(membrane_ops, "HOOP_TRACE", None):
                    hp = np.asarray(membrane_ops.HOOP_TRACE, np.float32)      # (T, nth)
                    np.savez_compressed(os.path.join(out_dir, "hoop.npz"),
                                        pmap=np.repeat(hp[:, :, None], 64, axis=2))
                    print(f"[{name}] hoop.npz  {hp.shape[0]} frames x {hp.shape[1]} latitude bands, "
                          f"mean tension {float(hp.mean()):.4g}", flush=True)
                if membrane_ops.BOND_SNAPSHOTS:
                    snaps = membrane_ops.BOND_SNAPSHOTS
                    extra["bond_frames"] = np.asarray([q[0] for q in snaps], np.int32)
                    # ragged: one flat array plus offsets, because the bond count grows with secretion
                    extra["bond_i"] = np.concatenate([q[1] for q in snaps])
                    extra["bond_j"] = np.concatenate([q[2] for q in snaps])
                    extra["bond_s"] = np.concatenate([q[3] for q in snaps])
                    extra["bond_off"] = np.cumsum([0] + [len(q[1]) for q in snaps]).astype(np.int64)
            if "mpm_block" in out.get("sets", {}):
                extra["bpos"] = np.asarray(out["sets"]["mpm_block"]["pos"], np.float32)
                extra["bstress"] = np.asarray(block_ops.BLOCK_STRESS, np.uint8)
                if block_ops.BLOCK_RAW:
                    extra["bvm"] = np.asarray(block_ops.BLOCK_RAW, np.float16)
            np.savez_compressed(os.path.join(out_dir, "traj.npz"),
                                pos=np.asarray(out["sets"]["mpm_particle"]["pos"], np.float32),
                                stress=np.asarray(ecm_ops.STRESS_HISTORY, np.uint8),
                                radius=np.asarray(ecm_ops.BALL_RADIUS, np.float32), **extra)
        except Exception as e:
            print(f"[{name}] traj.npz not written: {type(e).__name__}", flush=True)

    # THE REACTION, SAVED. Written whether or not anything will read it: a later tissue pass is the
    # only place this force can go, and re-running 400 frames of MPM to recover a map that was in
    # memory is the same defect `traj.npz` was added to fix.
    if ecm_ops.PRESSURE_HISTORY:
        try:
            np.savez_compressed(os.path.join(out_dir, "load.npz"),
                                pmap=np.asarray(ecm_ops.PRESSURE_HISTORY, np.float32),
                                scale=np.float32(next(
                                    (o.get("scale", 1.0) for o in spec["operators"]
                                     if o["op"] == "cell_to_ecm"), 1.0)))
        except Exception as e:
            print(f"[{name}] load.npz not written: {type(e).__name__}", flush=True)

    m = measure(out, spec, stress=ecm_ops.STRESS_HISTORY)
    m["wall_s"] = round(wall, 1)
    m["name"] = name
    json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    print(f"[{name}] {wall:.0f}s  contact_frame={m['contact_frame']}  "
          f"strained_frac end={m['strained_frac_end']}  front_r95={m['front_r95_end']}  "
          f"max_disp={m['max_disp']:.3f}  exploded={m['exploded']}", flush=True)

    try:
        render(name, out, spec, out_dir, movie=movie, **(render_kw or {}))
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"[{name}] render FAILED -- traj.npz is on disk, so `rerender('{out_dir}')` "
              f"redraws without re-simulating", flush=True)
    return m


def main():
    import ecm_spec as ES
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--frames", type=int, default=320)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--youngs", type=float, default=40.0)
    ap.add_argument("--substep", type=float, default=2.0e-4)
    ap.add_argument("--cavity-r", type=float, default=0.14)
    ap.add_argument("--cavity-h", type=float, default=0.14)
    ap.add_argument("--align", type=float, default=0.0)
    ap.add_argument("--growth", type=float, default=0.0009)
    ap.add_argument("--k", type=float, default=900.0)
    ap.add_argument("--particles", type=int, default=110000)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()
    spec = ES.build_spec(a.name, n_frames=a.frames, substep_dt=a.substep, youngs=a.youngs,
                         cavity_r=a.cavity_r, cavity_h=a.cavity_h, align=a.align,
                         growth=a.growth, k_contact=a.k, n_particles=a.particles, n_grid=a.grid)
    run(a.name, spec, device=a.device, movie=not a.no_movie)


if __name__ == "__main__":
    main()
