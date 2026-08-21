"""tissue -- PASS 1: run cellfix_B_new itself, and keep everything a matrix or a movie can need.

    from tissue import load_or_build
    T = load_or_build(frames=401, device="cuda:0")      # cached; the 2nd caller pays nothing

WHY THIS IS A MODULE AND NOT A FUNCTION INSIDE THE ECM RUN. Five ECM runs that differ only in the
MATRIX must load the SAME tissue, or the sweep compares two things at once and attributes the
difference to the wrong one. The tissue is therefore built once into
`log/okuda_ECM/_tissue/<name>.npz` and every run reads it. It also means a change of mind about a
colour never costs the vertex model again.

WHAT "COMPLY WITH cellfix_B_new" MEANS HERE, CONCRETELY. The spec is not rewritten, reparameterised
or rescaled: `log/okuda/cellfix_B_new/spec_run.yaml` is loaded verbatim and run by the stock engine,
so the operator stack IS the reference one --

    mesh_seed -> cell_geometry -> cell_grow -> cell_mechanics
                 -> edge_flip -> cell_divide -> topo_record

200 cells at radius 5 in a 50-unit box, growing and dividing to ~3,200 under the 3D AVM shape
energy. Nothing about the epithelium is this experiment's variable; the matrix is.

THE SURFACE THE MATRIX FEELS IS BUILT FROM THE APICAL VERTICES, not from cell centroids. The mesh
vertices ARE the apical surface -- `_draw` makes the basal ring by scaling them inward by 0.82 --
so binning them is binning the boundary itself. MEASURED, because the honest version of this claim
is small: the median apical-vertex radius runs 4.66 -> 15.91 against the cell-centroid radius's
4.60 -> 15.90, i.e. ~1%, since a face centroid is a chord's midpoint on the same surface. What
actually changes is the SAMPLING -- 6,400 vertices instead of 3,200 face centroids, so more of the
angular map is covered by real geometry and less of it by a filled-in row mean, and the max-per-bin
boundary is the outermost vertex rather than the outermost centroid.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "discovery_okuda", "ops"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

LOG = os.path.join(ROOT, "log", "okuda_ECM")
CACHE = os.path.join(LOG, "_tissue")
# THE REFERENCE SPECIFICATION MOVED, AND EVERY CACHED TISSUE HERE WAS BUILT FROM IT. `cellfix_B_new`
# was archived into `_superseded_pre_basis` by the Phase-12 basis refactor; the path here still pointed
# at where it used to be, so `load_or_build` worked on a cache hit and died on a rebuild. The constant
# names where the file IS -- one path, not a search over two.
# THE REFERENCE SPEC MOVED WHEN THE LOOP WAS ARCHIVED, and nothing noticed because the tissue cache
# already existed: `load_or_build` only opens this file when it has to BUILD, so a rebuild -- which is
# exactly what a growth-gated pass 2 needs -- would have died on a path that had been wrong for weeks.
# Both locations are tried and the failure names them.
_SPECS = (os.path.join(ROOT, "log", "okuda", "_archive", "_superseded_pre_basis",
                       "cellfix_B_new", "spec_run.yaml"),
          os.path.join(ROOT, "discovery_okuda", "_archive_runs", "2026-08-03_preclean",
                       "run_records", "cellfix_B_new", "spec_run.yaml"))
CELL_SPEC = next((p for p in _SPECS if os.path.exists(p)), _SPECS[0])

# 32 x 64 rather than 48 x 96. The map has to be resolved by the VERTICES PRESENT, and the opening
# frames have ~1,200 of them: 4,608 bins left two thirds of the sphere empty and filled from a row
# mean, which is a smoother sphere than the tissue and hides the early shape entirely. 2,048 bins
# are covered from frame 0 and still resolve a cell.
N_THETA, N_PHI = 32, 64
RENDER_FRAMES = 200         # frames whose full mesh is kept -- the movie draws exactly these
MESH_PAD = 1.12             # camera headroom, the `run_one.run_box` convention


def apical_map(vp, n_theta=N_THETA, n_phi=N_PHI):
    """Vertex cloud -> R(theta, phi), the FURTHEST apical vertex in each direction.

    Centroid-referenced: `vp` must already have the tissue centroid subtracted. Nothing pins the
    vesicle to the origin, and an origin-referenced radius reads the vesicle's DRIFT as growth --
    the same defect `tube_analysis._cell_centroids` documents for `protr`.
    """
    r = np.linalg.norm(vp, axis=1)
    ok = r > 1e-9
    if ok.sum() < 8:
        return np.zeros((n_theta, n_phi), np.float32)
    u = vp[ok] / r[ok, None]
    th = np.arccos(np.clip(u[:, 2], -1, 1))
    ph = np.arctan2(u[:, 1], u[:, 0]) % (2 * np.pi)
    it = np.clip((th / np.pi * n_theta).astype(int), 0, n_theta - 1)
    ip = np.clip((ph / (2 * np.pi) * n_phi).astype(int), 0, n_phi - 1)
    M = np.zeros((n_theta, n_phi), np.float32)
    np.maximum.at(M, (it, ip), r[ok].astype(np.float32))

    # AN EMPTY BIN IS A GAP IN THE SAMPLING, NOT A HOLE IN THE TISSUE. Left at zero it reads as
    # "the surface is at the centre here", and the matrix flows into a wedge that does not exist.
    for i in range(n_theta):
        row = M[i]
        if (row > 0).any():
            row[row == 0] = row[row > 0].mean()
        else:
            M[i] = M[M > 0].mean() if (M > 0).any() else 0.0
    # ONE SMOOTHING PASS, WRAPPED IN PHI. A max-per-bin map is jagged by construction -- each bin
    # takes its single furthest vertex -- and a jagged boundary makes the contact force a field of
    # spikes rather than a surface pressing. Wrapped in phi because phi is periodic and clamped in
    # theta because the poles are not.
    P = np.pad(M, ((1, 1), (0, 0)), mode="edge")
    P = np.concatenate([P[:, -1:], P, P[:, :1]], axis=1)
    M = sum(P[i:i + n_theta, j:j + n_phi] for i in range(3) for j in range(3)) / 9.0
    return M.astype(np.float32)


def _mesh_of(hist_t, pos_t, centroid):
    """One frame's mesh, centroid-referenced, with the per-cell fields the render colours by."""
    nv = int(hist_t["Nv"])
    d = {"pos": (pos_t[:nv] - centroid).astype(np.float32),
         "nF": np.int32(hist_t["nF"]), "Nv": np.int32(nv)}
    for k in ("E_srce", "E_trgt", "E_face"):
        d[k] = np.asarray(hist_t[k], np.int32)
    # `age` and `ndiv` ARE THE GREEN. cell_divide resets age to 0 on division and ndiv counts the
    # divisions a lineage has had; run_one paints a cell green when `age <= 4 and ndiv > 0`. Both
    # have to travel with the mesh or the ECM movie shows a tissue that never divided -- which is
    # exactly what the reference strip's first two rows are FOR.
    for k in ("myo", "myo_med", "myo_amount"):
        v = hist_t.get(k)
        if v is not None:
            d[k] = np.asarray(v, np.float32)
    for k in ("age", "ndiv"):
        v = hist_t.get(k)
        d[k] = (np.asarray(v, np.float32) if v is not None
                else np.full(int(hist_t["nF"]), np.nan, np.float32))
    return d


def build(frames, device, out_npz, n_render=RENDER_FRAMES, buffer_x=1, plate_gap=None,
          plate_stiff=0.6, load_npz=None, load_gain=1.0, gate_npz=None,
          gate_p_half="auto", gate_hill=4.0, gate_floor=0.15,
          gate_smooth_frames=25, gate_smooth_phi=360.0,
          myosin=None, myo_tau=20.0, myo_beta=1.0, myo_new=1.0,
          myo_keyed_on="length", myo_destabilising=1,
          myo_model="one_pool", myo_k_on=0.05, myo_tau_med=20.0, myo_k_ex=0.05, myo_beta_T=0.0,
          myo_ring=0.0, myo_new_rel=True,
          map_theta=N_THETA, map_phi=N_PHI, op_overrides=None, append_ops=None):
    import t1_ops as _T1
    _T1.T1_TRACE.clear()                       # per build, so a rebuild never inherits a previous run's
    """Run cellfix_B_new verbatim and write the cache.

    `buffer_x` MULTIPLIES THE VERTEX AND CELL RESERVOIRS AND NOTHING ELSE. At the reference buffers
    the run reports, loudly and by design, `RESERVOIR FULL: 1723 division(s) refused for want of
    vertex buffer (6396/6396)` -- division stops around frame 310 of 402 and the tissue coasts to
    the end at a fixed 3,170 cells, so the last ~quarter of every movie shows an epithelium that has
    stopped proliferating because of an ARRAY, not because of its biology. Every mechanical
    parameter is untouched: the reservoir is a memory allocation, so growing it changes what the run
    is ALLOWED to do, not what it is trying to do.

    `plate_gap` (in TISSUE units, the 50-unit world, measured from the world origin) inserts
    `plate_confine` after the force step: two rigid blocks the vesicle cannot grow past. This is the
    one place the tissue stops being cellfix_B_new verbatim, and it is an ADDED boundary condition
    rather than a changed parameter -- the operator stack, the energy and every constant are still the
    reference ones. Runs with plates are cached under their own name for exactly that reason.
    """
    import run_one as R
    S, engine_run = R._lazy_engine()
    # RENAMED UPSTREAM. `tube_analysis` became `tissue_analysis` in the Phase 12 refactor, and this
    # import is the only thing outside discovery_okuda that reached into it. Both names are tried so a
    # checkout on either side of that commit works, and the failure says which module is missing rather
    # than dying inside a cached-tissue path that happened not to need it.
    try:
        from tissue_analysis import _cell_centroids
    except ModuleNotFoundError:
        from tube_analysis import _cell_centroids

    spec = yaml.safe_load(open(CELL_SPEC))
    # THE REFERENCE PARAMETERS, OVERRIDDEN BY NAME. `plate_gap`, `gate_npz` and the rest ADD operators;
    # this changes the ones already there, which is the only way to ask questions like "does the radial
    # term hold a bud back" without forking the spec file. Applied here, before anything is appended,
    # so an override and an added operator cannot fight over the same dict. An operator named in the
    # overrides that is not in the spec is an ERROR and not a no-op: silently ignoring it is how a
    # sweep measures one tissue five times and prints five identical rows.
    for opname, kv in (op_overrides or {}).items():
        hit = [o for o in spec["operators"] if o.get("op") == opname]
        if not hit:
            raise SystemExit(f"[tissue] op_overrides names '{opname}', which is not in {CELL_SPEC}; "
                             f"the spec has {sorted({o['op'] for o in spec['operators']})}")
        for o in hit:
            o.update(kv)
        print(f"[tissue] {opname}: " + ", ".join(f"{k}={v}" for k, v in kv.items()), flush=True)
    for entry in (append_ops or []):
        # AN OPERATOR THAT IS NOT IMPORTED IS NOT REGISTERED, and the engine's failure for an unknown
        # op name is not obviously "you forgot an import". The entry names its own module.
        if entry.get("module"):
            import importlib
            importlib.import_module(entry["module"])
        op, after = dict(entry["op"]), entry.get("after")
        spec["operators"].append(op)
        i = (spec["schedule"].index(after) + 1) if after in spec["schedule"] else len(spec["schedule"])
        spec["schedule"].insert(i, op["op"])
        print(f"[tissue] + {op['op']} after {after}", flush=True)
    spec["general"]["n_frames"] = int(frames)
    spec["general"]["name"] = "cellfix_B_new_for_ecm"
    if buffer_x != 1:
        for st, key in (("cell", "n"), ("vertex", "n")):
            spec["sets"][st][key] = int(spec["sets"][st][key] * buffer_x)
        print(f"[tissue] reservoirs x{buffer_x}: cell {spec['sets']['cell']['n']}, "
              f"vertex {spec['sets']['vertex']['n']}", flush=True)
    # UNIQUE PER CACHE, because two builds can run at once (one per GPU) and a shared temp path
    # would have each write the other's spec out from under it.
    if plate_gap is not None:
        # AFTER `cell_mechanics`, BEFORE the topology ops. The relaxation must be allowed to push
        # into the plate first -- that push IS the pressure the confinement is resisting -- and the
        # projection then takes it back out. Placed before cell_divide/edge_flip so those never
        # see a vertex outside the domain.
        import plate_ops                                            # noqa: F401  register it
        spec["operators"].append({"op": "plate_confine", "at": "vertex", "axis": 2,
                                  "centre": 0.0, "gap_half": float(plate_gap),
                                  "stiff": float(plate_stiff)})
        i = spec["schedule"].index("cell_mechanics") + 1
        spec["schedule"].insert(i, "plate_confine")
        print(f"[tissue] rigid plates at z = +/-{plate_gap:.3g} tissue units "
              f"(stiff {plate_stiff})", flush=True)
    if myosin is not None:
        # BEFORE `cell_mechanics`, because the energy reads `m["myo"]` in the same frame it is written --
        # after it, the relaxation would always use the previous frame's myosin, which for a quantity with
        # a 20-frame timescale is a lag nobody would notice and everybody would inherit.
        import junction_ops                                           # noqa: F401  register it
        # THE SAME K_P / Lambda / Gamma `cell_mechanics` WAS GIVEN, read off that operator rather than
        # restated. `keyed_on="tension"` computes dE/dl_e, and if these three disagree with the energy
        # actually being minimised it is the tension of a different tissue -- a discrepancy that would
        # produce plausible numbers and no error.
        _se = next((o for o in spec["operators"] if o["op"] == "cell_mechanics"), {})
        _en = {k: float(_se.get(k, d)) for k, d in (("K_P", 1.0), ("Lam", 0.0), ("Gam", 0.0))}
        if str(myo_model) == "two_pool":
            # THE SECOND POOL, AND ITS OWN OPERATOR BEFORE THE BELT'S. `medioapical_myosin` acts on the
            # `cell` set and hands a flux to the junctions; the belt integrates it. Scheduled in that
            # order so the flux a frame delivers is the flux that frame's cells produced -- reversed,
            # the belt would always be integrating the previous frame's supply, which is a lag of one
            # frame on a quantity whose whole timescale is twenty.
            import medioapical_ops                                    # noqa: F401  register both
            spec["operators"].append({"op": "medioapical_myosin", "at": "cell", "mesh_at": "vertex",
                                      "k_on": float(myo_k_on), "tau_med": float(myo_tau_med),
                                      "k_ex": float(myo_k_ex), "beta_T": float(myo_beta_T),
                                      "rho0": 1.0, "dt": 1.0,
                                      "k_perim": _en["K_P"], "lam": _en["Lam"], "gam": _en["Gam"]})
            spec["operators"].append({"op": "junction_myosin", "model": "two_pool", "at": "vertex",
                                      "activity": float(myosin), "tau_jun": float(myo_tau),
                                      "myo_new": float(myo_new), "dt": 1.0, "inherit": True,
                                      "myo_new_rel": bool(myo_new_rel)})
            i = spec["schedule"].index("cell_mechanics")
            spec["schedule"].insert(i, "junction_myosin")
            spec["schedule"].insert(i, "medioapical_myosin")
        else:
            spec["operators"].append({"op": "junction_myosin", "at": "vertex",
                                      "k_perim": _en["K_P"], "lam": _en["Lam"], "gam": _en["Gam"],
                                      "activity": float(myosin), "tau": float(myo_tau),
                                      "beta": float(myo_beta), "myo_new": float(myo_new), "dt": 1.0,
                                      "inherit": True, "keyed_on": str(myo_keyed_on),
                                      "destabilising": bool(myo_destabilising)})
            i = spec["schedule"].index("cell_mechanics")
            spec["schedule"].insert(i, "junction_myosin")
        # AND THE CARRY, AFTER THE TOPOLOGY OPERATORS. `edge_flip` rewires the half-edge arrays
        # and `cell_divide` lengthens them, both AFTER `junction_myosin` has written `m["myo"]` for the
        # arrays as they were -- so what `topo_record` records is this frame's edges beside last
        # writing's myosin. Measured on the 401-frame nominal: 56 of 200 snapshots carried a myosin
        # array 6 to 1356 entries short of the edge arrays, and every reader indexes it positionally.
        # This operator re-keys it by vertex pair; it is scheduled last of the three so it sees the
        # final topology of the frame. See junction_ops.JunctionMyosinSync for why it cannot perturb
        # the trajectory.
        if str(myo_model) == "two_pool" and float(myo_ring) > 0.0:
            # AFTER THE TOPOLOGY OPERATORS AND BEFORE THE SYNC. The ring needs the new vertices to
            # exist (so, after `cell_divide`) and needs `myo_vseen` from the previous frame to tell them
            # from old ones (so, before the next `junction_myosin`); putting it before the sync is what
            # makes the deposit appear in the frame the division happened rather than the one after.
            spec["operators"].append({"op": "cytokinetic_ring", "at": "vertex",
                                      "ring": float(myo_ring), "tau_jun": float(myo_tau),
                                      "debit": True})
            k = max(spec["schedule"].index(o) for o in ("edge_flip", "cell_divide")
                    if o in spec["schedule"])
            spec["schedule"].insert(k + 1, "cytokinetic_ring")
        spec["operators"].append({"op": "junction_sync", "at": "vertex",
                                  "myo_new": float(myo_new), "inherit": True})
        j = max(spec["schedule"].index(o) for o in ("edge_flip", "cell_divide",
                                                    "cytokinetic_ring")
                if o in spec["schedule"])
        spec["schedule"].insert(j + 1, "junction_sync")
        print(f"[tissue] per-junction myosin: activity={myosin}, tau={myo_tau}, beta={myo_beta}, "
              f"myo_new={myo_new}; re-keyed after {spec['schedule'][j]}", flush=True)
    if gate_npz is not None:
        # AFTER `cell_grow`, whose per-cell increment it corrects, and BEFORE the force step
        # and the topology ops -- so `cell_mechanics` relaxes toward the GATED targets and `cell_divide`
        # tests a volume that grew at the gated rate. Placed anywhere later and the frame's mechanics
        # would already have been solved for the ungated targets.
        import load_ops                                               # noqa: F401  register it
        spec["operators"].append({"op": "ecm_gate_growth", "at": "vertex",
                                  "load": str(gate_npz), "p_half": gate_p_half,
                                  "hill": float(gate_hill), "floor": float(gate_floor),
                                  "smooth_frames": int(gate_smooth_frames),
                                  "smooth_phi_deg": float(gate_smooth_phi)})
        i = spec["schedule"].index("cell_grow") + 1
        spec["schedule"].insert(i, "ecm_gate_growth")
        print(f"[tissue] ECM-stress growth gate from {os.path.basename(str(gate_npz))} "
              f"(p_half {gate_p_half}, hill {gate_hill}, floor {gate_floor})", flush=True)
    if load_npz is not None:
        # THE MATRIX PUSHING BACK, from a pressure map a previous pass 2 recorded. Same slot as the
        # plates -- after the relaxation, before the topology ops -- for the same reason: the force
        # step has to be allowed to push outward first, because that push is what the load resists.
        import load_ops                                              # noqa: F401  register it
        spec["operators"].append({"op": "ecm_load", "at": "vertex",
                                  "load": str(load_npz), "gain": float(load_gain),
                                  "mu": 1.0, "dt": 1.0, "cap_frac": 0.04})
        i = spec["schedule"].index("cell_mechanics") + 1
        spec["schedule"].insert(i, "ecm_load")
        print(f"[tissue] matrix load from {os.path.basename(str(load_npz))}, gain {load_gain}",
              flush=True)
    p = os.path.join("/tmp", os.path.basename(out_npz).replace(".npz", "") + ".yaml")
    open(p, "w").write(yaml.safe_dump(spec, sort_keys=False))
    t0 = time.time()
    H, out = engine_run(S.load(p), device=device)
    hist = (H.level("vertex")._mesh or {}).get("hist", [])
    if not hist:
        raise RuntimeError("no topo_snapshot history -- the tissue cannot be drawn or coupled")
    posf = out["sets"]["vertex"]["pos"]
    T = min(posf.shape[0], len(hist))

    maps, r_ap, r_med, ncell, cent, r_eq, r_ax, r_xyz = [], [], [], [], [], [], [], []
    for t in range(T):
        mt = hist[t]
        nv = int(mt["Nv"])
        vp = posf[t][:nv].astype(np.float64)
        cen, rad, live = _cell_centroids(vp, mt)
        c = vp.mean(0)                                    # the vertex centroid: what `_draw` centres on
        v = vp - c
        maps.append(apical_map(v, map_theta, map_phi))
        r_ap.append(float(np.median(np.linalg.norm(v, axis=1))))
        r_med.append(float(np.median(rad[live])) if live.any() else 0.0)
        ncell.append(int(mt["nF"]))
        cent.append(c)
        # THE TWO SEMI-AXES, so "it became an ovoid" is a number and not an impression. 98th
        # percentile rather than max: one stray vertex should not set the shape of the tissue, and a
        # just-divided sliver can sit briefly outside the surface.
        r_eq.append(float(np.percentile(np.hypot(v[:, 0], v[:, 1]), 98)))
        r_ax.append(float(np.percentile(np.abs(v[:, 2]), 98)))
        # AND THE THREE AXES SEPARATELY. `r_eq` pools x and y, so a tissue that grew along y and not
        # along x -- which is exactly what an anisotropic matrix should produce -- has the same r_eq as
        # a round one. The quantity the experiment is about would be invisible to its own metric.
        r_xyz.append([float(np.percentile(np.abs(v[:, k]), 98)) for k in range(3)])

    keep = np.unique(np.linspace(0, T - 1, min(n_render, T)).astype(int))
    mesh = {"mesh_frames": keep.astype(np.int32)}
    extent = 0.0
    for j, t in enumerate(keep):
        d = _mesh_of(hist[int(t)], posf[int(t)].astype(np.float64), cent[int(t)])
        extent = max(extent, float(np.abs(d["pos"]).max()))
        for k, v in d.items():
            mesh[f"m{j}_{k}"] = v

    M = np.stack(maps)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz, smap=M, r_apical=np.asarray(r_ap, np.float32),
        r_med=np.asarray(r_med, np.float32), n_cells=np.asarray(ncell, np.int32),
        centroid=np.asarray(cent, np.float32),
        # ONE camera half-width for the WHOLE run, measured over every kept frame. Per-frame
        # autofit is what hid growth in every archived movie until run_one.run_box was written:
        # a vesicle that doubles in radius renders at constant apparent size.
        Lbox=np.float32(extent * MESH_PAD),
        r_eq=np.asarray(r_eq, np.float32), r_ax=np.asarray(r_ax, np.float32),
        r_xyz=np.asarray(r_xyz, np.float32),
        plate_gap=np.float32(-1.0 if plate_gap is None else plate_gap),
        # T1 FLIPS, SAVED WITH THE TISSUE. The trace is a module global that only fills while pass 1 is
        # actually running, so on a cache hit every downstream run reported ZERO T1s -- the observable
        # that is supposed to discriminate the two myosin laws, silently zeroed by the cache.
        t1_trace=(np.asarray(_T1.T1_TRACE, np.int64) if _T1.T1_TRACE
                  else np.zeros((0, 4), np.int64)), **mesh)
    ar = r_eq[-1] / max(r_ax[-1], 1e-9)
    ax3 = r_xyz[-1]
    print(f"[tissue] semi-axes x/y/z = {ax3[0]:.2f} / {ax3[1]:.2f} / {ax3[2]:.2f}  "
          f"(in-plane x:y = {ax3[0] / max(ax3[1], 1e-9):.3f})", flush=True)
    print(f"[tissue] cellfix_B_new: {T} frames, {ncell[0]} -> {ncell[-1]} cells, "
          f"apical radius {r_ap[0]:.2f} -> {r_ap[-1]:.2f} (cell-centroid radius "
          f"{r_med[0]:.2f} -> {r_med[-1]:.2f}), semi-axes equatorial {r_eq[-1]:.2f} / axial "
          f"{r_ax[-1]:.2f} = ASPECT {ar:.2f}, {len(keep)} meshes kept, "
          f"{time.time()-t0:.0f}s -> {os.path.relpath(out_npz, ROOT)}", flush=True)
    return out_npz


def load_or_build(frames=401, device="cuda:0", name="cellfix_B_new", rebuild=False,
                  buffer_x=1, plate_gap=None, plate_stiff=0.6, load_npz=None,
                  load_gain=1.0, tag_extra="", gate_npz=None, gate_p_half="auto",
                  gate_hill=4.0, gate_floor=0.15, gate_smooth_frames=25,
                  gate_smooth_phi=360.0, myosin=None, myo_tau=20.0, myo_beta=1.0,
                  myo_keyed_on="length", myo_destabilising=1,
                  myo_model="one_pool", myo_k_on=0.05, myo_tau_med=20.0, myo_k_ex=0.05,
                  myo_beta_T=0.0, myo_ring=0.0, myo_new_rel=True,
                  myo_new=1.0, map_theta=N_THETA, map_phi=N_PHI,
                  op_overrides=None, append_ops=None):
    """The cache path, built if missing. Frames are part of the filename: a 401-frame tissue and a
    120-frame one are different tissues, and silently reusing one for the other would be a run
    whose movie stops before the thing it was testing happened."""
    tag = f"{name}_f{int(frames)}" + (f"_x{int(buffer_x)}" if buffer_x != 1 else "")
    if plate_gap is not None:
        tag += f"_plate{plate_gap:g}".replace(".", "p")
    # THE CACHE KEY IS DERIVED FROM EVERY INPUT, NOT ASSEMBLED BY THE CALLER. This is the second time
    # the same bug shipped. First the key carried the gate's parameters but not WHICH pressure map they
    # applied to, so the caps+plane run silently loaded the caps-only tissue and reported its semi-axes
    # to three decimals -- a convincing null. Then the fix added the map and still left the PARAMETERS to
    # a caller-supplied `tag_extra`, so a p_half sweep that forgot to pass one measured a single tissue
    # three times and printed three identical rows. A key built by hand is a key that will be wrong
    # again, so it is now a hash over the whole configuration: any argument that reaches `build` and
    # changes the result is in it by construction.
    import hashlib
    cfg = {"plate_gap": plate_gap, "plate_stiff": plate_stiff,
           "gate": None, "load": None,
           "gate_p_half": gate_p_half, "gate_hill": gate_hill, "gate_floor": gate_floor,
           "gate_smooth_frames": gate_smooth_frames, "gate_smooth_phi": gate_smooth_phi,
           "load_gain": load_gain, "myosin": myosin, "myo_tau": myo_tau, "myo_beta": myo_beta, "myo_inherit": 1,
           "myo_keyed_on": myo_keyed_on, "myo_destabilising": myo_destabilising,
           "myo_new": myo_new}
    # ADDED ONLY WHEN IT IS NOT THE DEFAULT, for the reason the map resolution is: unconditionally it
    # rehashes every key ever written and the next run finds no cache, tries to rebuild, and dies on a
    # pass-1 spec that has moved. `one_pool` therefore hashes exactly as it did before this model
    # existed, and every archived tissue still loads.
    if str(myo_model) != "one_pool":
        cfg.update({"myo_model": myo_model, "myo_k_on": myo_k_on, "myo_tau_med": myo_tau_med,
                    "myo_k_ex": myo_k_ex, "myo_beta_T": myo_beta_T})
    if float(myo_ring) != 0.0:
        cfg["myo_ring"] = myo_ring
    # ADDED WHEN TRUE, WHICH IS THE OPPOSITE OF THE CONVENTION ABOVE AND DELIBERATE. `myo_new` used to
    # be an absolute line density and every two-pool cache on disk was built that way, so it is the
    # LEGACY value that must keep its key: adding the flag only for the corrected reading leaves those
    # caches loadable and reproducible while the new behaviour gets a key of its own.
    if str(myo_model) == "two_pool" and bool(myo_new_rel):
        cfg["myo_new_rel"] = True
    # THE SURFACE MAP'S RESOLUTION IS IN THE KEY, because the map IS the epithelium as far as every
    # membrane operator is concerned -- `integrin_adhesion`, `bm_contact`,
    # `adhesion_pull` and `surface_track` read it and nothing else. At 32x64 a bin is 1.63 x 1.63
    # tissue units at the end of the run, about 2x2 cells, so the anchor a fibre pulls toward is a
    # staircase two cells wide and neighbouring integrins share a value while the surface under them
    # differs by 0.09 (median) to 0.18 (p90) tissue units -- most of a fibre's 0.223 length.
    #
    # ADDED ONLY WHEN IT IS NOT THE DEFAULT, and that is not tidiness. Put in unconditionally it
    # changes the hash of every key ever written, so the next run finds no cache, tries to rebuild, and
    # dies on a pass-1 spec that is no longer on disk -- which is exactly what happened at 18:35.
    if (map_theta, map_phi) != (N_THETA, N_PHI):
        cfg["map_theta"], cfg["map_phi"] = map_theta, map_phi
    # ADDED ONLY WHEN PRESENT, for the reason every other conditional key above is: unconditionally it
    # rehashes every cache ever written. Two tissues that differ in one operator parameter MUST differ
    # in their key, which is what this is for.
    if op_overrides:
        cfg["op_overrides"] = repr(sorted((k, sorted(v.items())) for k, v in op_overrides.items()))
    if append_ops:
        cfg["append_ops"] = repr(append_ops)
    for key, path in (("gate", gate_npz), ("load", load_npz)):
        if path is not None:
            sz = os.path.getsize(path) if os.path.exists(path) else 0
            cfg[key] = f"{os.path.abspath(path)}:{sz}"
    if any(v is not None for v in (plate_gap, gate_npz, load_npz, myosin)) \
            or (map_theta, map_phi) != (N_THETA, N_PHI) or op_overrides or append_ops:
        tag += "_" + hashlib.sha1(repr(sorted(cfg.items())).encode()).hexdigest()[:10]
    tag += tag_extra
    out = os.path.join(CACHE, f"{tag}.npz")
    if rebuild or not os.path.exists(out):
        build(frames, device, out, buffer_x=buffer_x, plate_gap=plate_gap,
              plate_stiff=plate_stiff, load_npz=load_npz, load_gain=load_gain,
              gate_npz=gate_npz, gate_p_half=gate_p_half, gate_hill=gate_hill,
              gate_floor=gate_floor, gate_smooth_frames=gate_smooth_frames,
              gate_smooth_phi=gate_smooth_phi, myosin=myosin, myo_tau=myo_tau,
              myo_beta=myo_beta, myo_new=myo_new, myo_keyed_on=myo_keyed_on,
              myo_destabilising=myo_destabilising, myo_model=myo_model, myo_k_on=myo_k_on,
              myo_tau_med=myo_tau_med, myo_k_ex=myo_k_ex, myo_beta_T=myo_beta_T,
              myo_ring=myo_ring, myo_new_rel=myo_new_rel,
              map_theta=map_theta, map_phi=map_phi,
              op_overrides=op_overrides, append_ops=append_ops)
    else:
        z = np.load(out)
        print(f"[tissue] reusing {os.path.relpath(out, ROOT)}  "
              f"({z['smap'].shape[0]} frames, {int(z['n_cells'][-1])} cells, "
              f"apical r {float(z['r_apical'][0]):.2f} -> {float(z['r_apical'][-1]):.2f})",
              flush=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=401)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--buffer-x", type=int, default=1)
    ap.add_argument("--plate-gap", type=float, default=None,
                    help="rigid plates at z = +/- this, in TISSUE units (unconfined r_eq ~16.5)")
    ap.add_argument("--plate-stiff", type=float, default=0.6)
    a = ap.parse_args()
    load_or_build(a.frames, a.device, rebuild=a.rebuild, buffer_x=a.buffer_x,
                  plate_gap=a.plate_gap, plate_stiff=a.plate_stiff)
