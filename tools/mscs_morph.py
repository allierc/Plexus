"""MscS closed -> open with the MPM morph trainer (exp04, from 2026-09-29): the preparation and the ruler.

    PYTHONPATH=src python tools/mscs_morph.py prep            # shapes + model + training spec
    PYTHONPATH=src python tools/mscs_morph.py score <name>    # a trained run's path, measured against the open structure

THE PARTICLES ARE THE CLOSED STRUCTURE'S ATOMS, the target is the open structure's surface. 6PWN (closed, nanodisc,
residues 1-280) and 2VV5 (open, residues 25-281) are read from their OPM files (membrane normal z, centre at z = 0);
the open heptamer's chains are matched to the closed one's by the cyclic order that best superposes the cytoplasmic
cage (residues 129-280, which barely moves on gating), and the open structure is superposed onto the closed one on
that cage. Only atoms present in both are kept (same chain, residue number and atom name, residues 25-280): every
particle then has its exact open position, which the trainer never sees (it scores grid mass against the open
surface) and this ruler uses to say how RIGHT the learned motion is, not only how close its outline is.

THE RULER (`score`): the trained run is re-run with its fitted rate field, every frame kept; per frame
    rmsd_open        RMS distance of every atom to its own open position (nm) -- 0 when the motion is exactly right
    rmsd_open_tm_ca  the same for the transmembrane C-alphas (residues 25-128), where the gate is
    pore_r_min       the narrowest free radius of the pore in the membrane core, from the atoms (nm): at each height
                     the distance from the axis to the nearest atom within 0.3 nm of it, minus 0.15 nm (an atom's
                     radius); closed 6PWN and open 2VV5 give the two ends
    helix_ok         the fraction of C-alpha pairs (i, i+4) in TM1-TM3 whose distance stays within 0.1 nm of the
                     range the two deposited states span (both are real helices, and the open one kinks TM3: against
                     the closed state alone it scored 0.88) -- a helix bent or stretched beyond either fails it
and the same numbers for the straight path x = x_closed + lambda (x_open - x_closed) at the same lambda (the path
`morph_gate` uses), so the two paths can be set side by side.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = os.path.join(ROOT, "experiments", "exp04_membrane_channels", "data", "opm")
SHAPE = "mscs_morph"                     # graphs_data/shapes/mscs_morph: the atoms (closed, open) and labels
TARGET = "mscs_open_surface"             # graphs_data/shapes/mscs_open_surface: the open structure's surface
BOX_NM = 18.0                            # the world's side (0.35 world units) in nm
WORLD = 0.35
TM = (25, 128)                           # TM1-TM3 residues
CAGE = (129, 280)


def read_pdb(path):
    """{(chain, resnum, atom): xyz nm} for the protein's heavy atoms (ATOM records; OPM's DUM planes are HETATM)."""
    out = {}
    for l in open(path):
        if not l.startswith("ATOM"):
            continue
        name, el = l[12:16].strip(), (l[76:78].strip() or l[12:16].strip()[0])
        if el == "H":
            continue
        key = (l[21], int(l[22:26]), name)
        if key not in out:                                       # first altloc
            out[key] = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]) / 10.0
    return out


def kabsch(P, Q):
    """R, t with R @ P_i + t ~ Q_i (least squares)."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def prep(a):
    import pyvista as pv
    from skimage import measure
    C = read_pdb(os.path.join(DATA, "6pwn.pdb"))
    O = read_pdb(os.path.join(DATA, "2vv5.pdb"))
    cc, oc = sorted({k[0] for k in C}), sorted({k[0] for k in O})
    assert len(cc) == len(oc) == 7, (cc, oc)
    # THE CHAIN MAPPING: the cyclic order (either direction) that best superposes the cage C-alphas
    best = None
    for step in (1, -1):
        for s in range(7):
            mp = {oc[(s + step * i) % 7]: cc[i] for i in range(7)}
            P, Q = [], []
            for (ch, r, nm), x in O.items():
                if nm == "CA" and CAGE[0] <= r <= CAGE[1] and (mp[ch], r, "CA") in C:
                    P.append(x); Q.append(C[(mp[ch], r, "CA")])
            P, Q = np.array(P), np.array(Q)
            R, t = kabsch(P, Q)
            rms = float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))
            if best is None or rms < best[0]:
                best = (rms, mp, R, t)
    rms, mp, R, t = best
    print(f"[prep] open chains -> closed chains {mp}; cage C-alpha RMSD after superposition {rms:.3f} nm")
    keys = sorted(k for k in C if TM[0] <= k[1] <= CAGE[1] and (next(o for o in mp if mp[o] == k[0]), k[1], k[2]) in O)
    inv = {v: k for k, v in mp.items()}
    Xc = np.array([C[k] for k in keys])
    Xo = np.array([O[(inv[k[0]], k[1], k[2])] for k in keys]) @ R.T + t
    ctr = 0.5 * (Xc.min(0) + Xc.max(0))                          # the closed structure's box centre -> the world's
    print(f"[prep] {len(keys)} atoms in both (residues {TM[0]}-{CAGE[1]}); closed extent "
          f"{np.round(Xc.max(0) - Xc.min(0), 2)} nm, open {np.round(Xo.max(0) - Xo.min(0), 2)} nm")
    from plexus.paths import graphs_data_path
    sd = os.path.join(graphs_data_path(), "shapes", SHAPE)
    os.makedirs(sd, exist_ok=True)
    np.savez_compressed(os.path.join(sd, "points.npz"), closed=((Xc - ctr) * 1e-9).astype(np.float64),
                        open=((Xo - ctr) * 1e-9).astype(np.float64))
    json.dump({"atoms": [[k[0], k[1], k[2]] for k in keys], "centre_nm": ctr.tolist(), "box_nm": BOX_NM,
               "chain_map_open_to_closed": mp, "cage_rmsd_nm": rms}, open(os.path.join(sd, "atoms.json"), "w"))
    open(os.path.join(sd, "PROVENANCE.md"), "w").write(
        "MscS (E. coli) atoms for the exp04 morph test: `closed` = 6PWN (Reddy et al. 2019, nanodisc), `open` = 2VV5\n"
        "(Wang et al. 2008), both from OPM, the open superposed on the closed on the cytoplasmic cage (129-280);\n"
        "only atoms in both (residues 25-280). Metres, centred on the closed structure's box. tools/mscs_morph.py.\n")
    # THE TARGET SURFACE: a Gaussian density of the open atoms (sigma 0.15 nm on a 0.1 nm grid), cut at the level
    # whose enclosed volume is the atoms' own (0.0155 nm^3 per heavy atom, a protein's packing), in WORLD units
    s_w = WORLD / BOX_NM
    Xw = (Xo - ctr) * s_w + WORLD / 2
    h = 0.1 * s_w
    lo, hi = Xw.min(0) - 1.0 * s_w, Xw.max(0) + 1.0 * s_w
    shape = np.ceil((hi - lo) / h).astype(int) + 1
    rho = np.zeros(shape)
    idx = np.round((Xw - lo) / h).astype(int)
    np.add.at(rho, tuple(idx.T), 1.0)
    from scipy.ndimage import gaussian_filter
    rho = gaussian_filter(rho, 0.15 / 0.1)
    v_target = len(Xo) * 0.0155 * s_w ** 3
    levels = np.quantile(rho[rho > 0], np.linspace(0.05, 0.95, 91))
    lev = min(levels, key=lambda L: abs(float((rho > L).sum()) * h ** 3 - v_target))
    verts, faces, _n, _v = measure.marching_cubes(rho, float(lev), spacing=(h, h, h))
    verts = verts + lo
    mesh = pv.PolyData(verts, np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)).clean()
    td = os.path.join(graphs_data_path(), "shapes", TARGET)
    os.makedirs(td, exist_ok=True)
    mesh.save(os.path.join(td, "source.obj")) if hasattr(mesh, "save") else None
    pv.save_meshio(os.path.join(td, "source.obj"), mesh) if not os.path.exists(os.path.join(td, "source.obj")) else None
    open(os.path.join(td, "PROVENANCE.md"), "w").write(
        "MscS open (2VV5) surface for the exp04 morph target, in the morph model's WORLD units (0.35 = 18 nm):\n"
        "a Gaussian density of the atoms of graphs_data/shapes/mscs_morph (sigma 0.15 nm), cut where its volume is\n"
        "0.0155 nm^3 per heavy atom. tools/mscs_morph.py prep.\n")
    b = np.array(mesh.bounds).reshape(3, 2)
    size, centre = float((b[:, 1] - b[:, 0]).max()), (0.5 * (b[:, 0] + b[:, 1])).tolist()
    print(f"[prep] target surface: {mesh.n_points} vertices, volume {mesh.volume / s_w ** 3:.0f} nm^3 "
          f"(atoms' own {v_target / s_w ** 3:.0f}); size {size:.4f} world, centre {np.round(centre, 4).tolist()}")
    # THE MODEL: the morph toy's material, seeded with the closed atoms
    n = len(Xc)
    bx = [float(v) for v in ((Xc.min(0) - ctr) * s_w + WORLD / 2)] + [float(v) for v in ((Xc.max(0) - ctr) * s_w + WORLD / 2)]
    model = {
        "general": {"name": "mscs_morph", "seed": 0, "n_frames": 20, "dt": 0.002, "record_cap": 3, "boundary": "wall",
                    "dim": 3, "world": [WORLD] * 3, "units": {"length_um": BOX_NM * 1e-3 / WORLD}},
        "sets": {"mpm_particle": {"n": n, "state": {
            "pos": {"width": 3, "integration": "second_order_coordinate", "boundary": "world", "role": "coordinate"},
            "vel": {"width": 3, "integration": "second_order_rate", "role": "rate"},
            "rate": {"width": 6, "integration": "none", "boundary": "free"}},
            "types": {"protein": {"fraction": 1.0, "youngs": 90.0, "density": 1.0, "block": bx}}}},
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": 48}},
        "seed": [{"op": "cloud_seed", "at": "mpm_particle", "cloud": f"{SHAPE}/closed", "origin": [WORLD / 2] * 3,
                  "scale": 1.0 / (BOX_NM * 1e-9 / WORLD)}],
        "operators": [{"op": "mpm_strain", "at": "mpm_particle"},
                      {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid", "drag": 0.5, "a_max": 200.0},
                      {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 0.9},
                      {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 0.9, "vmax": 1.0e9},
                      {"op": "deform_control", "at": "mpm_particle", "implementation": "block", "block": "rate",
                       "after_frame": 1}],
        "schedule": [{"substep_dt": 0.00034, "compile": True,
                      "steps": ["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"]}, "deform_control"],
        "plotting": {}}
    import yaml
    mp_ = os.path.join(ROOT, "config", "si_material", "mscs_morph.yaml")
    yaml.safe_dump(model, open(mp_, "w"), sort_keys=False, default_flow_style=None, width=110)
    half = float(max(bx[3] - bx[0], bx[4] - bx[1], bx[5] - bx[2]) / 2 + 1.0 * s_w)
    cen = [0.5 * (bx[i] + bx[i + 3]) for i in range(3)]
    train = {"name": a.name, "model": "config/si_material/mscs_morph.yaml",
             "learnable": [{"block": "rate", "of": "mpm_particle", "with": "lattice", "over": "material", "K": 12,
                            "extent": [round(c, 5) for c in cen] + [round(half, 5)]}],
             "task": {"reference": {"shape": TARGET, "size": round(size, 6), "centre": [round(c, 6) for c in centre]},
                      "observe": {"set": "mpm_particle", "block": "pos", "measure": "grid_mass", "grid": "mpm_grid"},
                      "loss": "log_mse"},
             "training": {"optimizer": "adam", "lr": 3.0, "lr_min": 0.15, "schedule": "cosine", "seed": 0,
                          "stages": [{"resolution": {"mpm_grid": 32}, "iters": 80},
                                     {"resolution": {"mpm_grid": 48}, "iters": 80},
                                     {"resolution": {"mpm_grid": 64}, "iters": 80}],
                          "render": {"frames": 200, "px": 1100}}}
    tp = os.path.join(ROOT, "config", "training", "morph", f"{a.name}.yaml")
    yaml.safe_dump(train, open(tp, "w"), sort_keys=False, default_flow_style=None, width=110)
    print(f"[prep] model {os.path.relpath(mp_, ROOT)} ({n} particles); training {os.path.relpath(tp, ROOT)}; "
          f"grid 64 over {BOX_NM} nm = {BOX_NM / 64:.2f} nm a cell")


# ---- the ruler -----------------------------------------------------------------------------------------------------
def _labels():
    from plexus.paths import graphs_data_path
    sd = os.path.join(graphs_data_path(), "shapes", SHAPE)
    z = np.load(os.path.join(sd, "points.npz"))
    info = json.load(open(os.path.join(sd, "atoms.json")))
    return z["closed"] * 1e9, z["open"] * 1e9, info


def measures(X, Xc, Xo, atoms, z_core=(-1.6, 1.6)):
    """The four numbers for one conformation X (nm, the closed structure's centred frame)."""
    res = np.array([a[1] for a in atoms]); name = np.array([a[2] for a in atoms]); ch = np.array([a[0] for a in atoms])
    ca = name == "CA"; tm = (res >= TM[0]) & (res <= TM[1])
    out = {"rmsd_open": float(np.sqrt(((X - Xo) ** 2).sum(1).mean())),
           "rmsd_open_tm_ca": float(np.sqrt(((X[ca & tm] - Xo[ca & tm]) ** 2).sum(1).mean()))}
    # the pore: the axis through the TM C-alphas' centre; heights in the membrane core, in the closed frame's z
    return out | _pore_and_helix(X, Xc, Xo, res, ca, tm, ch, z_core)


def _pore_and_helix(X, Xc, Xo_, res, ca, tm, ch, z_core):
    axis_xy = X[ca & tm][:, :2].mean(0)
    rs = []
    zc0 = float(X[ca & tm][:, 2].mean())
    for zz in np.arange(zc0 + z_core[0], zc0 + z_core[1], 0.1):
        m = np.abs(X[:, 2] - zz) < 0.3
        if m.any():
            rs.append(float(np.hypot(*(X[m][:, :2] - axis_xy).T).min()) - 0.15)
    ok, tot = 0, 0
    for c_ in np.unique(ch):
        idx = {int(r): i for i, (r, is_ca, c2, t) in enumerate(zip(res, ca, ch, tm)) if is_ca and c2 == c_ and t}
        for r, i in idx.items():
            j = idx.get(r + 4)
            if j is not None:
                tot += 1
                d, dc, do = (np.linalg.norm(Y[i] - Y[j]) for Y in (X, Xc, Xo_))
                ok += (min(dc, do) - 0.1) <= d <= (max(dc, do) + 0.1)
    # STERIC CLASHES: heavy-atom pairs closer than 0.25 nm that are not in the same or neighbouring residues of one
    # chain -- a path that walks helices through each other (the straight line's risk) shows here
    from scipy.spatial import cKDTree
    pr = cKDTree(X).query_pairs(0.25, output_type="ndarray")
    same = (ch[pr[:, 0]] == ch[pr[:, 1]]) & (np.abs(res[pr[:, 0]] - res[pr[:, 1]]) <= 1)
    return {"pore_r_min": round(min(rs), 3) if rs else None, "helix_ok": round(ok / max(tot, 1), 3),
            "clashes": int((~same).sum())}


def score(a):
    import torch
    from plexus import engine, trainer as T
    Xc, Xo, info = _labels()
    atoms = info["atoms"]
    print("closed:", measures(Xc, Xc, Xo, atoms))
    print("open:  ", measures(Xo, Xc, Xo, atoms))
    spec = T.load(a.name)
    out = T.out_dir(spec)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=a.device)
    learn = T.Learnables(spec["learnable"], a.device)
    learn.restore(ck["fitted"])
    engine.quiet(True)
    sim = T._model(spec, train=True, resolution=T._last_resolution(spec))
    frames = []
    with torch.no_grad():
        T._shape_rollout(sim, learn, "mpm_particle", a.device, grad=False, keep=frames)
    s_w = WORLD / BOX_NM
    rows = []
    X0 = (frames[0] - WORLD / 2) / s_w
    for k, Xw in enumerate(frames):
        X = (Xw - WORLD / 2) / s_w                               # world -> nm, the centred frame
        lam = k / max(len(frames) - 1, 1)
        m = measures(X, Xc, Xo, atoms)
        ms = measures(Xc + lam * (Xo - Xc), Xc, Xo, atoms)
        rows.append({"frame": k, "morph": m, "straight": ms, "seed_drift_nm": float(np.abs(X0 - Xc).max()) if k == 0 else None})
        print(f"frame {k:3d}  morph: rmsd {m['rmsd_open']:.3f} nm, TM CA {m['rmsd_open_tm_ca']:.3f}, pore {m['pore_r_min']}, "
              f"helix {m['helix_ok']}, clashes {m['clashes']}   straight(l={lam:.2f}): pore {ms['pore_r_min']}, "
              f"helix {ms['helix_ok']}, clashes {ms['clashes']}")
    p = os.path.join(out, "results", "mscs_path_scores.json")
    json.dump(rows, open(p, "w"), indent=1)
    print(f"[score] -> {p}")


def fit_corr(a):
    """THE RIG'S KNOWLEDGE INTO THE MORPH: the same learnable rate field, trained so each particle ends at ITS OWN
    atom's open position (the correspondence the two deposited structures give), not at the open outline. A
    prototype of a loss the trainer does not have (its shape task scores grid mass only): it reuses the trainer's
    own model, learnables and rollout, and writes the fit where `score` reads it (a spec `<name>_corr`)."""
    import time
    import torch
    import yaml
    from plexus import engine, trainer as T
    spec = T.load(a.name)
    name = a.name + "_corr"
    sp_path = os.path.join(ROOT, "config", "training", "morph", f"{name}.yaml")
    raw = yaml.safe_load(open(spec["_path"])); raw["name"] = name
    yaml.safe_dump(raw, open(sp_path, "w"), sort_keys=False, default_flow_style=None, width=110)
    spec = T.load(name)
    out = T.out_dir(spec)
    os.makedirs(os.path.join(out, "models"), exist_ok=True); os.makedirs(os.path.join(out, "results"), exist_ok=True)
    Xc, Xo, info = _labels()
    s_w = WORLD / BOX_NM
    tgt = torch.as_tensor(Xo * s_w + WORLD / 2, dtype=torch.float32, device=a.device)
    tm = torch.as_tensor(np.array([TM[0] <= at[1] <= TM[1] for at in info["atoms"]]), device=a.device)
    engine.quiet(True)
    torch.manual_seed(0)
    learn = T.Learnables(spec["learnable"], a.device)
    tr = spec["training"]; lr, lr_min = float(tr["lr"]), float(tr["lr_min"])
    hist, t0 = [], time.time()
    for si, st in enumerate(tr["stages"]):
        sim = T._model(spec, train=True, resolution=st.get("resolution"))
        if not learn.p:
            with torch.no_grad():
                T._shape_rollout(sim, learn, "mpm_particle", a.device, grad=False)
        opt = torch.optim.Adam(learn.groups(lr), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(st["iters"]), eta_min=lr_min)
        last = learn.snapshot()
        for it in range(int(st["iters"])):
            opt.zero_grad()
            _, X, vol = T._shape_rollout(sim, learn, "mpm_particle", a.device, grad=True)
            d2 = ((X - tgt) ** 2).sum(1)
            loss = d2.mean() / s_w ** 2                       # nm^2: the mean squared distance to the open atoms
            if not torch.isfinite(loss):
                learn.restore(last)
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                continue
            loss.backward(); opt.step(); sch.step(); learn.clamp_(); last = learn.snapshot()
            rms_tm = float(d2[tm].mean().sqrt() / s_w)
            hist.append({"stage": si, "iter": it, "msd_nm2": float(loss), "rms_tm_nm": rms_tm})
            if it % 20 == 0 or it == int(st["iters"]) - 1:
                print(f"  stage {si} iter {it:3d}  rms to own open atom {float(loss) ** 0.5:.3f} nm (TM {rms_tm:.3f})",
                      flush=True)
    torch.save({"fitted": learn.snapshot(), "model": spec["model"], "task": spec["task"],
                "learnable": spec["learnable"]}, os.path.join(out, "models", "best.pt"))
    json.dump(hist, open(os.path.join(out, "results", "corr_history.json"), "w"))
    print(f"[fit_corr] {name}: {(time.time() - t0) / 60:.1f} min -> {out}")


def relax_frames(frames, Xc, atoms, device, steps=300, k_b=100.0, eps=1.0, sig=0.25, k_t=1.0):
    """THE HYBRID (exp04 M5b): each MPM frame's atoms relaxed at the atom scale the grid cannot resolve -- bonds held at
    the closed structure's lengths (every atom pair within 0.2 nm in the same or an adjacent residue), a WCA wall
    between every other pair (contact `sig` nm), and a tether of stiffness `k_t` to the MPM frame so the large-scale
    motion stays the morph's. Energies in kT-like units, lengths in nm; Adam, `steps` per frame, starting from the MPM
    frame; the neighbour list rebuilt every 50 steps. STATED: a minimisation, not dynamics -- it removes overlaps and
    keeps covalent geometry, it does not choose a path."""
    import torch
    from scipy.spatial import cKDTree
    ch = np.array([a[0] for a in atoms]); res = np.array([a[1] for a in atoms])
    ci = np.unique(ch, return_inverse=True)[1]
    bp = cKDTree(Xc).query_pairs(0.2, output_type="ndarray")
    bp = bp[(ci[bp[:, 0]] == ci[bp[:, 1]]) & (np.abs(res[bp[:, 0]] - res[bp[:, 1]]) <= 1)]
    d0 = torch.as_tensor(np.linalg.norm(Xc[bp[:, 0]] - Xc[bp[:, 1]], axis=1), device=device)
    bi = torch.as_tensor(bp, device=device)
    out = []
    for k, Xm in enumerate(frames):
        Xm_t = torch.as_tensor(Xm, device=device, dtype=torch.float64)
        X = Xm_t.clone().requires_grad_(True)
        opt = torch.optim.Adam([X], lr=0.002)
        for it in range(steps):
            if it % 50 == 0:
                pr = cKDTree(X.detach().cpu().numpy()).query_pairs(sig, output_type="ndarray")
                pr = pr[~((ci[pr[:, 0]] == ci[pr[:, 1]]) & (np.abs(res[pr[:, 0]] - res[pr[:, 1]]) <= 1))]
                ni = torch.as_tensor(pr, device=device)
            opt.zero_grad()
            rb = (X[bi[:, 0]] - X[bi[:, 1]]).norm(dim=1)
            E = k_b * ((rb - d0) ** 2).sum() + k_t * ((X - Xm_t) ** 2).sum()
            if len(ni):
                r = (X[ni[:, 0]] - X[ni[:, 1]]).norm(dim=1).clamp_min(0.05)
                sr6 = (sig / r) ** 6
                E = E + eps * torch.where(r < sig, sr6 * sr6 - 2 * sr6 + 1, torch.zeros_like(r)).sum()
            E.backward(); opt.step()
        out.append(X.detach().cpu().numpy())
        print(f"  relax frame {k}: {len(ni)} overlapping pairs left in the list", flush=True)
    return out


def waypoints(a):
    """PHASE G2: a recorded path (M5c by default) as the rig's waypoints. The rig's MscS beads are the C-alphas of 6PWN,
    chains A-G as parts c0-c6, residues 1-280 (shapes/exp04r_mscs_c). The path's C-alphas (residues 25-280) are put
    in the rig's frame by the superposition of the closed atoms' C-alphas on the rig's closed beads; residues 1-24,
    which the open structure lacks, move with residue 25. Every tenth display frame of the recorded run is one of
    its model frames exactly (201 display frames linear between 21); written as parts `w<j>_c<k>` (metres) of
    shapes/exp04r_mscs_path, the frame of exp04r_mscs_c."""
    from plexus.paths import graphs_data_path
    gd = graphs_data_path()
    info = json.load(open(os.path.join(gd, "shapes", SHAPE, "atoms.json")))
    atoms = info["atoms"]
    ch = np.array([x[0] for x in atoms]); res = np.array([x[1] for x in atoms]); nmv = np.array([x[2] for x in atoms])
    chains = sorted(set(ch))
    tj = np.load(os.path.join(gd, "channel", f"exp04_v{a.path_label}", "trajectory.npz"))
    T = tj["MscS_1__pos"].shape[0]
    idx = list(range(0, T, 10))
    rig = np.load(os.path.join(gd, "shapes", "exp04r_mscs_c", "points.npz"))
    B = [np.asarray(rig[f"c{k}"], float) * 1e9 for k in range(7)]           # nm, the rig's closed beads
    def frame_ca(t):
        X = np.zeros((len(atoms), 3))
        for k, c in enumerate(chains):
            X[ch == c] = (tj[f"MscS_{k + 1}__pos"][t] - 0.5) * BOX_NM
        return X
    X0 = frame_ca(0)
    ca = nmv == "CA"
    P, Q = [], []
    for k, c in enumerate(chains):
        for r in range(TM[0], CAGE[1] + 1):
            m = ca & (ch == c) & (res == r)
            if m.any() and r - 1 < len(B[k]):
                P.append(X0[m][0]); Q.append(B[k][r - 1])
    R, t0 = kabsch(np.array(P), np.array(Q))
    rms = float(np.sqrt((((np.array(P) @ R.T + t0) - np.array(Q)) ** 2).sum(1).mean()))
    print(f"[waypoints] path C-alphas -> rig beads: RMS {rms:.3f} nm over {len(P)} residues")
    out = {}
    for j, t in enumerate(idx):
        Xt = frame_ca(t)
        for k, c in enumerate(chains):
            W = B[k].copy()
            d25 = None
            for r in range(TM[0], CAGE[1] + 1):
                m = ca & (ch == c) & (res == r)
                if m.any() and r - 1 < len(W):
                    moved = (Xt[m][0] - X0[m][0]) @ R.T
                    W[r - 1] = B[k][r - 1] + moved
                    if d25 is None:
                        d25 = moved
            W[:TM[0] - 1] = B[k][:TM[0] - 1] + (d25 if d25 is not None else 0.0)
            out[f"w{j:02d}_c{k}"] = W * 1e-9
    sd = os.path.join(gd, "shapes", "exp04r_mscs_path")
    os.makedirs(sd, exist_ok=True)
    np.savez_compressed(os.path.join(sd, "points.npz"), **out)
    open(os.path.join(sd, "PROVENANCE.md"), "w").write(
        f"MscS closed -> open waypoints for morph_gate[waypoints]: the exp04 step {a.path_label} path's C-alphas in the "
        f"frame of exp04r_mscs_c ({len(idx)} waypoints, parts w<j>_c<k>, metres). tools/mscs_morph.py waypoints.\n")
    print(f"[waypoints] {len(idx)} waypoints x 7 chains -> {sd}")


# ---- the record: each path as a step of exp04's watcher ----------------------------------------------------------
# three panels (the renderer's legible maximum): what the path does, how right it is, how physical it is; the helix
# fraction goes in the row
# SHORT TITLES, LARGE TYPE (the human, 2026-09-29): the pore's narrowest radius, the RMS distance of every atom to its
# own open position, the atom pairs closer than 0.25 nm
PANELS = [("pore_r", "pore radius (nm)", 0.0, 0.8), ("rmsd_open", "RMSD to open (nm)", 0.0, 1.5),
          ("clashes", "clashes (pairs)", 0.0, 1600.0)]


def _morph_frames(name, device):
    import torch
    from plexus import engine, trainer as T
    spec = T.load(name)
    ck = torch.load(os.path.join(T.out_dir(spec), "models", "best.pt"), weights_only=False, map_location=device)
    learn = T.Learnables(spec["learnable"], device); learn.restore(ck["fitted"])
    engine.quiet(True)
    sim = T._model(spec, train=True, resolution=T._last_resolution(spec))
    frames = []
    with torch.no_grad():
        T._shape_rollout(sim, learn, "mpm_particle", device, grad=False, keep=frames)
    s_w = WORLD / BOX_NM
    return [(np.asarray(f, float) - WORLD / 2) / s_w for f in frames]          # nm, the centred frame


def record(a):
    """Write one path as exp04 step `exp04_v<label>`: trajectory (per chain, and the ruler's numbers as blocks of a
    one-element set `path`), the display spec (exp04's view of record: per-chain density surfaces, elevation 40,
    azimuth 30, zoom 1.0), the replayed movie and its last frame as the still."""
    import shutil
    import subprocess
    import yaml
    from plexus.paths import graphs_data_path
    Xc, Xo, info = _labels()
    atoms = info["atoms"]
    ch = np.array([at[0] for at in atoms])
    chains = sorted(set(ch))
    if a.path == "straight":
        frames = [Xc + l * (Xo - Xc) for l in np.linspace(0.0, 1.0, 201)]
    else:
        raw = _morph_frames(a.path.split(":")[-1], a.device)
        if a.path.startswith("relax:"):
            raw = relax_frames(raw, Xc, atoms, a.device, steps=a.relax_steps, eps=a.relax_eps, k_t=a.relax_tether)
        # DISPLAY FRAMES, linear between the model's recorded frames (its rate field acts once a frame): 201 for a
        # 10 s movie; the ruler is read on every one of them
        t = np.linspace(0.0, len(raw) - 1, 201)
        frames = []
        for u in t:
            i0 = int(np.floor(u)); i1 = min(i0 + 1, len(raw) - 1)
            frames.append(raw[i0] + (u - i0) * (raw[i1] - raw[i0]))
    ms = [measures(X, Xc, Xo, atoms) for X in frames]
    run = f"exp04_v{a.label}"
    d = os.path.join(graphs_data_path(), "channel", run)
    os.makedirs(d, exist_ok=True)
    to_w = lambda X: X / BOX_NM + 0.5                                    # nm -> the display world (1.0 = 18 nm)
    tj = {"world": np.float64(1.0), "world_size": np.ones(3, np.float32)}
    names = [f"MscS_{k + 1}" for k in range(len(chains))]
    for n, c in zip(names, chains):
        tj[f"{n}__pos"] = np.stack([to_w(X[ch == c]) for X in frames]).astype(np.float32)
        tj[f"{n}_atoms__pos"] = tj[f"{n}__pos"]           # THE ATOMS THEMSELVES, as dots inside a translucent skin
    tj["path__pos"] = np.tile(np.array([[[0.5, 0.5, 0.05]]], np.float32), (len(frames), 1, 1))
    for key, *_ in PANELS:
        tj[f"path__{key}"] = np.array([[[m["pore_r_min" if key == "pore_r" else key] or 0.0]] for m in ms], np.float32)
    np.savez_compressed(os.path.join(d, "trajectory.npz"), **tj)
    base = yaml.safe_load(open(os.path.join(ROOT, "config", "channel", "exp04_v14b.yaml")))["plotting"]
    pl = {k: base[k] for k in ("renderer", "up_axis", "box_frame", "background", "ssao", "ssao_radius_frac", "legend",
                                "silhouette", "surface_specular", "surface_ambient", "surface_diffuse", "light",
                                "render_3d", "surface_min_points") if k in base}
    # TRANSLUCENT SKINS WITH THE ATOMS INSIDE (the human, 2026-09-29): a skin alone hid the clashes the ruler counts
    atoms_ = [f"{n}_atoms" for n in names]
    pl.update({"compartment_sets": names + atoms_, "hide_sets": ["path"], "subject": "MscS_1",
               "surface": {**{n: dict(base["surface"]["MscS_1"], recontour=True) for n in names},   # a carried skin creased
                           **{a_: {"render": "dots", "point_size": 4.0} for a_ in atoms_}},
               "opacity": {**{n: 0.25 for n in names}, **{a_: 1.0 for a_ in atoms_}},
               "colors": {**{n: base["colors"][n] for n in names}, **{a_: base["colors"][n] for a_, n in zip(atoms_, names)}},
               "camera": {"elev": 40.0, "azim": 30.0}, "zoom": 1.0, "duration_s": 10.0, "real_time": False,
               "max_frames": 201, "stills": 4, "keep_stills": False, "replay_curves": True,
               "curve": [{"quantity": f"block:path:{k}", "ymin": lo, "ymax": hi, "ylabel": lab, "font_size": 28,
                          "tick_font_size": 20} for k, lab, lo, hi in PANELS]})
    sets = {n: {"n": int((ch == c).sum()), "title": f"MscS, chain {k + 1} (atoms)", "state": {
        "pos": {"width": 3, "integration": "none", "boundary": "world", "role": "coordinate"}}}
        for k, (n, c) in enumerate(zip(names, chains))}
    for n, c in zip(names, chains):
        sets[f"{n}_atoms"] = {"n": int((ch == c).sum()), "title": f"MscS, chain {names.index(n) + 1}: its atoms", "state": {
            "pos": {"width": 3, "integration": "none", "boundary": "world", "role": "coordinate"}}}
    sets["path"] = {"n": 1, "start": [[0.5, 0.5, 0.05]], "title": "the path's own readings (the ruler)", "state": {
        "pos": {"width": 3, "integration": "none", "boundary": "world", "role": "coordinate"},
        **{k: {"width": 1, "integration": "none", "unit": "1", "title": lab} for k, lab, *_ in PANELS}}}
    sz = os.path.join(graphs_data_path(), "shapes", SHAPE, "points.npz")
    zz = dict(np.load(sz))
    if any(f"closed_{c}" not in zz for c in chains):              # written once: a rewrite raced a reader (fit_corr)
        for c in chains:
            zz[f"closed_{c}"] = zz["closed"][ch == c]
        np.savez_compressed(sz, **zz)
    spec = {"general": {"name": run, "title": a.why, "seed": 0, "n_frames": len(frames) - 1, "dt": 1.0,
                        "record_cap": len(frames), "boundary": "wall", "dim": 3, "world": [1.0, 1.0, 1.0],
                        "units": {"length_um": BOX_NM * 1e-3}},
            "sets": sets,
            "seed": [{"op": "cloud_seed", "at": n_, "cloud": f"{SHAPE}/closed_{c}", "origin": [0.5, 0.5, 0.5],
                      "scale": 1.0 / (BOX_NM * 1e-9)} for n, c in zip(names, chains) for n_ in (n, f"{n}_atoms")],
            "fields": {}, "operators": [], "schedule": [], "plotting": pl}
    cp = os.path.join(ROOT, "config", "channel", f"{run}.yaml")
    yaml.safe_dump(spec, open(cp, "w"), sort_keys=False, default_flow_style=None, width=110)
    shutil.copyfile(cp, os.path.join(d, "spec.yaml"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "Plexus_Main.py"), "-o", "plot", f"channel/{run}",
                        "--device", "cpu"], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    mv = os.path.join(d, "movie.mp4")
    if not os.path.exists(mv):
        print((r.stdout + r.stderr)[-3000:]); raise SystemExit(f"[record] {run}: no movie")
    subprocess.run(["/workspace/.conda_envs/neural-graph-linux/bin/ffmpeg", "-loglevel", "error", "-y", "-sseof", "-0.2",
                    "-i", mv, "-frames:v", "1", os.path.join(d, "3d.png")], check=True)
    m0, m1 = ms[0], ms[-1]
    print(f"[record] {run}: {len(frames)} frames; pore {m0['pore_r_min']} -> {m1['pore_r_min']} nm, rms to own open "
          f"atom {m0['rmsd_open']:.3f} -> {m1['rmsd_open']:.3f} nm, helix pairs intact {m1['helix_ok']}, clashes max "
          f"{max(m['clashes'] for m in ms)}; movie + 3d.png in {d}")


def add_row(a):
    """Append step `a.label`'s row to exp04's results table (after the table's last row) and rebuild the watcher's
    record: every path is a step the human can see."""
    import re
    import subprocess
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import channel_spec as S
    from exp import load, _rewrite
    md = os.path.join(ROOT, "experiments", "exp04_membrane_channels.md")
    cells = [a.label, f"`channel/exp04_v{a.label}`", "MscS", "no judge (a path, not a rig run)", a.ruler, a.verdict,
             S.VERSIONS[a.label]["why"], a.said]
    cells = [c.replace("|", "/").replace("\n", " ") for c in cells]
    fm, body = load(md)
    lines = body.rstrip("\n").split("\n")
    last = max(i for i, l in enumerate(lines) if re.match(r"^\| [A-Za-z]?\d+[a-z]? \| `channel/exp04_v", l))
    lines.insert(last + 1, "| " + " | ".join(cells) + " |")
    _rewrite(md, fm, "\n".join(lines) + "\n")
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "exp_record.py"), "4"], env=env)
    subprocess.Popen([sys.executable, os.path.join(ROOT, "tools", "exp_record.py"), "4", "--caption-missing"], env=env,
                     stdout=open("/tmp/caption_exp04.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("prep"); p.add_argument("--name", default="mscs_open")
    s = sub.add_parser("score"); s.add_argument("name"); s.add_argument("--device", default="cuda:0")
    c = sub.add_parser("fit_corr"); c.add_argument("name"); c.add_argument("--device", default="cuda:0")
    r = sub.add_parser("record"); r.add_argument("label"); r.add_argument("path", help="'straight' or a trained morph name")
    r.add_argument("--why", default=""); r.add_argument("--device", default="cuda:0")
    r.add_argument("--relax_steps", type=int, default=300); r.add_argument("--relax_eps", type=float, default=1.0)
    r.add_argument("--relax_tether", type=float, default=1.0)
    wp = sub.add_parser("waypoints"); wp.add_argument("--path_label", default="M5c")
    w = sub.add_parser("row"); w.add_argument("label"); w.add_argument("--ruler", required=True)
    w.add_argument("--verdict", required=True); w.add_argument("--said", required=True)
    a = ap.parse_args()
    {"prep": prep, "score": score, "fit_corr": fit_corr, "record": record, "row": add_row, "waypoints": waypoints}[a.verb](a)


if __name__ == "__main__":
    main()
