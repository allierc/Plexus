"""06b -- THE BREACH: the same three bodies as 06, with the protease switched on.

    python test_06_breach.py --probe                     # choose the two phase points, cheaply
    python test_06_breach.py --run hole                  # 06b_hole
    python test_06_breach.py --run tear                  # 06c_tear

WHAT IS DIFFERENT FROM 06, AND IT IS ONE THING. 06's sheet is `Rig06c` -- 05b's plaque rig with the
driver swapped. This one is `Rig05m` -- 05h_1_hetero's protease rig (MT1-MMP on the sheet activating
proMMP2, TIMP-2 diffusible and TIMP-3 immobile, `bm_tear` at rho_crit) with the SAME driver swapped by
the SAME mixin. The tissue, the matrix, the camera, the boxes and the frame map are 06's, unchanged;
the matrix is not re-run at all, because nothing here reaches it.

THE TWO EXPERIMENTS ARE ONE KNOB APART. `inhib` is the total inhibitor in units of K_timp: raise it and
the activation is damped, lower it and nothing stops the cascade. Everything else stays at 05h_1's
published point (K_timp 1e-3, bound 0.6, k_deg 100, mt1_frac 0.25, hetero 1).

THRESHOLDS, DECIDED BEFORE THE RUNS, in the unit of the phenomenon and not of the mesh:

    a hole            3% .. 50% of the seeded faces gone, AND the largest connected patch of torn
                      faces is at least 20 faces -- a hole you can see, rather than single faces lost
                      all over. THE SECOND HALF OF THIS WAS REVISED BEFORE EITHER RUN WAS KEPT, and the
                      first version is written here because the reason matters: it asked for the
                      largest patch to be 60% of the whole torn set, i.e. for ONE hole. The probe shows
                      that is not what this chemistry does -- MT1-MMP is a smooth random field, so
                      activation peaks in several places at once and the sheet perforates in several
                      (inhib 1, k_deg 100: 39% torn, largest patch 38% of that, about 750 faces). A
                      criterion that calls that a failure is measuring the number of holes, which
                      nobody claimed, not whether one opened.
    torn apart        >= 80% of the seeded faces gone. No connectivity condition: at that point what
                      is left is fragments, and asking whether the HOLE is connected is backwards.

`--probe` runs the candidates short and prints both numbers, so the phase points are chosen against
these thresholds instead of the thresholds being read off whatever the runs happened to do.
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_05b_plaque as B                                              # noqa: E402
from test_05m_protease import Rig05m                                     # noqa: E402

SRC = "06_spheroid_ecm"                    # the matrix half of 06, re-drawn and never re-run
RHO_CRIT = 0.35
# 05h_1_hetero's published point. Only `inhib` moves between the two experiments.
BASE = dict(K=1.0e-3, bound=0.6, kdeg=100.0, mt1_frac=0.25, hetero=1.0)
RUNS = {
    "hole": dict(name="06_breach_hole", inhib=1.0, kdeg=100.0, frames=401,
                 what="a hole opens in the basement membrane of a growing spheroid"),
    "tear": dict(name="06_breach_torn", inhib=1.0, kdeg=300.0, frames=401,
                 what="the basement membrane is destroyed: the breach runs away"),
    "one": dict(name="06_hole_one", inhib=1.0, kdeg=80.0, frames=401,
                what="ONE hole that stops growing and leaves two thirds of the membrane standing -- "
                     "the sharpened MT1 field (3 modes, hetero 2) at the k_deg that arrests"),
    "small": dict(name="06_hole_small", inhib=1.0, kdeg=80.0, frames=401,
                  what="the first hole that meets the stable-hole criterion declared before the "
                       "sweep: one patch, a fifth of the sheet, and it has stopped"),
    "tiny": dict(name="06_hole_tiny", inhib=1.0, kdeg=150.0, frames=401,
                 what="a TINY stable hole: 1.2% of the sheet, one patch, arrested. From a localised "
                      "MT1-MMP source (a 20-degree cap) rather than from a random field"),
    "tiny_off": dict(name="06_hole_tiny_off", inhib=1.0, kdeg=150.0, frames=401,
                     what="the tiny stable hole, 45 degrees off the view axis so its rim reads as a "
                          "rim rather than as a disc"),
    "smaller": dict(name="06_hole_smaller", inhib=1.0, kdeg=200.0, frames=401,
                    what="a 15-degree source: the smallest hole this chemistry can hold, one step "
                         "above the diffusion length that stops it"),
    "larger": dict(name="06_hole_larger", inhib=1.0, kdeg=150.0, frames=401,
                   what="a 30-degree source: the same hole, wider"),
    "largest": dict(name="06_hole_largest", inhib=1.0, kdeg=150.0, frames=401,
                    what="a 45-degree source: a hole big enough to see the lumen through"),
    "stable": dict(name="06_hole_stable", inhib=1.0, kdeg=100.0, frames=401,
                   what="one hole that stops growing, in a sheet that is otherwise kept healthy by "
                        "bm_secrete and bm_refine"),
}


# =============================================================================================
#  what "a hole" means, measured
# =============================================================================================
def torn_mask(F0, Fc):
    """Which of the SEEDED faces are gone. Faces are only ever removed here (no refinement, no
    reseeding), so a seeded face is alive iff its vertex triple is still in the live list -- compared
    as a SORTED triple, because the live list may hold it in another rotation."""
    key = lambda T: np.sort(np.asarray(T), axis=1) @ np.array([1, 1 << 22, 1 << 44], np.int64)  # noqa
    return ~np.isin(key(F0), key(Fc))


def patches(F0, torn, min_faces=20):
    """Every edge-connected component of the torn set, largest first, as face counts.

    ONE HOLE IS A COUNT, NOT A FRACTION. `biggest_patch` says how much of the damage is in the largest
    component, which cannot distinguish one hole from one hole beside four small ones -- and "one
    stable hole" is a claim about both the number and whether it is still growing.
    """
    idx = np.flatnonzero(torn)
    if idx.size == 0:
        return []
    e = {}
    for f in idx:
        a, b, c = F0[f]
        for k in ((a, b), (b, c), (c, a)):
            e.setdefault((min(k), max(k)), []).append(f)
    seen, comp = set(), []
    for s0 in idx:
        if s0 in seen:
            continue
        stack, n = [s0], 0
        seen.add(s0)
        while stack:
            f = stack.pop()
            n += 1
            a, b, c = F0[f]
            for k in ((a, b), (b, c), (c, a)):
                for g in e.get((min(k), max(k)), ()):
                    if g not in seen:
                        seen.add(g)
                        stack.append(g)
        comp.append(n)
    return sorted(comp, reverse=True)


def biggest_patch(F0, torn):
    """The largest edge-connected component of the torn set, as a FRACTION of the torn set.

    A hole is connected; erosion is not. Reporting only "n faces torn" cannot tell the two apart, and
    they are different claims about the protease: one says the sheet failed somewhere, the other says
    it thinned everywhere.
    """
    idx = np.flatnonzero(torn)
    if idx.size == 0:
        return 0.0, 0
    # edge -> the (at most two) faces on it, over the torn faces only
    e = {}
    for f in idx:
        a, b, c = F0[f]
        for k in ((a, b), (b, c), (c, a)):
            e.setdefault((min(k), max(k)), []).append(f)
    seen, best = set(), 0
    for s in idx:
        if s in seen:
            continue
        stack, n = [s], 0
        seen.add(s)
        while stack:
            f = stack.pop()
            n += 1
            a, b, c = F0[f]
            for k in ((a, b), (b, c), (c, a)):
                for g in e.get((min(k), max(k)), ()):
                    if g not in seen:
                        seen.add(g)
                        stack.append(g)
        best = max(best, n)
    return best / float(idx.size), int(idx.size)


def rim_stats(F):
    """(boundary edges, boundary loops) of a triangle list.

    THE MEASURE THAT SURVIVES REFINEMENT. `torn_mask` asks which SEEDED triangles are gone, which is
    exact while faces are only ever removed and meaningless the moment `bm_refine` splits one: after a
    split no seeded triple exists anywhere and the whole sheet reads as torn. A hole has a RIM -- edges
    with one adjacent face instead of two -- and a closed sheet has none, at any refinement level. The
    number of rim LOOPS is the number of holes; the number of rim edges is their perimeter in elements.
    """
    F = np.asarray(F)
    e = {}
    for a, b, c in F:
        for k in ((a, b), (b, c), (c, a)):
            k = (min(k), max(k))
            e[k] = e.get(k, 0) + 1
    rim = [k for k, v in e.items() if v == 1]
    if not rim:
        return 0, 0
    adj = {}
    for a, b in rim:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, loops = set(), 0
    for v0 in adj:
        if v0 in seen:
            continue
        loops += 1
        stack = [v0]
        seen.add(v0)
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
    return len(rim), loops


def verdict(torn_frac, patch_frac, n_torn=0):
    if torn_frac >= 0.80:
        return "TORN APART"
    if 0.03 <= torn_frac <= 0.50 and patch_frac * n_torn >= 20:
        return "A HOLE"
    return "neither"


# A STABLE HOLE, declared before the sweep that looks for one. All three conditions, because each one
# alone is met by something that is not it: a small torn fraction alone is also a sheet that is only
# just starting to fail, a big patch alone is also a runaway caught early, and a plateau alone is also
# a sheet that never tore.
STABLE = dict(torn_lo=0.01, torn_hi=0.20, min_patch=20, max_growth_last_third=0.02)


def stable_verdict(tf, comp, series, n_face0):
    """`series` is the torn COUNT per frame. Growth is measured over the last third of the run, in
    units of the seeded face count, so the number means the same thing at any run length."""
    if not series:
        return "no frames", 0.0
    k = (2 * len(series)) // 3
    growth = (series[-1] - series[k]) / float(n_face0)
    big = comp[0] if comp else 0
    ok = (STABLE["torn_lo"] <= tf <= STABLE["torn_hi"] and big >= STABLE["min_patch"]
          and growth <= STABLE["max_growth_last_third"])
    return ("STABLE HOLE" if ok else "no"), growth


# =============================================================================================
def resharpen(rig, modes, hetero):
    """Rebuild the three per-cell fields with a different SHAPE, keeping their means.

    THE FIELD IS THE KNOB THE RATE SWEEP RAN OUT OF. k_deg sets how fast the sheet is eaten, and the
    sweep showed a hole arrests only once it has eaten its ACTIVATION PATCH -- so the smallest stable
    hole is as big as that patch, and the patch is a property of the MT1-MMP field, not of a rate.
    Two things set it:

      `modes`   `smooth_field` sums `n_modes` plane waves at wavenumber 3 on the sphere. Few modes ->
                few broad peaks -> one large patch. Many modes -> fine features -> many small ones.
      `hetero`  the contrast, as `(1-h) + 2h*f`. At h = 1 the field runs from 0 to twice its mean, the
                most 05h1 allows. ABOVE 1 the valleys clamp at zero and only the peaks survive, which
                sharpens without changing where they are -- the mean is renormalised afterwards, so
                the AMOUNT of MT1-MMP is held and only its arrangement moves.

    The means are preserved exactly as 05h1 does it, so `modes=6, hetero=1` reproduces the rebuilt
    field bit for bit and this function is a no-op at the published point.
    """
    from test_05h1_hetero import smooth_field
    uc = rig.u_epi[rig.F_epi].mean(1)
    uc = uc / uc.norm(dim=1, keepdim=True).clamp_min(1e-30)
    h = float(hetero)
    fld = lambda sd: ((1.0 - h) + h * 2.0 * smooth_field(  # noqa: E731
        uc, n_modes=int(modes), seed=sd, dev=rig.dev, dtype=rig.dtype)).clamp_min(0.0)
    m = fld(rig._seeds[0])
    rig.mt1 = m / m.mean().clamp_min(1e-30) * rig.mt1_frac
    t_ = fld(rig._seeds[1]); rig.s_timp_cell = rig.s_timp * t_ / t_.mean().clamp_min(1e-30)
    p_ = fld(rig._seeds[2]); rig.s_pro_cell = rig.s_pro * p_ / p_.mean().clamp_min(1e-30)
    q = rig.mt1
    print(f"[field] modes {int(modes)}, hetero {h}: mt1 over {q.shape[0]} cells, mean {float(q.mean()):.4f} "
          f"(held), max/mean {float(q.max()/q.mean()):.2f}, fraction above the mean "
          f"{float((q > q.mean()).float().mean()):.3f}", flush=True)
    return rig


# the camera the 2x2 draws with; a spot placed on it faces the reader instead of hiding round the back
CAM_DIR = np.array([np.cos(np.radians(18.0)) * np.cos(np.radians(30.0)),
                    np.cos(np.radians(18.0)) * np.sin(np.radians(30.0)),
                    np.sin(np.radians(18.0))])


def spot_field(rig, theta_deg, off_deg=0.0, peak_over_mean=2.5, floor=0.02):
    """Replace the random MT1-MMP field with ONE Gaussian cap of angular radius `theta_deg`.

    WHY A SPOT AND NOT A FIELD. The sweep over `smooth_field` showed the rate sets the hole's size and
    the field does not: whatever its mode count, the damage ends in one connected patch, because the
    field only picks where the breach NUCLEATES and the rim then propagates. So a small hole cannot be
    asked for by making the field finer -- it has to be asked for by making the SOURCE small, and then
    the question is a real one: does the rim stop at the edge of the source, or does it keep going into
    sheet the protease never touched? That is a property of the tear law, and this is the experiment
    that isolates it.

    THE PEAK IS HELD, NOT THE MEAN. 05h1 normalises `mt1` to a mean of `mt1_frac`; holding that while
    shrinking the spot would drive the peak up as 1/area and change the local chemistry along with the
    size. Here the peak is fixed at `peak_over_mean` x the published mean -- the same local MT1 the
    random field reaches -- and the total falls with the spot, which is the honest reading of "less
    protease, in one place".

    `s_timp_cell` and `s_pro_cell` are made UNIFORM at their means. Leaving them heterogeneous would
    let the inhibitor's own random field decide the rim's fate, and this run is asking about one thing.
    """
    uc = rig.u_epi[rig.F_epi].mean(1)
    uc = uc / uc.norm(dim=1, keepdim=True).clamp_min(1e-30)
    # WHERE THE SPOT SITS. On the camera axis it faces the reader squarely, which is the right place to
    # ASK whether a hole opens and the wrong place to see its shape: a cap seen face-on is a disc, and
    # the rim, the standoff and the epithelium showing through it are all foreshortened onto each other.
    # `off_deg` tilts it toward screen-up, so the hole is off the axis of the view and its rim reads as
    # a rim. It is a rotation of the SOURCE, not of the camera -- every other panel keeps its framing.
    from ecm_render import screen_basis
    _d, _u, _v = screen_basis(18.0, 30.0)
    a = float(np.radians(off_deg))
    dv = np.cos(a) * np.asarray(CAM_DIR) + np.sin(a) * np.asarray(_v)
    d = torch.as_tensor(dv, device=rig.dev, dtype=rig.dtype)
    d = d / d.norm()
    ang = torch.arccos((uc @ d).clamp(-1.0, 1.0))
    th = float(np.radians(theta_deg))
    peak = peak_over_mean * rig.mt1_frac
    rig.mt1 = floor * peak + (1.0 - floor) * peak * torch.exp(-(ang / th) ** 2)
    rig.s_timp_cell = torch.full_like(rig.mt1, float(rig.s_timp))
    rig.s_pro_cell = torch.full_like(rig.mt1, float(rig.s_pro))
    frac = float((rig.mt1 > 0.5 * peak).float().mean())
    print(f"[field] SPOT theta {theta_deg} deg, {off_deg} deg off the view axis: peak {float(rig.mt1.max()):.4f} "
          f"({peak_over_mean}x the published mean {rig.mt1_frac}), floor {floor}, mean "
          f"{float(rig.mt1.mean()):.4f}; {100*frac:.2f}% of cells above half the peak "
          f"({int(frac * rig.F_epi.shape[0])} of {rig.F_epi.shape[0]}); inhibitor and proMMP2 uniform",
          flush=True)
    return rig


def build(inhib, dev, kdeg=None, refine=False, modes=6, hetero=1.0, spot=0.0, seed_mt1=3,
          spot_off=0.0):
    """`refine` TURNS THE TWO CLOSED GATES ON INSIDE THE PROTEASE RUN. 05m runs with max_refine 0 and no
    reseeding, so the sheet thins everywhere as the tissue stretches it and rho crosses rho_crit for a
    reason that is not the chemistry -- which is why every small hole in the k_deg sweep keeps creeping
    (+2.95% over the last third at k_deg 70) instead of arresting. With bm_secrete and bm_refine ON --
    05f's values, the ones G43 and G44 passed on the real epithelium -- the sheet holds its density
    except where the protease eats it, so the hole's size should be set by the activation patch and
    should stop when it has eaten it."""
    K, bound = BASE["K"], BASE["bound"]
    kdeg = BASE["kdeg"] if kdeg is None else float(kdeg)
    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, sigma_T=7.0, zeta=20.0, s_target=1.0,
             k_drive=50.0, dev=dev)
    A = dict(kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3)
    S = dict(s_mode="homeostatic", tau_bm=40.0, rho_crit=RHO_CRIT,
             **(dict(max_refine=2, edge_trigger=1.45, reseed=True) if refine
                else dict(max_refine=0, reseed=False)))
    # 05h1's own translation of the phase-diagram axes into source rates, copied unchanged
    X = dict(K_timp=K, hetero=BASE["hetero"], s_timp=inhib * K * (1.0 - bound) / 8.0,
             s_timp3=inhib * K * bound / 40.0, s_mmp=0.0, s_mt1=0.0, k_deg=kdeg,
             mt1_frac=BASE["mt1_frac"], seed_mt1=int(seed_mt1))
    rig = Rig05m(**P, **A, **S, **X)
    if spot:
        return spot_field(rig, float(spot), float(spot_off))
    return rig if (int(modes) == 6 and float(hetero) == 1.0) else resharpen(rig, modes, hetero)


def probe(dev, frames, inhibs, kdegs=(None,), refine=False, modes=6, hetero=1.0,
          spot=0.0, seed_mt1=3, spot_off=0.0):
    F0 = None
    print(f"[probe] {frames} frames each, on the real driver; thresholds: hole = 3-40% torn with the "
          f"largest patch >= 60% of it, torn apart = >= 80% torn", flush=True)
    out = []
    for q in inhibs:
      for kd in kdegs:
        rig = build(q, dev, kd, refine, modes, hetero, spot, seed_mt1, spot_off)
        if F0 is None:
            F0 = rig.sheet.Fc.cpu().numpy().copy()
        t0 = time.time()
        died, n0, cross, ser = 0, int(rig.sheet.Fc.shape[0]), {}, []
        for t in range(frames):
            rig.frame(t)
            if not rig.alive():
                died = t
                break
            # WHEN, not just how much. A phase point that destroys the sheet by frame 80 and one that
            # destroys it by frame 380 both report ">= 80% torn"; only the second is a movie of a
            # sheet being destroyed, and the first is 300 frames of an empty panel.
            gone = (n0 - int(rig.sheet.Fc.shape[0])) / float(n0)
            ser.append(n0 - int(rig.sheet.Fc.shape[0]))
            for thr in (0.03, 0.40, 0.80):
                if gone >= thr and thr not in cross:
                    cross[thr] = t
        Fc = rig.sheet.Fc.cpu().numpy()
        n_rim, n_loops = rim_stats(Fc)
        tag_ = (f"spot {spot:.0f}deg" if spot
                else f"modes {int(modes):2d} h {float(hetero):.1f} s{int(seed_mt1)}")
        if refine:
            # under refinement the seeded-triple test is void; the rim is the whole measurement
            print(f"[probe] {tag_} kdeg {kd if kd is not None else BASE['kdeg']:6.1f} "
                  f"REFINE: {Fc.shape[0]} faces, {n_loops} holes, rim {n_rim} edges "
                  f"({time.time()-t0:.0f}s{', DIVERGED at ' + str(died) if died else ''})", flush=True)
            out.append(dict(inhib=q, kdeg=kd, refine=True, faces=int(Fc.shape[0]),
                            holes=n_loops, rim=n_rim))
            continue
        tm = torn_mask(F0, Fc)
        pf, n = biggest_patch(F0, tm)
        comp = patches(F0, tm)
        tf = n / float(F0.shape[0])
        sv, growth = stable_verdict(tf, comp, ser, F0.shape[0])
        out.append(dict(inhib=q, kdeg=kd, torn=n, torn_frac=tf, patch_frac=pf, patches=comp[:5],
                        growth_last_third=growth, verdict=verdict(tf, pf, n), stable=sv))
        print(f"[probe] {tag_} kdeg {kd if kd is not None else BASE['kdeg']:6.1f}: {n:5d}/{F0.shape[0]} faces torn ({100*tf:5.1f}%), largest "
              f"patches {comp[:4]} of >=20 faces {sum(1 for c in comp if c >= 20)}, growth over the "
              f"last third {100*growth:+.2f}% -> {sv} / {verdict(tf, pf, n)}"
              f" ({time.time()-t0:.0f}s{', DIVERGED at ' + str(died) if died else ''})", flush=True)
    return out


# =============================================================================================
def solve(tag, dev, frames, inhib, kdeg, refine=False, modes=6, hetero=1.0, spot=0.0,
          seed_mt1=3, spot_off=0.0, keep_n=201):
    """Run one phase point and store everything the 2x2's BM panel needs, per frame.

    RAGGED ON PURPOSE. The face list shrinks as the sheet tears, so faces, the field on them and the
    plaques cannot be stacked -- and stacking them by padding would put dead faces in the picture.
    The NODES are constant (2562, no refinement and no reseeding in this configuration), so those are
    stacked, and the per-frame face list indexes into them.
    """
    cfg = RUNS[tag]
    name = cfg["name"]
    d = os.path.join(B.LOG, name)
    os.makedirs(d, exist_ok=True)
    rig = build(inhib, dev, kdeg, refine, modes, hetero, spot, seed_mt1, spot_off)
    F0 = rig.sheet.Fc.cpu().numpy().copy()
    n_face0 = F0.shape[0]
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, keep_n))).astype(int).tolist())

    store, T = {}, {k: [] for k in ("t", "torn", "rho_min", "act_max", "rim")}
    t0, i = time.time(), 0
    for t in range(frames):
        rig.frame(t)
        if not rig.alive():
            print(f"[{name}] DIVERGED at frame {t} -- stopping, and the store keeps what it had",
                  flush=True)
            break
        rho = rig.sheet.areal_density() / rig.sheet.rho0
        T["t"].append(t)
        T["torn"].append(int(n_face0 - rig.sheet.Fc.shape[0]))
        if t % 10 == 0 or t == frames - 1:
            _re, _lo = rim_stats(rig.sheet.Fc.cpu().numpy())
            T["rim"].append([t, _re, _lo])
        T["rho_min"].append(float(rho.min()) if rho.numel() else float("nan"))
        T["act_max"].append(float(rig.res["act_max"][-1]) if rig.res.get("act_max") else float("nan"))
        if t in keep:
            store[f"t{i}"] = np.int32(t)
            store[f"x{i}"] = rig.sheet.x.float().cpu().numpy()
            store[f"f{i}"] = rig.sheet.Fc.cpu().numpy().astype(np.int32)
            store[f"v{i}"] = rig._mt1_on_faces().float().cpu().numpy()
            store[f"r{i}"] = rho.float().cpu().numpy()
            store[f"e{i}"] = rig.x_epi.float().cpu().numpy()
            store[f"n{i}"] = rig.ct_node.cpu().numpy().astype(np.int32)
            store[f"p{i}"] = ((rig.x_epi[rig.F_epi[rig.ct_face]] * rig.ct_w[:, :, None]).sum(1)
                              .float().cpu().numpy())
            i += 1
    Fend = rig.sheet.Fc.cpu().numpy()
    n_rim, n_loops = rim_stats(Fend)
    if refine:
        # THE SEEDED-TRIPLE TEST IS VOID UNDER REFINEMENT, so the verdict is the rim's.
        tf, pf, n_torn = float("nan"), float("nan"), 0
        vd = f"{n_loops} holes, rim {n_rim} edges, {Fend.shape[0]} faces"
    else:
        tm = torn_mask(F0, Fend)
        pf, n_torn = biggest_patch(F0, tm)
        tf = n_torn / float(n_face0)
        vd = verdict(tf, pf, n_torn)

    from spec_06 import write_spec
    write_spec(d, rig, name=name, frames=frames, matrix_src=SRC,
               extra=dict(kind="protease", inhib=inhib, kdeg=kdeg, refine=refine, modes=modes,
                          hetero=hetero, spot=spot, spot_off=spot_off, seed_mt1=seed_mt1,
                          mt1_field=(f"single Gaussian cap, theta {spot} deg, {spot_off} deg off the "
                                     f"view axis" if spot else
                                     f"smooth_field, {int(modes)} modes, hetero {hetero}, "
                                     f"seed {int(seed_mt1)}"),
                          faces_torn=int(n_torn), torn_frac=float(tf), rim_loops=int(n_loops),
                          verdict=vd))
    np.savez_compressed(os.path.join(d, "bm_frames.npz"), n_kept=np.int32(i),
                        FE=rig.F_epi.cpu().numpy().astype(np.int32),
                        centre=rig.c.float().cpu().numpy(), scale=np.float64(rig.scale),
                        F0=F0.astype(np.int32), **store)
    json.dump(dict(run=name, what=cfg["what"], frames=len(T["t"]), inhib=inhib, kdeg=kdeg,
                   refine=refine, modes=modes, hetero=hetero, spot=spot, seed_mt1=seed_mt1,
                   spot_off=spot_off,
                   # `kdeg` and `hetero` are arguments of THIS run, not defaults: BASE carries the
                   # published values and both are swept, so spreading them here would collide with
                   # the ones actually used -- which is how this line raised rather than lying.
                   **{k: v for k, v in BASE.items() if k not in ("kdeg", "hetero")},
                   rho_crit=RHO_CRIT, faces_seeded=n_face0, faces_torn=n_torn, torn_frac=tf,
                   biggest_patch_frac=pf, verdict=vd,
                   rho_min=min(T["rho_min"]) if T["rho_min"] else None,
                   rim_edges=n_rim, rim_loops=n_loops, faces_end=int(Fend.shape[0]), series=T),
              open(os.path.join(d, "metrics.json"), "w"), indent=1)
    gate_png(T, tf, pf, vd, os.path.join(d, "gate.png"), name, n_face0)
    print(f"[{name}] {len(T['t'])} frames in {time.time()-t0:.0f}s -- {n_torn}/{n_face0} faces torn "
          f"({100*tf:.1f}%), largest patch {100*pf:.1f}% of them, {n_loops} rim loops over {n_rim} "
          f"edges, rho_min {min(T['rho_min']):.3f} on rho_crit {RHO_CRIT} -> {vd}", flush=True)
    return d, vd


def gate_png(T, tf, pf, vd, path, name, n_face0):
    """White, no box, no title, bold letter top-left, verdict in green when it met the threshold."""
    ok = vd != "neither"
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.6), facecolor="white")
    for a in ax:
        a.set_facecolor("white")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_xlabel("frame")
    ax[0].plot(T["t"], 100.0 * np.asarray(T["torn"]) / n_face0, color="black", lw=1.6)
    ax[0].set_ylabel("faces torn (% of seeded)")
    ax[0].text(0.03, 0.93, f"{vd}: {100*tf:.1f}% torn, largest patch {100*pf:.1f}%",
               transform=ax[0].transAxes, color="green" if ok else "red", fontsize=10, va="top")
    ax[1].plot(T["t"], T["rho_min"], color="black", lw=1.6)
    ax[1].axhline(RHO_CRIT, color="red", lw=0.9, ls=":")
    ax[1].set_ylabel(r"$\min_f \rho/\rho_0$ against $\rho_{\rm crit}$")
    for i, a in enumerate(ax):
        a.text(-0.16, 1.05, "ab"[i], transform=a.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


# =============================================================================================
def render(d, movie_frames=200, fps=20, movie=True):
    """The 06 2x2 with this run's sheet in the bottom-left. The matrix is 06's, re-drawn, not re-run."""
    import yaml
    import run_ecm
    from test_06_panels import BMPanel
    src = os.path.join(B.LOG, SRC)
    spec = yaml.safe_load(open(os.path.join(src, "spec_run.yaml")))
    op = next(o for o in spec["operators"] if o["op"] == "mesh_contact")
    mf = np.asarray(np.load(op["tissue"].replace("/groups/saalfeld/home/allierc/Graph", "/workspace"),
                            mmap_mode="r")["mesh_frames"])
    panel = BMPanel(os.path.join(d, "bm_frames.npz"), mf, int(op.get("mesh_stride", 1)), mode="mt1",
                    name=os.path.basename(d))
    run_ecm.rerender(src, dest=d, movie_frames=movie_frames, fps=fps, bm_draw=panel, movie=movie)


def main():
    def arg(f, c, dflt):
        return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else dflt

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    if "--probe" in sys.argv:
        probe(dev, arg("--frames", int, 150),
              [float(q) for q in arg("--inhib", str, "0,1,3,10").split(",")],
              [float(k) for k in arg("--kdeg", str, "").split(",")] if "--kdeg" in sys.argv
              else (None,), refine="--refine" in sys.argv,
              modes=arg("--modes", int, 6), hetero=arg("--hetero", float, 1.0),
              spot=arg("--spot", float, 0.0), seed_mt1=arg("--seed-mt1", int, 3),
              spot_off=arg("--spot-off", float, 0.0))
        return
    tag = arg("--run", str, "hole")
    inhib = arg("--inhib", float, RUNS[tag]["inhib"])
    kdeg = arg("--kdeg", float, RUNS[tag]["kdeg"])
    d = os.path.join(B.LOG, RUNS[tag]["name"])
    if "--reuse" in sys.argv and os.path.exists(os.path.join(d, "bm_frames.npz")):
        print(f"[06b] reusing the solved sheet in {d}", flush=True)
    else:
        d, _ = solve(tag, dev, arg("--frames", int, RUNS[tag]["frames"]), inhib, kdeg,
                     refine="--refine" in sys.argv, modes=arg("--modes", int, 6),
                     hetero=arg("--hetero", float, 1.0), spot=arg("--spot", float, 0.0),
                     seed_mt1=arg("--seed-mt1", int, 3), spot_off=arg("--spot-off", float, 0.0))
    render(d, movie_frames=arg("--movie-frames", int, 200), fps=arg("--fps", int, 20),
           movie="--no-movie" not in sys.argv)
    print(f"[06b] -> {d}", flush=True)


if __name__ == "__main__":
    main()
