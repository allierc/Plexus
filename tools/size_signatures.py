#!/usr/bin/env python
"""The size-control SIGNATURES of one run, from its trajectory -- experiment 3's ruler.

    PYTHONPATH=src python tools/size_signatures.py tissue/exp03_g1_sizer_s1 [...] [--until F] [--json out]

Six signatures, each the one a strand of the literature tests a size-control hypothesis by
(experiments/exp03_size_hypotheses.md has the predictions per hypothesis):

  S1  slope of added volume on birth volume, dV = V_d - V_b on V_b, both / v_ref
      -1 sizer, 0 adder, +1 timer under exponential growth (Amir 2014; Campos 2014;
      Taheri-Araghi 2015; Cadart 2018)
  S2  mother-daughter correlation of birth volume, rho = (1 + S1)/2 for a linear rule
  S3  slope of G1 duration on ln(birth volume), frames per e-fold: -1/lambda for a G1 sizer
      under exponential growth, 0 for a G1 clock (Cadart 2018)
  S4  median V/v_ref drift over the last two mean cycles, and the rise of the population CV from
      the settle frame to the end
  S5  cycle-length distribution (Smith & Martin 1973): minimum, the exponential rate of the
      fraction not yet divided past it (the alpha curve) and how straight that tail is (R^2 of
      its log-linear fit), and the mean sibling |L1 - L2| (the beta curve: 1/rate for a hazard)
  S6  growth exponent beta in dV/dt ~ V^beta, fitted WITHIN each cell on its recorded volume
      (median over cells): 1 exponential, 0 linear. `S6_beta_xcell` is the old across-cell value
  S7  how a size deviation is passed down: the regression slope of a daughter's birth volume on her
      mother's (the seeded cells' "birth" is their volume at the settle frame) -- (1 + S1)/2 for a
      linear rule: 0 sizer, 0.5 adder, 1 timer -- and, per generation, the mean and CV of birth and
      division volume, which is where size control is SEEN: a wide seed's CV falls in one
      generation under a sizer, halves per generation under an adder, and does not fall under a timer

LINEAGE is read from `cell_id` / `parent_id` exactly as `tools/size_report.py` reads it (a cell is
born the frame its id first appears; it divided if its id is some later cell's parent). Birth
volume is read `birth_lag` frames after the septum. Volumes are the rules' own convention (the
polyhedron whenever `sep` exists). `--until` scores only frames before it -- the first wrecked
frame from `tools/growth_audit.py`, so no signature is read off a broken mesh.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.ptp(x[ok]) == 0:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def signatures(spec, birth_lag=12, until=None):
    from plexus.paths import graphs_data_path
    from size_report import _convention, _volumes
    group, name = spec.split("/", 1)
    p = os.path.join(graphs_data_path(group, name), "trajectory.npz")
    from size_report import LazyNpz
    z = LazyNpz(p)
    nFs = z["vertex__mesh_nF"]
    T = len(nFs) if until is None else max(2, min(len(nFs), int(until)))
    if "cell__cell_id" not in z.files or "cell__parent_id" not in z.files:
        raise SystemExit(f"{spec}: no cell_id / parent_id -- lineage cannot be read")
    ids = [np.rint(np.asarray(z["cell__cell_id"][t])[:int(nFs[t]), 0]).astype(np.int64) for t in range(T)]
    pids = [np.rint(np.asarray(z["cell__parent_id"][t])[:int(nFs[t]), 0]).astype(np.int64) for t in range(T)]
    phase = ([np.asarray(z["cell__phase"][t])[:int(nFs[t]), 0] for t in range(T)]
             if "cell__phase" in z.files else None)
    first, last, parent, g1_end = {}, {}, {}, {}
    where_first, where_last = {}, {}
    for t in range(T):
        for j, (c, pc) in enumerate(zip(ids[t], pids[t])):
            c = int(c)
            if c not in first:
                first[c] = t; where_first[c] = j; parent[c] = int(pc)
            last[c] = t; where_last[c] = j
            if phase is not None and c not in g1_end and t > first[c] and phase[t][j] >= 1:
                g1_end[c] = t
    parents = {pc for pc in parent.values()}
    ref_frame = int(z["vertex__mesh_scalar_ref_frame"][0]) if "vertex__mesh_scalar_ref_frame" in z.files else 0
    cells = []                                   # (id, tb, tread, td, tlast)
    seeded = []                                  # (id, tlast) of seeded cells that divided
    for c, tb in first.items():
        tl = last[c]
        if tl >= T - 1 or c not in parents:
            continue                              # alive at the end, or died
        if tb == 0:
            seeded.append((c, tl))
            continue
        cells.append((c, tb, min(tb + birth_lag, tl), tl + 1, tl))
    # the read frame's index for each cell
    idx_at = {}
    need = {min(ref_frame, T - 1), T - 1}
    for c, tb, tr, td, tl in cells:
        need |= {tr, tl}
    for c, tl in seeded:
        need.add(tl)
    for t in range(T):
        if t in need:
            for j, c in enumerate(ids[t]):
                idx_at[(int(c), t)] = j
    conv = _convention(p)
    V = {t: _volumes(z, t, conv) for t in sorted(need)}
    live = lambda t: (np.asarray(z["cell__occ"][t])[:int(nFs[t])] > 0) if "cell__occ" in z.files else slice(None)  # noqa: E731
    v_ref = float(np.median(V[min(ref_frame, T - 1)][live(min(ref_frame, T - 1))]))
    rec = {}
    for c, tb, tr, td, tl in cells:
        vb = V[tr][idx_at[(c, tr)]] / v_ref
        vd = V[tl][idx_at[(c, tl)]] / v_ref
        rec[c] = dict(tb=tb, tr=tr, td=td, vb=vb, vd=vd, L=td - tb, p=parent[c],
                      g1=(g1_end[c] - tb) if c in g1_end else np.nan)
    # GENERATIONS: a seeded cell is generation 0, its "birth" volume its volume at the settle frame
    tr0 = min(ref_frame, T - 1)
    gen0 = {}
    for j, c in enumerate(ids[tr0]):
        gen0[int(c)] = V[tr0][j] / v_ref if j < len(V[tr0]) else np.nan
    seeded_vd = {c: V[tl][idx_at[(c, tl)]] / v_ref for c, tl in seeded if (c, tl) in idx_at}
    depth = {}
    def _depth(c):
        if c in depth:
            return depth[c]
        if c in gen0 and first.get(c, 1) == 0:
            depth[c] = 0
        else:
            depth[c] = _depth(parent[c]) + 1 if parent.get(c) in first else 1
        return depth[c]
    out = dict(spec=spec, frames=T, until=until, convention=conv, v_ref=v_ref, cycles=len(rec),
               cells=[int(nFs[0]), int(nFs[T - 1])])
    if len(rec) < 10:
        out["note"] = "fewer than 10 complete cycles"
        return out
    r = list(rec.values())
    Vb = np.array([x["vb"] for x in r]); Vd = np.array([x["vd"] for x in r])
    L = np.array([x["L"] for x in r], float); G1 = np.array([x["g1"] for x in r], float)
    out["S1_slope"] = _fit(Vb, Vd - Vb)
    md = [(rec[x["p"]]["vb"], x["vb"]) for x in r if x["p"] in rec]
    out["S2_rho"] = float(np.corrcoef(*zip(*md))[0, 1]) if len(md) >= 10 else float("nan")
    out["S2_pairs"] = len(md)
    out["S3_g1_slope"] = _fit(np.log(Vb), G1) if np.isfinite(G1).sum() >= 10 else float("nan")
    out["S3_g1_mean"] = float(np.nanmean(G1)) if np.isfinite(G1).any() else float("nan")
    # S4 -- stationarity
    tr0 = min(ref_frame, T - 1)
    w = int(min(T - 1 - tr0, 2 * L.mean()))
    t_w = T - 1 - w
    if t_w not in V:
        V[t_w] = _volumes(z, t_w, conv)
    med = lambda t: float(np.median(V[t][live(t)])) / v_ref            # noqa: E731
    cv = lambda t: float(V[t][live(t)].std() / V[t][live(t)].mean())    # noqa: E731
    out["S4_drift"] = med(T - 1) / med(t_w) - 1.0
    out["S4_cv_rise"] = cv(T - 1) - cv(tr0)
    out["S4_median_end"] = med(T - 1)
    out["cv_Vd"] = float(Vd.std() / Vd.mean())
    # S5 -- cycle-length distribution
    Lmin = float(np.percentile(L, 2))
    tail = np.sort(L[L >= Lmin])
    surv = 1.0 - np.arange(len(tail)) / len(tail)
    keep = surv > 0.02
    if keep.sum() >= 10:
        a, b = np.polyfit(tail[keep] - Lmin, np.log(surv[keep]), 1)
        pred = a * (tail[keep] - Lmin) + b
        ss = np.sum((np.log(surv[keep]) - pred) ** 2); st = np.sum((np.log(surv[keep]) - np.log(surv[keep]).mean()) ** 2)
        out["S5_L_min"], out["S5_L_median"] = Lmin, float(np.median(L))
        out["S5_tail_rate"], out["S5_tail_r2"] = float(-a), float(1 - ss / max(st, 1e-12))
    sib = {}
    for x in r:
        sib.setdefault(x["p"], []).append(x["L"])
    d = [abs(v[0] - v[1]) for v in sib.values() if len(v) == 2]
    out["S5_sibling_dL"] = float(np.mean(d)) if d else float("nan")
    out["S5_sibling_pairs"] = len(d)
    # S6 -- growth exponent across cells
    gr = np.array([(x["vd"] - x["vb"]) / max(x["td"] - 1 - x["tr"], 1) for x in r])
    vm = 0.5 * (Vb + Vd)
    ok = (gr > 0) & (vm > 0)
    out["S6_beta_xcell"] = _fit(np.log(vm[ok]), np.log(gr[ok])) if ok.sum() >= 10 else float("nan")
    # S6 WITHIN EACH CELL -- the growth law read along one cell's own life, so the cycle rule cannot
    # leak into it. The across-cell estimate above compares whole-cycle mean rates between cells,
    # and a checkpoint makes cycle length depend on birth size: it read 1.5-2.0 on exponentially
    # growing checkpoint arms (exp 3, finding 6). Here each cell's recorded volume is sampled every
    # `STEP` frames from its birth read to `TAIL` frames before division, log(dV/dt) is regressed
    # on log(V) inside that cell, and the median slope over cells is the exponent: 1 exponential,
    # 0 linear. Needs the recorded `volume` block; without it the across-cell value stands in.
    STEP, TAIL = 8, 4
    if "cell__volume" in z.files:
        vol = z["cell__volume"]
        slot = {}
        def _slot(t):
            if t not in slot:
                slot[t] = {int(c): j for j, c in enumerate(ids[t])}
            return slot[t]
        betas = []
        for c, x in rec.items():
            ts = list(range(x["tr"], x["td"] - 1 - TAIL, STEP))
            if len(ts) < 5:
                continue
            v = np.array([vol[t][_slot(t)[c], 0] for t in ts], float)
            dv = np.diff(v) / STEP
            vm_ = 0.5 * (v[1:] + v[:-1])
            okc = (dv > 0) & (vm_ > 0)
            if okc.sum() >= 4 and np.ptp(np.log(vm_[okc])) > 0.05:
                betas.append(np.polyfit(np.log(vm_[okc]), np.log(dv[okc]), 1)[0])
        out["S6_beta"] = float(np.median(betas)) if len(betas) >= 10 else float("nan")
        out["S6_cells"] = len(betas)
    else:
        out["S6_beta"] = out["S6_beta_xcell"]
    # S7 -- a deviation passed down, and the per-generation view
    vb_of = {c: x["vb"] for c, x in rec.items()}
    vb_of.update({c: v for c, v in gen0.items() if first.get(c, 1) == 0})
    md2 = [(vb_of[x["p"]], x["vb"]) for x in r if x["p"] in vb_of]
    out["S7_decay"] = _fit(*zip(*md2)) if len(md2) >= 10 else float("nan")
    gens = {}
    for c, x in rec.items():
        gens.setdefault(_depth(c), []).append((x["vb"], x["vd"]))
    g0 = [(gen0[c], seeded_vd.get(c, np.nan)) for c in gen0 if first.get(c, 1) == 0]
    if g0:
        gens[0] = g0
    rows = []
    for g in sorted(gens):
        a_ = np.array(gens[g], float)
        vb_, vd_ = a_[:, 0], a_[:, 1]
        cvf = lambda v: float(np.nanstd(v) / np.nanmean(v)) if np.isfinite(v).sum() >= 3 else float("nan")  # noqa: E731
        rows.append(dict(gen=int(g), n=int(len(a_)), mean_vb=float(np.nanmean(vb_)), cv_vb=cvf(vb_),
                         mean_vd=float(np.nanmean(vd_)) if np.isfinite(vd_).any() else float("nan"),
                         cv_vd=cvf(vd_)))
    out["S7_generations"] = rows
    # ... and over TIME: division volume binned by the frame the cell divided, 200 frames a bin --
    # the mean should hold and, under size control, the CV should fall or stay low
    tb_ = []
    for x in r:
        tb_.append(((x["td"] // 200) * 200, x["vd"]))
    bins = {}
    for k, v in tb_:
        bins.setdefault(int(k), []).append(v)
    out["S7_time"] = [dict(t=k, n=len(v), mean_vd=float(np.mean(v)),
                           cv_vd=float(np.std(v) / np.mean(v)) if len(v) >= 3 else float("nan"))
                      for k, v in sorted(bins.items())]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--until", type=int, default=None)
    ap.add_argument("--birth-lag", type=int, default=12)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    keys = ["S1_slope", "S2_rho", "S3_g1_slope", "S4_drift", "S4_cv_rise", "S5_tail_rate", "S5_tail_r2", "S6_beta",
            "S7_decay"]
    print(f"{'run':<30} {'cyc':>5} " + " ".join(f"{k:>11}" for k in keys))
    for sp in a.specs:
        o = signatures(sp, a.birth_lag, a.until)
        print(f"{sp.split('/')[-1]:<30} {o.get('cycles', 0):5d} "
              + " ".join(f"{o.get(k, float('nan')):11.3f}" for k in keys))
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
