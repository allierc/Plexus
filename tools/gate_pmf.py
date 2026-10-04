#!/usr/bin/env python
"""exp04 Phase G2: the free energy of MscS's gate along its recorded path, from the clamped runs of G1.

    PYTHONPATH=src python tools/gate_pmf.py --rest G1a G1b G1c G1d G1e --stretched G1f G1g G1h G1i G1j [--label G2a]

Each G1 run holds the gate at one lambda (0 closed, 1 open, by arc length along the M5c path) with `morph_gate[clamp]`,
which writes the push of everything else along the path, F = sum_i f_i . d_i (kT per unit lambda), into cell.f_gate.
Over the hold (the ramp's end + `--discard` frames, to the run's end):

    <F>(lambda)      mean push along the path, with a block-averaged standard error (blocks of `--block` records)
    G(lambda)        = - int_0^lambda <F> dlambda (trapezoid), kT, per tension
    W                = [G(1) - G(0)]_rest - [G(1) - G(0)]_stretched : the work the stretch adds toward opening, kT
    dA_eff           = W kT / (tau_stretched - tau_rest) : the area change the lipids actually deliver, nm^2
    P_open(tau)      = 1 / (1 + exp(dG0 - W(tau))), W(tau) = dA_eff (tau - tau_rest) / kT, dG0 the protein's measured
                       zero-tension preference (Pliotas 2015: 29 kJ/mol = 11.7 kT); tau_1/2 = dG0 kT / dA_eff
    pocket lipids    lipid beads inside the protein's outline (in-plane radius under the outermost protein bead of
                     their 30-degree sector, within 0.6 nm of their height) and lipids touching it (within 0.9 nm of a
                     bead), per record, mean over the hold -- Pliotas 2015's pocket occupancy, read off the rig

The rest and stretched sets need the same lambdas. Writes graphs_data/channel/exp04_v<label>/ (3d.png: the figure;
gate_pmf.json: every number) for the watcher, and prints the table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
KT_J = 1.380649e-23 * 298.15
DG0_KT = 29.0e3 / 6.02214076e23 / KT_J          # Pliotas 2015 (quoting its refs 2, 36): 29 kJ/mol


def run_dir(label):
    from plexus.paths import graphs_data_path
    return os.path.join(graphs_data_path(), "channel", f"exp04_v{label}")


def one(label, discard, block):
    d = run_dir(label)
    spec = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
    u = spec["general"]["units"]
    L_nm = float(u["length_um"]) * 1e3
    k_tau = float(u["force_nN"]) / float(u["length_um"])                  # sim tension -> mN/m
    mg = next(o for o in spec["operators"] if o["op"] == "morph_gate")
    rd = next(o for o in spec["operators"] if o["op"] == "radial_drive")
    lam = float(mg["lambda0"])
    t = np.load(os.path.join(d, "trajectory.npz"))
    F = t["cell__f_gate"][:, 0, 0].astype(np.float64)
    tau = t["cell__tension"][:, 0, 0].astype(np.float64) * k_tau
    n = len(F)
    N = int(spec["general"]["n_frames"])
    fr = np.linspace(0, N, n)
    t_hold = float(rd["protocol"][2][0])                                  # the ramp's end
    m = fr >= t_hold + discard
    Fh = F[m]
    nb = max(len(Fh) // block, 1)
    bm = np.array([Fh[i * block:(i + 1) * block].mean() for i in range(nb)])
    se = float(bm.std(ddof=1) / math.sqrt(nb)) if nb > 1 else float("nan")
    # pocket occupancy: lipids inside the protein's outline, and touching it
    names = [k[:-5] for k in t.files if k.startswith("MscS_") and k.endswith("__pos")]
    Xp = np.concatenate([t[f"{nm}__pos"][-1] for nm in names]) * L_nm
    c = np.array(mg["origin"]) * L_nm
    Xp = Xp - c
    thp = np.arctan2(Xp[:, 1], Xp[:, 0]); rp = np.hypot(Xp[:, 0], Xp[:, 1])
    Lp = t["lipid__pos"][m].astype(np.float64) * L_nm - c
    idx = np.linspace(0, len(Lp) - 1, min(len(Lp), 200)).astype(int)
    inside, touch = [], []
    # the protein's outline as a table: its outermost bead radius per 30-degree sector and 0.6 nm height bin (a lipid
    # bead inside it is 'inside the outline'), looked up for every bead at once
    nb_t, zb = 12, 0.6
    sec_p = ((thp + np.pi) / (2 * np.pi) * nb_t).astype(int) % nb_t
    zlo = float(Xp[:, 2].min()) - 1.0
    zbin_p = ((Xp[:, 2] - zlo) / zb).astype(int)
    nz = int(zbin_p.max()) + 3
    rmax = np.full((nb_t, nz), -1.0)
    for s_, z_, r_ in zip(sec_p, zbin_p, rp):
        for dz in (-1, 0, 1):                                     # +-0.6 nm of the bead's height
            if 0 <= z_ + dz < nz:
                rmax[s_, z_ + dz] = max(rmax[s_, z_ + dz], r_)
    import torch
    Pt = torch.tensor(Xp)
    for i in idx:
        X = Lp[i]
        th = np.arctan2(X[:, 1], X[:, 0]); r = np.hypot(X[:, 0], X[:, 1])
        s_ = ((th + np.pi) / (2 * np.pi) * nb_t).astype(int) % nb_t
        z_ = np.clip(((X[:, 2] - zlo) / zb).astype(int), 0, nz - 1)
        inside.append(int((r < rmax[s_, z_]).sum()))
        touch.append(int((torch.cdist(torch.tensor(X), Pt).min(1).values < 0.9).sum()))
    return {"label": label, "lambda": lam, "F_mean_kT": float(Fh.mean()), "F_se_kT": se, "F_sd_kT": float(Fh.std()),
            "n_records": int(len(Fh)), "tension_mN_m": float(tau[m].mean()), "tension_sd_mN_m": float(tau[m].std()),
            "lipids_inside_outline": float(np.mean(inside)), "lipids_touching": float(np.mean(touch)),
            "F_trace": F.tolist(), "frames": fr.tolist(), "hold_from": float(t_hold + discard)}


def protein_pmf(label, lams):
    """The protein's OWN share of G along the path, exactly: the inter-chain WCA push (the rig's `pair_potential` law wca
    between protein sets, per-pair cap f_max applied) projected on the path's tangent, integrated like the measured push.
    It does not depend on tension; G minus it is the lipids' share."""
    import torch
    from plexus.operators.channel_ops import MorphGateClamp
    from plexus.paths import graphs_data_path
    spec = yaml.safe_load(open(os.path.join(run_dir(label), "spec.yaml")))
    mg = dict(next(o for o in spec["operators"] if o["op"] == "morph_gate"))
    pw = next(o for o in spec["operators"] if o["op"] == "pair_potential" and o.get("law") == "wca")
    sig, fcap = float(pw["sigma"]), float(pw.get("f_max", 0.0)) or None
    shp = next(s_["cloud"].split("/")[0] for s_ in spec["seed"] if s_.get("op") == "cloud_seed" and s_["at"] == mg["sets"][0])
    pts = np.load(os.path.join(graphs_data_path(), "shapes", shp, "points.npz"))
    keys = [f"c{k}" for k in range(len(mg["sets"]))]
    X0 = np.concatenate([pts[k] for k in keys]) * float(mg["open_scale"]) + np.array(mg["origin"])
    ch = torch.tensor(np.concatenate([np.full(len(pts[k]), i) for i, k in enumerate(keys)]))
    op = MorphGateClamp({**{k: v for k, v in mg.items() if k != "op"}, "_at": mg["sets"][0], "lambda0": 0.0})
    op._load(torch.tensor(X0, dtype=torch.float64))
    rc = sig * 2 ** (1 / 6)
    fine = np.linspace(0.0, 1.0, 161)
    F = []
    for lam in fine:
        X, D = op._at(float(lam))
        d = X[:, None, :] - X[None, :, :]; r = d.norm(dim=-1)
        m = (ch[:, None] != ch[None, :]) & (r < rc) & (r > 0)
        rr = torch.where(m, r, torch.ones_like(r))
        mag = torch.where(m, 24 * (2 * (sig / rr) ** 12 - (sig / rr) ** 6) / rr, torch.zeros_like(r))
        if fcap:
            mag = mag.clamp(max=fcap)
        F.append(float(((mag[..., None] * d / rr[..., None]).sum(1) * D).sum()))
    F = np.array(F)
    Gf = np.concatenate([[0.0], -np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(fine))])
    return np.interp(lams, fine, Gf), fine, Gf


def pmf(rows):
    rows = sorted(rows, key=lambda r: r["lambda"])
    lam = np.array([r["lambda"] for r in rows]); F = np.array([r["F_mean_kT"] for r in rows])
    se = np.array([r["F_se_kT"] for r in rows])
    G = np.concatenate([[0.0], -np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(lam))])
    # the trapezoid's error, the blocks independent: var(G_k) = sum over the rule's weights^2 se^2
    w = np.zeros((len(lam), len(lam)))
    for k in range(1, len(lam)):
        for j in range(k):
            w[k, j] += 0.5 * (lam[j + 1] - lam[j]); w[k, j + 1] += 0.5 * (lam[j + 1] - lam[j])
    Gse = np.sqrt((w ** 2 * se[None, :] ** 2).sum(1))
    return lam, F, se, G, Gse, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rest", nargs="+", required=True); ap.add_argument("--stretched", nargs="+", required=True)
    ap.add_argument("--label", default=None); ap.add_argument("--discard", type=float, default=40000.0)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--more", nargs="+", action="append", default=[],
                    help="another stretched set (repeatable): W is then fitted as tau x dA_eff / kT through all of them")
    a = ap.parse_args()
    R = [one(l, a.discard, a.block) for l in a.rest]
    S = [one(l, a.discard, a.block) for l in a.stretched]
    lr, Fr, ser, Gr, Gser, R = pmf(R)
    ls, Fs, ses, Gs, Gses, S = pmf(S)
    if not np.allclose(lr, ls):
        raise SystemExit(f"the rest and stretched runs hold different lambdas: {lr} vs {ls}")
    tau_r = float(np.mean([r["tension_mN_m"] for r in R])); tau_s = float(np.mean([r["tension_mN_m"] for r in S]))
    W = float((Gr[-1] - Gr[0]) - (Gs[-1] - Gs[0]))
    W_se = float(math.hypot(Gser[-1], Gses[-1]))
    dtau = tau_s - tau_r
    dA = W * KT_J / (dtau * 1e-3) * 1e18 if dtau > 0 else float("nan")        # nm^2
    dA_se = W_se * KT_J / (dtau * 1e-3) * 1e18 if dtau > 0 else float("nan")
    tau_half = tau_r + DG0_KT * KT_J / (dA * 1e-18) * 1e3 if dA > 0 else float("inf")
    W8 = dA * 1e-18 * 8e-3 / KT_J if dA == dA else float("nan")               # at 8 mN/m above rest
    P8 = 1.0 / (1.0 + math.exp(min(DG0_KT - W8, 700))) if W8 == W8 else float("nan")
    P0 = 1.0 / (1.0 + math.exp(DG0_KT))
    out = {"lambda": lr.tolist(), "tension_rest_mN_m": tau_r, "tension_stretched_mN_m": tau_s,
           "F_rest_kT": Fr.tolist(), "F_rest_se_kT": ser.tolist(), "F_stretched_kT": Fs.tolist(), "F_stretched_se_kT": ses.tolist(),
           "G_rest_kT": Gr.tolist(), "G_rest_se_kT": Gser.tolist(), "G_stretched_kT": Gs.tolist(), "G_stretched_se_kT": Gses.tolist(),
           "W_kT": W, "W_se_kT": W_se, "dA_eff_nm2": dA, "dA_eff_se_nm2": dA_se, "dG0_kT": DG0_KT,
           "tau_half_mN_m": tau_half, "W_at_8_mN_m_kT": W8, "P_open_8_mN_m": P8, "P_open_rest": P0,
           "pockets_rest": [r["lipids_inside_outline"] for r in R], "pockets_stretched": [r["lipids_inside_outline"] for r in S],
           "touching_rest": [r["lipids_touching"] for r in R], "touching_stretched": [r["lipids_touching"] for r in S],
           "runs": [{k: v for k, v in r.items() if k not in ("F_trace", "frames")} for r in R + S]}
    # THE JOINT FIT over every stretched set: W_g = A - B_g, A = [G(1) - G(0)]_rest is COMMON to all of them, so the
    # slope through the origin, s = sum tau_g W_g / sum tau_g^2, has var = ((sum tau_g)^2 se_A^2 + sum tau_g^2 se_Bg^2) / (sum tau_g^2)^2
    groups = [(tau_s, float(Gs[-1] - Gs[0]), float(Gses[-1]), "+".join(a.stretched[:1]) + "..")]
    for more in a.more:
        M = [one(l, a.discard, a.block) for l in more]
        lm, Fm, sem, Gm, Gmse, M = pmf(M)
        groups.append((float(np.mean([r["tension_mN_m"] for r in M])) - tau_r + tau_r, float(Gm[-1] - Gm[0]), float(Gmse[-1]), more[0] + ".."))
    A, seA = float(Gr[-1] - Gr[0]), float(Gser[-1])
    tg = np.array([g[0] - tau_r for g in groups]); Wg = np.array([A - g[1] for g in groups]); seB = np.array([g[2] for g in groups])
    slope = float((tg * Wg).sum() / (tg ** 2).sum())
    slope_se = float(math.sqrt((tg.sum() ** 2 * seA ** 2 + (tg ** 2 * seB ** 2).sum())) / (tg ** 2).sum())
    dA_fit = slope * KT_J / 1e-3 * 1e18; dA_fit_se = slope_se * KT_J / 1e-3 * 1e18
    tau_half_fit = tau_r + DG0_KT / slope if slope > 0 else float("inf")
    out.update({"fit_groups": [{"set": g[3], "tension_mN_m": g[0], "W_kT": float(w_), "W_se_kT": float(math.hypot(seA, g[2]))}
                               for g, w_ in zip(groups, Wg)],
                "fit_slope_kT_per_mN_m": slope, "fit_slope_se": slope_se, "dA_eff_fit_nm2": dA_fit, "dA_eff_fit_se_nm2": dA_fit_se,
                "tau_half_fit_mN_m": tau_half_fit,
                "P_open_fit_8_mN_m": 1.0 / (1.0 + math.exp(min(DG0_KT - slope * 8.0, 700)))})
    # THE FULLY EMERGENT READING: the rig's own zero-tension cost A (the lipids' and the protein's, no stated energy)
    # against the stretch's work: dG(tau) = A - W(tau); tau_1/2 where W reaches A (on the joint line)
    tau_half_emergent = tau_r + A / slope if slope > 0 else float("inf")
    out.update({"G_rest_open_kT": A, "tau_half_emergent_mN_m": tau_half_emergent,
                "P_open_emergent": [{"tension_mN_m": g[0], "dG_kT": float(A - w_),
                                     "P_open": 1.0 / (1.0 + math.exp(min(A - w_, 700)))} for g, w_ in zip(groups, Wg)]})
    print(f"FULLY EMERGENT: the rig's own zero-tension cost {A:.1f} +- {seA:.1f} kT; tau_1/2 {tau_half_emergent:.2f} mN/m; "
          + "; ".join(f"at {g[0]:.2f} mN/m dG {A - w_:+.1f} kT, P_open {1.0 / (1.0 + math.exp(min(A - w_, 700))):.3g}"
                      for g, w_ in zip(groups, Wg)))
    if len(groups) > 1:
        for g, w_ in zip(groups, Wg):
            print(f"  {g[3]:8s} at {g[0]:5.2f} mN/m: W {w_:5.1f} +- {math.hypot(seA, g[2]):.1f} kT")
        print(f"JOINT FIT W = tau dA_eff / kT: dA_eff = {dA_fit:.1f} +- {dA_fit_se:.1f} nm^2; tau_1/2 = {tau_half_fit:.2f} mN/m; "
              f"P_open at 8 mN/m {out['P_open_fit_8_mN_m']:.3f}")
    Gp, fine, Gpf = protein_pmf(R[0]["label"], lr)
    out.update({"G_protein_kT": Gp.tolist(), "G_lipids_rest_kT": (Gr - Gp).tolist(), "G_lipids_stretched_kT": (Gs - Gp).tolist(),
                "W_lambda_kT": (Gr - Gs).tolist(), "W_lambda_se_kT": np.hypot(Gser, Gses).tolist()})
    print(f"{'lambda':>7} | {'<F> rest (kT)':>16} {'G rest':>12} | {'<F> 8 mN/m (kT)':>16} {'G stretched':>12} | pockets rest/str | touching rest/str")
    for i in range(len(lr)):
        print(f"{lr[i]:7.2f} | {Fr[i]:8.1f} +- {ser[i]:5.1f} {Gr[i]:6.1f}+-{Gser[i]:4.1f} | {Fs[i]:8.1f} +- {ses[i]:5.1f} "
              f"{Gs[i]:6.1f}+-{Gses[i]:4.1f} | {R[i]['lipids_inside_outline']:5.1f} / {S[i]['lipids_inside_outline']:5.1f} | "
              f"{R[i]['lipids_touching']:5.1f} / {S[i]['lipids_touching']:5.1f}")
    print("the protein's own share (inter-chain WCA along the path, exact): " + ", ".join(f"{g:.0f}" for g in Gp) + " kT; "
          "the lipids' at rest: " + ", ".join(f"{g:.0f}" for g in Gr - Gp) + " kT")
    print(f"tension over the holds: rest {tau_r:.2f}, stretched {tau_s:.2f} mN/m")
    print(f"W = {W:.1f} +- {W_se:.1f} kT (the stretch's work toward opening); dA_eff = {dA:.1f} +- {dA_se:.1f} nm^2 "
          f"(Pliotas 2015's two-state fit: 8.4; the rig's outline: 15.1)")
    print(f"with dG0 {DG0_KT:.1f} kT: tau_1/2 = {tau_half:.2f} mN/m; P_open at 8 mN/m above rest {P8:.3f}; at rest {P0:.2e}")
    if a.label:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = run_dir(a.label); os.makedirs(d, exist_ok=True)
        json.dump(out, open(os.path.join(d, "gate_pmf.json"), "w"), indent=1)
        fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
        lab_r, lab_s = f"rest ({abs(tau_r) if abs(tau_r) < 0.05 else tau_r:.1f} mN/m)", f"stretched ({tau_s:.1f} mN/m)"
        ax[0].errorbar(lr, Fr, ser, color="tab:blue", marker="o", capsize=3, label=lab_r)
        ax[0].errorbar(ls, Fs, ses, color="tab:red", marker="o", capsize=3, label=lab_s)
        ax[0].axhline(0, color="0.6", lw=0.8)
        ax[0].set_ylabel("push along path (kT)", fontsize=16)
        ax[1].errorbar(lr, Gr, Gser, color="tab:blue", marker="o", capsize=3, label=lab_r)
        ax[1].errorbar(ls, Gs, Gses, color="tab:red", marker="o", capsize=3, label=lab_s)
        ax[1].plot(fine, Gpf, color="0.55", lw=1.5, ls="--", label="protein's own clashes")
        ax[1].set_ylabel("free energy G (kT)", fontsize=16)
        W_l = Gr - Gs; W_e = np.hypot(Gser, Gses)
        ax[2].fill_between(lr, W_l - W_e, W_l + W_e, color="0.8")
        ax[2].plot(lr, W_l, "o-", color="0.2", label="measured")
        ax[2].axhline(dtau * 1e-3 * 8.4e-18 / KT_J, color="0.5", ls=":", label="tau x 8.4 nm2 (Pliotas 2015)")
        ax[2].axhline(0, color="0.6", lw=0.8)
        ax[2].set_ylabel("stretch's work toward opening (kT)", fontsize=16)
        if len(groups) > 1:
            # W AGAINST TENSION: every stretched set against the one rest set, the joint line through the origin
            tt = np.array([0.0] + list(tg)); ww = np.array([0.0] + list(Wg))
            ee = np.array([0.0] + [math.hypot(seA, g[2]) for g in groups])
            ax[3].errorbar(tt[1:], ww[1:], ee[1:], fmt="o", color="0.2", capsize=3, label="measured")
            xx = np.linspace(0, max(tt) * 1.1, 50)
            ax[3].plot(xx, slope * xx, color="0.2", lw=1.2, label=f"fit: dA_eff {dA_fit:.1f} +- {dA_fit_se:.1f} nm2")
            ax[3].plot(xx, xx * 1e-3 * 8.4e-18 / KT_J, color="0.5", ls=":", label="8.4 nm2 (Pliotas 2015)")
            ax[3].axhline(DG0_KT, color="0.7", ls="--", lw=0.8, label=f"MscS's measured zero-tension cost {DG0_KT:.1f} kT")
            ax[3].axhline(A, color="0.3", ls="--", lw=0.8, label=f"the rig's own zero-tension cost {A:.1f} kT")
            ax[3].set_xlabel("membrane tension (mN/m)", fontsize=16)
            ax[3].set_ylabel("stretch's work, closed to open (kT)", fontsize=16)
        else:
            ax[3].plot(lr, [r["lipids_inside_outline"] for r in R], "o-", color="tab:blue", label=lab_r)
            ax[3].plot(ls, [r["lipids_inside_outline"] for r in S], "o-", color="tab:red", label=lab_s)
            ax[3].set_ylabel("lipids inside outline", fontsize=16)
        for k, x in enumerate(ax):
            if not x.get_xlabel():
                x.set_xlabel("gate (0 closed, 1 open)", fontsize=16)
            x.tick_params(labelsize=13); x.legend(fontsize=11, frameon=False)
            for s_ in ("top", "right"):
                x.spines[s_].set_visible(False)
            x.text(0.0, 1.04, "abcd"[k], transform=x.transAxes, fontsize=18)
        fig.tight_layout()
        fig.savefig(os.path.join(d, "3d.png"), dpi=120, facecolor="white")
        print(f"wrote {d}/3d.png and gate_pmf.json")


if __name__ == "__main__":
    main()
