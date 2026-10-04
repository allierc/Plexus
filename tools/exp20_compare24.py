#!/usr/bin/env python
"""exp20 BATCH 4 COMPILED: the 24 fish of the deposit, each with the SIREN law (input mask, no W prior, stimuli only,
exp17's rig) and its no-network twin -- one table, the figures, and the tests of the paper's findings.

    PYTHONPATH=src:tools python tools/exp20_compare24.py

Reads, per fish: data/design.json (session), baselines_<rec>.json (gut-responsive cells, the replicable cells' held-out
evoked change), the runs' _test.json (short / long skill), their movie npz (brain-mean R2, trainer _brain_mean_metrics),
their _freetrial.json (per-pulse traces of the gut-responsive cells inside the free rollout: the law, W = 0) and
data/omega_<run>.json (the SIREN modulation). The evoked change of a pulse = the gut-responsive cells' mean dF/F over
0-20 s after the onset minus the 10 s before; "gut" = every site but 1 (off the fish): the gut in glucose, glutamate,
L-glucose and fish water, the portal vessel in blood glucose.

THE PAPER'S TESTS (Chen 2026):
  Fig. 2a-d   the replicable response per condition (the top-1 % cells by training-pulse t, on the held-out gut pulses;
              one count rule in every fish): glucose and glutamate respond, L-glucose and fish water do not
  Fig. 2g     foregut (site 2) against the other gut spot (site 5) in the fish that have both (glucose 4-6)
  Fig. 5d     response width (FWHM of the gut-responsive cells' mean response) at the vessel sites of blood glucose
              against the gut sites, recorded -- and the law's
  the network -- the law's evoked change over the recorded, against W = 0 at inference and the no-network twin
Writes data/compare24.json and presentation/figs/compare24_traces_<a|b>.png, compare24_conditions.png.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
DATA = os.path.join(EXP, "data")
FIGS = os.path.join(EXP, "presentation", "figs")
RUNS = os.path.join(os.environ["GNN_OUTPUT_ROOT"], "log", "training", "gutbrain")
GD = os.path.join(os.environ["GNN_OUTPUT_ROOT"], "graphs_data", "zebrafish")
COND = [("glucose", "D-glucose, gut"), ("glutamate", "glutamate, gut"), ("Lglucose", "L-glucose, gut (control)"),
        ("fish_water", "fish water, gut (control)"), ("blood_glucose", "D-glucose, portal vessel")]
CTRL_SITE = 1
EV_S, PRE_S = 20.0, 10.0


def names(cond, k):
    if cond == "glucose":
        return f"gb_sx_f{k}_mask_siren", f"gb_b4_glucose_f{k}_now"
    return f"gb_b4_{cond}_f{k}", f"gb_b4_{cond}_f{k}_now"


def load(n, what):
    p = os.path.join(RUNS, n, "results", f"{n}_{what}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def brain_r2(n, stem=None):
    from plexus import trainer as T
    p = os.path.join(RUNS, n, "results", f"{stem or n}_movie.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return float(T._brain_mean_metrics(z["mean_obs_all"], z["mean_pred_all"])["brain_mean_r2"])


def fwhm(trace, pre, dt):
    """Full width at half maximum of a response trace (baseline = the mean before the pulse), in s; None if no peak."""
    y = np.asarray(trace, float) - np.mean(trace[:pre])
    post = y[pre:]
    if post.max() <= 0:
        return None
    k = int(post.argmax())
    half = post[k] / 2
    lo = k
    while lo > 0 and post[lo - 1] >= half:
        lo -= 1
    hi = k
    while hi < len(post) - 1 and post[hi + 1] >= half:
        hi += 1
    return (hi - lo + 1) * dt


def per_fish(cond, k, D):
    net, now = names(cond, k)
    rec_name = None
    ft = load(net, "freetrial")
    if ft:
        rec_name = ft["recording"]
    d = D.get((cond, k), {})
    row = {"condition": cond, "fish": k, "net": net, "now": now, "minutes": d.get("minutes"), "cells": d.get("cells"),
           "volume_s": d.get("volume_s"), "pulses": len(d.get("pulses", []))}
    base = rec_name.replace("_nosw", "") if rec_name else f"gutbrain_{cond}_f{k}"
    bj = os.path.join(DATA, f"baselines_{base}.json")
    if os.path.exists(bj):
        b = json.load(open(bj))
        row.update({"gut_responsive": b["gut_responsive"], "top1_heldout": b.get("evoked_top1_heldout_gut"),
                    "noise_sigma2": b["noise_sigma2"]})
    t = load(net, "test")
    if t:
        row.update({"short": t["skill_short"], "long": t["skill_long"], "cell_r2": t["free"]["r2_denoised"],
                    "brain_r2": brain_r2(net), "brain_r2_W0": brain_r2(net, f"{net}_W0"),
                    "brain_r2_nostim": brain_r2(net, f"{net}_no_stimulus")})
    tn = load(now, "test")
    if tn:
        row.update({"brain_r2_now": brain_r2(now), "cell_r2_now": tn["free"]["r2_denoised"], "short_now": tn["skill_short"]})
    om = os.path.join(DATA, f"omega_{net}.json")
    if os.path.exists(om):
        o = json.load(open(om))
        row.update({"omega_mean": o["mean"], "omega_p5": o["p5"], "omega_p95": o["p95"]})
    fn = load(now, "freetrial")
    if ft and "pulses" in ft["arms"]["full"]:
        pre = ft["window"][0]
        dt = d.get("volume_s") or 1.0
        ev = int(round(EV_S / dt))
        e = lambda Y: float(np.mean(np.asarray(Y)[:, pre:pre + ev].mean(1) - np.asarray(Y)[:, :pre].mean(1))) if len(Y) else None
        sites = sorted({p["site"] for p in ft["arms"]["full"]["pulses"]})
        row["sites"] = {}
        for s in sites:
            P = [p for p in ft["arms"]["full"]["pulses"] if p["site"] == s]
            Yr, Yl = [p["trace_rec"] for p in P], [p["trace_free"] for p in P]
            r_ = {"n": len(P), "rec": e(Yr), "law": e(Yl), "fwhm_rec": fwhm(np.mean(Yr, 0), pre, dt),
                  "fwhm_law": fwhm(np.mean(Yl, 0), pre, dt)}
            if "W0" in ft["arms"]:
                r_["W0"] = e([p["trace_free"] for p in ft["arms"]["W0"]["pulses"] if p["site"] == s])
            if fn and "pulses" in fn["arms"]["full"]:
                r_["now"] = e([p["trace_free"] for p in fn["arms"]["full"]["pulses"] if p["site"] == s])
            row["sites"][str(s)] = r_
        gut = [p for p in ft["arms"]["full"]["pulses"] if p["site"] != CTRL_SITE]
        if gut:
            row["gut"] = {"n": len(gut), "rec": e([p["trace_rec"] for p in gut]), "law": e([p["trace_free"] for p in gut]),
                          "trace_rec": np.mean([p["trace_rec"] for p in gut], 0).tolist(),
                          "trace_law": np.mean([p["trace_free"] for p in gut], 0).tolist(),
                          "sd_rec": np.std([p["trace_rec"] for p in gut], 0).tolist(),
                          "fwhm_rec": fwhm(np.mean([p["trace_rec"] for p in gut], 0), pre, dt),
                          "fwhm_law": fwhm(np.mean([p["trace_free"] for p in gut], 0), pre, dt), "pre": pre, "dt": dt}
            if "W0" in ft["arms"]:
                g0 = [p for p in ft["arms"]["W0"]["pulses"] if p["site"] != CTRL_SITE]
                row["gut"]["W0"] = e([p["trace_free"] for p in g0])
                row["gut"]["trace_W0"] = np.mean([p["trace_free"] for p in g0], 0).tolist()
            if fn and "pulses" in fn["arms"]["full"]:
                gn = [p for p in fn["arms"]["full"]["pulses"] if p["site"] != CTRL_SITE]
                row["gut"]["now"] = e([p["trace_free"] for p in gn])
                row["gut"]["trace_now"] = np.mean([p["trace_free"] for p in gn], 0).tolist()
    return row


def figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    halves = {"a": [r for r in rows if r["condition"] in ("glucose", "glutamate")],
              "b": [r for r in rows if r["condition"] not in ("glucose", "glutamate")]}
    lab = dict(COND)
    for h, rs in halves.items():
        n = len(rs)
        cols = 6 if n <= 12 else 7
        nr = int(np.ceil(n / cols))
        fig, ax = plt.subplots(nr, cols, figsize=(2.4 * cols, 2.3 * nr + 0.4), facecolor="black", squeeze=False)
        for a_ in ax.ravel():
            a_.set_facecolor("black"); a_.axis("off")
        for a_, r in zip(ax.ravel(), rs):
            g = r.get("gut")
            a_.axis("on")
            for sp in a_.spines.values():
                sp.set_color("0.5")
            a_.tick_params(colors="0.7", labelsize=6)
            if not g:
                a_.set_title(f"{r['condition']} {r['fish']}: no data", color="0.7", fontsize=8)
                continue
            tt = (np.arange(len(g["trace_rec"])) - g["pre"]) * g["dt"]
            m, s = np.array(g["trace_rec"]), np.array(g["sd_rec"])
            a_.fill_between(tt, m - s, m + s, color="#4caf50", alpha=0.15, lw=0)
            a_.plot(tt, m, color="#4caf50", lw=1.3)
            a_.plot(tt, g["trace_law"], color="white", lw=1.3)
            if "trace_W0" in g:
                a_.plot(tt, g["trace_W0"], color="#ff5252", lw=0.9)
            if "trace_now" in g:
                a_.plot(tt, g["trace_now"], color="#42a5f5", lw=0.9)
            a_.axvline(0, color="#ffd54f", lw=0.6, ls=":")
            a_.set_ylim(0, 0.7)
            a_.set_title(f"{r['condition'].replace('_', ' ')} {r['fish']}: {g['n']} pulses", color="white", fontsize=8)
        fig.text(0.5, 0.005, "green recorded (mean +- SD over the fish's gut pulses), white the law (free rollout), red the same "
                 "law W = 0, blue trained with no network; s from the pulse; mean dF/F of the gut-responsive cells",
                 color="0.8", fontsize=8, ha="center")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(FIGS, f"compare24_traces_{h}.png"), dpi=160, facecolor="black")
        plt.close(fig)
    # the conditions: the replicable response, the recorded and learned evoked change
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), facecolor="black")
    for a_, (key, ttl) in zip(ax, (("top1_heldout", "replicable response, held-out gut pulses (dF/F)"),
                                   ("rec", "recorded evoked change at the gut sites (dF/F)"),
                                   ("ratio", "the law's evoked change / recorded"))):
        a_.set_facecolor("black")
        for sp in a_.spines.values():
            sp.set_color("0.6")
        a_.tick_params(colors="0.8", labelsize=8)
        for i, (c, cl) in enumerate(COND):
            ys = []
            for r in rows:
                if r["condition"] != c:
                    continue
                g = r.get("gut") or {}
                v = (r.get("top1_heldout") if key == "top1_heldout" else g.get("rec") if key == "rec"
                     else (g["law"] / g["rec"] if g.get("rec") and abs(g["rec"]) > 0.02 else None))
                if v is not None:
                    ys.append(v)
            a_.scatter(np.full(len(ys), i) + np.linspace(-0.12, 0.12, max(len(ys), 1))[:len(ys)], ys, color="white", s=14)
            if ys:
                a_.plot([i - 0.25, i + 0.25], [np.median(ys)] * 2, color="#ffd54f", lw=2)
        a_.axhline(0, color="0.5", lw=0.6)
        a_.set_xticks(range(len(COND)), [c[1].replace(", ", "\n") for c in COND], fontsize=7, color="0.9")
        a_.set_title(ttl, color="white", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "compare24_conditions.png"), dpi=160, facecolor="black")
    plt.close(fig)


def main():
    D = {(d["condition"], d["fish"]): d for d in json.load(open(os.path.join(DATA, "design.json")))}
    rows = [per_fish(c, k, D) for c, _ in COND for k in sorted(kk for cc, kk in D if cc == c)]
    tests = {}
    fm = []
    for r in rows:
        s = r.get("sites", {})
        if "2" in s and "5" in s and s["2"]["rec"]:
            fm.append({"fish": f"{r['condition']} {r['fish']}", "site5_over_site2_rec": s["5"]["rec"] / s["2"]["rec"],
                       "site5_over_site2_law": (s["5"]["law"] / s["2"]["law"]) if s["2"]["law"] else None})
    tests["foregut_midgut"] = fm
    wid = {"gut": [], "vessel": []}
    for r in rows:
        g = r.get("gut")
        if g and g.get("fwhm_rec") is not None and g.get("rec", 0) > 0.03:
            wid["vessel" if r["condition"] == "blood_glucose" else "gut"].append(
                {"fish": f"{r['condition']} {r['fish']}", "fwhm_rec": g["fwhm_rec"], "fwhm_law": g.get("fwhm_law")})
    tests["fwhm"] = wid
    json.dump({"rows": rows, "tests": tests}, open(os.path.join(DATA, "compare24.json"), "w"), indent=1, default=float)
    figures(rows)
    for r in rows:
        g = r.get("gut") or {}
        f = lambda v, fmt="{:+.3f}": fmt.format(v) if isinstance(v, (int, float)) and v is not None and np.isfinite(v) else "--"
        print(f"{r['condition']:14s} {r['fish']} resp {r.get('gut_responsive', '--'):>6} top1 {f(r.get('top1_heldout'))} "
              f"rec {f(g.get('rec'))} law {f(g.get('law'))} W0 {f(g.get('W0'))} now {f(g.get('now'))} | brainR2 {f(r.get('brain_r2'), '{:+.2f}')} "
              f"now {f(r.get('brain_r2_now'), '{:+.2f}')} W0 {f(r.get('brain_r2_W0'), '{:+.2f}')} | omega {f(r.get('omega_mean'), '{:.2f}')}")
    print("[compare24] foregut/midgut:", json.dumps(fm)[:400])
    print("[compare24] fwhm:", {k: [round(x["fwhm_rec"], 1) for x in v] for k, v in wid.items()})


if __name__ == "__main__":
    main()
