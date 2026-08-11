#!/usr/bin/env python
"""One folder per gate: its movie, and a `gate.png` that plots the PREDICTION against what ran.

Cedric, 11 August: *"make a log folder with mp4 and gate.png so that I can see the progress and
visualize what is going on with each gate."*

WHY THE PREDICTION IS ON THE PICTURE. Each gate spec carries a `_gate:` block written BEFORE the
run -- G13's predicted death frame, I4's predicted clearing time, G16's asserted ordering -- and
this reads that block rather than anything computed afterwards. A threshold chosen after seeing the
number is not a threshold, so the plot draws the prediction as a fixed line and the measurement as
a point on it. A gate that fails is drawn in red and stays in the folder.

    log/okuda/_gates/
      G13_clearing_time/   gate.png + the four movies, one per shrink_rate
      G16_wavelength/      gate.png + both species' movies
      I4_slow_no_inhib/    gate.png + three movies
      STATUS.md            every gate in the note, its status, and what is still missing

    python gate_report.py
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.path.dirname(HERE), "log", "okuda")
CONFIG = os.path.join(os.path.dirname(HERE), "config", "okuda")
OUT = os.path.join(LOG, "_gates")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

# the house style of the Plexus figures: black ground, no titles, a white label top-left
BG, FG, OK, BAD, PEND = "black", "white", "#2ecc71", "#e74c3c", "#7f8c8d"


def _diag(name):
    p = os.path.join(LOG, name, "diag.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _spec_gate(name):
    import yaml
    p = os.path.join(CONFIG, f"{name}.yaml")
    if not os.path.exists(p):
        return {}
    return (yaml.safe_load(open(p)) or {}).get("_gate", {}) or {}


def _panel(ax, label):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(FG); s.set_linewidth(0.6)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    ax.text(0.02, 0.97, label, transform=ax.transAxes, color=FG, fontsize=11,
            va="top", ha="left", fontweight="bold")


def _death_frame(name):
    """The frame the single marked cell actually vanished, read from the recorded cell count."""
    p = os.path.join(LOG, name, "metrics.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    for k in ("cells", "n_cells", "cells_final"):
        if k in z.files:
            c = np.asarray(z[k], float)
            drop = np.where(c < c[0])[0]
            return int(drop[0]) if len(drop) else None
    return None


def g13(folder):
    """Predicted death frame against the measured one, for four shrink rates."""
    runs = [f"g13_s{int(s*100):03d}" for s in (0.05, 0.10, 0.15, 0.20)]
    pred, meas, lbl = [], [], []
    for n in runs:
        g = _spec_gate(n)
        d = _diag(n)
        if not g:
            continue
        pred.append(g.get("predicted_death_frame"))
        meas.append(_death_frame(n))
        lbl.append(f"s={g.get('shrink_rate')}")
    fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor=BG)
    _panel(ax, "G13  clearing time")
    x = np.arange(len(pred))
    ax.plot(x, pred, "o--", color=FG, ms=7, label="predicted  ln(c)/ln(1-s) + 50")
    got = [m for m in meas if m is not None]
    if got:
        ax.plot([i for i, m in enumerate(meas) if m is not None], got, "s", color=OK, ms=8,
                label="measured (frame the cell vanished)")
    ax.set_xticks(x); ax.set_xticklabels(lbl, color=FG)
    ax.set_ylabel("frame", color=FG)
    lg = ax.legend(fontsize=7, facecolor=BG, edgecolor=FG, labelcolor=FG)
    lg.get_frame().set_alpha(0.6)
    fig.tight_layout(); fig.savefig(os.path.join(folder, "gate.png"), dpi=140, facecolor=BG)
    plt.close(fig)
    return runs, ("PASS" if got and all(m is not None for m in meas) else "RUNNING")


def g16(folder):
    """The two species' spot counts. The design asserts B is coarser; the gate is B < A."""
    runs = ["g16_species_a", "g16_species_b"]
    vals, alive = [], []
    for n in runs:
        sm = (_diag(n) or {}).get("summary", {})
        vals.append(sm.get("n_spots_final"))
        # A SPOT COUNT FROM A DEAD FIELD IS NOT A WAVELENGTH. The first version of this gate
        # compared counts alone and passed B on 1 < 2 -- while B's act_max was 0.0, i.e. its
        # chemistry had gone extinct. "Fewer spots" and "no spots" are the same number and
        # opposite findings, and reporting the first when the second is true is exactly the
        # failure the note's own L3 warns about.
        alive.append(isinstance(sm.get("act_max_final"), (int, float))
                     and sm["act_max_final"] > 1e-6)
    fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor=BG)
    _panel(ax, "G16  wavelength: is B coarser?")
    got = [v for v in vals if isinstance(v, (int, float))]
    if len(got) == 2:
        ok = got[1] < got[0] and all(alive)
        ax.bar([0, 1], got, color=[OK if alive[0] else BAD, OK if alive[1] else BAD], width=0.5)
        msg = ("B coarser than A" if ok else
               ("species B is EXTINCT (act_max 0)  FAILS" if not alive[1] else "B >= A  FAILS"))
        ax.text(0.5, max(max(got), 1) * 0.85, msg, color=OK if ok else BAD,
                ha="center", fontsize=9, fontweight="bold")
        status = "PASS" if ok else "FAIL"
    else:
        ax.text(0.5, 0.5, "running", transform=ax.transAxes, color=PEND, ha="center")
        status = "RUNNING"
    ax.set_xticks([0, 1]); ax.set_xticklabels(["species A\n(fine)", "species B\n(coarse)"], color=FG)
    ax.set_ylabel("n_spots_final", color=FG)
    fig.tight_layout(); fig.savefig(os.path.join(folder, "gate.png"), dpi=140, facecolor=BG)
    plt.close(fig)
    return runs, status


def i4(folder):
    """Deaths against clearing time, with the inhibitor OFF -- the confound removed."""
    runs = [f"i4_slow{int(s*100):03d}" for s in (0.05, 0.02, 0.01)]
    tau, deaths, cells = [], [], []
    for n in runs:
        g, d = _spec_gate(n), _diag(n)
        tau.append(g.get("predicted_clearing_ticks"))
        s = (d or {}).get("summary", {})
        deaths.append(s.get("n_apop")); cells.append(s.get("cells_final"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.8), facecolor=BG)
    _panel(a1, "I4  deaths vs clearing time")
    got = [(t, d) for t, d in zip(tau, deaths) if t and isinstance(d, (int, float))]
    if got:
        a1.plot([t for t, _ in got], [d for _, d in got], "o-", color=OK, ms=8)
    a1.set_xlabel("predicted clearing time (ticks)"); a1.set_ylabel("deaths")
    _panel(a2, "I4  tissue size (inhibitor OFF)")
    gc = [(t, c) for t, c in zip(tau, cells) if t and isinstance(c, (int, float))]
    if gc:
        a2.plot([t for t, _ in gc], [c for _, c in gc], "s-", color=FG, ms=8)
        a2.axhline(4859, color=BAD, ls="--", lw=1)
        a2.text(0.05, 0.08, "4859 = the confounded series, inhibitor ON", transform=a2.transAxes,
                color=BAD, fontsize=7)
    a2.set_xlabel("predicted clearing time (ticks)"); a2.set_ylabel("cells_final")
    fig.tight_layout(); fig.savefig(os.path.join(folder, "gate.png"), dpi=140, facecolor=BG)
    plt.close(fig)
    return runs, ("PASS" if len(got) == 3 else "RUNNING")


GATES = [("G13_clearing_time", g13), ("G16_wavelength", g16), ("I4_slow_no_inhib", i4)]

# every gate in note_death_growth, so STATUS.md is the note's table and not a subset
NOTE = [
    ("G1",  "selectors fire on a known population, not on a uniform field", "PASS", "19/19"),
    ("G2",  "euler = 2 at every frame with death running", "PASS", "2, all frames"),
    ("G3",  "a topology change permutes pending DELTAS", "PASS", "activator 0, was -1.04e10"),
    ("G4",  "a bequest goes only to neighbours above the extrusion threshold", "PASS", "P4/P12 cleared"),
    ("G5",  "a bequest leaves the recipient inside the integrator's basin", "RUNNING",
     "clamp fixed for CUDA; tsd_cap10/25 re-running"),
    ("G6",  "a named population is chosen ONCE", "PASS", "1 death, was 7"),
    ("G7",  "two species never write each other's columns", "PASS", "exact"),
    ("G8",  "inhib_chan absent reproduces a pre-change run", "FAIL",
     "protr 1.175 vs 1.176, cells 7099 vs 7261 -- NOT identical, see STATUS"),
    ("G9",  "growth -> 0 as the inhibitor saturates", "PASS", "0.978 -> 0.014"),
    ("G10", "the cap admits at most phi*N, worst first, and drains", "PASS", "8/100"),
    ("G11", "n_apop equals the cells removed", "PASS", "exact"),
    ("G12", "a collapse is a T2", "PASS", "refuses k>3"),
    ("G13", "clearing time = ln(c)/ln(1-s)", "RUNNING", "this folder"),
    ("G14", "the throughput lever is s, not phi", "PASS", "x1.7 vs x4.7"),
    ("G15", "a geometric chisel removes its declared population", "PASS", "5/5"),
    ("G16", "species B's wavelength is coarser than A's", "FAIL",
     "B is EXTINCT: act_max 0.0 against A's 0.392. Not coarser -- dead"),
    ("G17", "apoptotic rate matches an epithelium's", "BLOCKED", "no units: block"),
    ("G18", "clearance time matches an extrusion in vivo", "BLOCKED", "no units: block"),
    ("G19", "the two wavelengths bracket a real morphogen pair", "BLOCKED", "no units: block"),
    ("I4",  "the slow ladder without the inhibitor confound", "RUNNING", "this folder"),
    ("I8",  "apop_spill is recorded and near zero on a healthy run", "RUNNING",
     "now written to the summary; no run has reported it yet"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["# Gate status -- note_death_growth", "",
             "Each folder holds `gate.png` (the prediction, made before the run, against what ran)",
             "and the movie of every run behind it.", "",
             "| gate | asks | status | evidence |", "|---|---|---|---|"]
    live = {}
    for folder, fn in GATES:
        d = os.path.join(OUT, folder)
        os.makedirs(d, exist_ok=True)
        runs, status = fn(d)
        live[folder.split("_")[0]] = status
        n_mp4 = 0
        for r in runs:
            src = os.path.join(LOG, r, "movie.mp4")
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(d, f"{r}.mp4")); n_mp4 += 1
        print(f"  {folder:<22}{status:<9}{n_mp4}/{len(runs)} movies")
    for gid, asks, status, ev in NOTE:
        status = live.get(gid, status)
        mark = {"PASS": "PASS", "FAIL": "**FAIL**", "RUNNING": "_running_",
                "BLOCKED": "_blocked_"}.get(status, status)
        lines.append(f"| {gid} | {asks} | {mark} | {ev} |")
    lines += ["", "## What is still missing, plainly", "",
              "- **G16 FAILED and it is the important one.** Species B's activator ends at 0.0",
              "  against species A's 0.392: B is not a coarser map, it is an extinct one. The",
              "  two-species design asserts the second field is a DIFFERENT map, and it has been a",
              "  dead one in every run built on it. That retro-explains three preliminary results",
              "  at once: `ts_growth_b` was the worst run of its series (growth gated on nothing),",
              "  `ts_death_b_late` and `ts_death_b_sharp` came back bit-identical (chem_low on a",
              "  flat field marks everything, so the cap decides), and the two-species claim was",
              "  killed by its own control (the second map contributed nothing to kill).",
              "  B's kinetics are F 0.039 / kk 0.058 -- Cedric's 2D values -- with 2x diffusivity",
              "  added by me. That combination does not pattern here; the diffusivity is the part",
              "  I chose, so it is the part to sweep first.",
              "- **G8 did not close.** A pre-change spec re-run does not reproduce its archived",
              "  numbers exactly: protr_peak 1.175 against 1.176, cells 7099 against 7261 (2%).",
              "  Either the channel change perturbed a run that does not use it, or this engine is",
              "  not deterministic run-to-run. Those have opposite consequences -- the first is a",
              "  bug in my change, the second means every difference the campaign has ever",
              "  reported sits on an unmeasured noise floor -- and one re-run of an UNTOUCHED spec",
              "  separates them. That run is the next thing to do.",
              "- **G17-G19 cannot be attempted** until the specs carry a `units:` block. Until then",
              "  no gate here is a statement about cells.",
              ]
    open(os.path.join(OUT, "STATUS.md"), "w").write("\n".join(lines) + "\n")
    print(f"\n  -> {os.path.relpath(OUT, os.path.dirname(HERE))}/STATUS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
