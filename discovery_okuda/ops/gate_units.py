#!/usr/bin/env python
"""gate_units -- the numbers of 01, 02, 03 and 04 in micrometres, seconds and pascals.

    python gate_units.py    ->  log/okuda_ECM/0u_units/{units.json,units.png}

WHY THIS IS A GATE AND NOT A CONVERSION TABLE. Every note in this folder compares a simulation
number with a measured one: a modulus against Candiello's $0.4$--$3$ MPa, a turnover time against
collagen IV's $3$--$10$ hours, a penetration against a basement membrane's $100$ nm. Until the run
declares a scale, each of those comparisons is between a number and a quantity, and `plexus/units.py`
exists precisely because that has already cost this codebase three real errors -- a membrane 24x too
thick, a modulus quoted as a pressure, and a turnover time whose hours depended on an unstated
minutes-per-frame.

So the gate is: DECLARE the three base scales, DERIVE everything, and ask whether what comes out
lands where the literature says it should. A conversion that lands outside the range is not a
failed unit test -- it is the model saying it is not the system it claims to be, in the one
currency where that can be checked.

THE CALIBRATION, and each of the three is fixed by ONE measured thing, not fitted:

  length   by CELL SIZE, which the vertex model resolves. 6,076 cells on a sphere of radius 16.56
           tissue units is 0.567 tissue^2 of apical surface each, so a cell is 0.85 tissue units
           across; an epithelial cell is ~8.5 um, giving 10 um per tissue unit. The MPM box is
           1/0.008530 = 117.2 tissue units wide, hence 1172 um.
  time     by the CELL CYCLE. 200 -> 6,076 cells is 4.93 doublings over 401 frames; a 12-24 h cycle
           puts the run at 2.5-4.9 days and one frame at 8.8-17.7 min. 600 s is taken as the
           nominal and the RANGE is carried through every rate below.
  force    by the STROMA. `youngs: 15` is the one material constant with a literature range that is
           narrow and uncontested -- a collagen I gel at 1-3 mg/ml is 10-100 Pa (Yang 2009) -- so
           force_nN is set to put the stroma at 100 Pa, and everything else in force units follows.
           This is the only place a scale is CHOSEN rather than measured, and it is chosen on the
           softest, best-measured body in the model rather than on the sheet, which is the one under
           dispute.

WHAT WOULD FALSIFY IT. Any derived quantity landing outside its literature range while the three
above are held. Those are the rows of the table this writes.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

from plexus.units import Units                                       # noqa: E402

LOG = os.path.join(_ROOT, "log", "okuda_ECM")

# --- the calibration, from the docstring ------------------------------------------------------
TISSUE_UM = 10.0                     # um per TISSUE unit, from cell size
SCALE = 0.008530                     # box units per tissue unit (04's `scale`)
BOX_UM = TISSUE_UM / SCALE           # um per BOX unit
FRAME_S = 600.0                      # s per frame, nominal
FRAME_S_RANGE = (530.0, 1060.0)      # 8.8-17.7 min, from a 12-24 h cycle over 4.93 doublings
STROMA_E_SIM = 15.0                  # `youngs` of the stroma
STROMA_E_PA = 100.0                  # the pascal it is calibrated to


def _panel(ax, letter):
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")


def build():
    """The three base scales, and the `Units` object each note's run would declare."""
    # stress_Pa = 1e3 * force_nN / length_um^2, and we want STROMA_E_SIM stress units = STROMA_E_PA
    stress_per_unit = STROMA_E_PA / STROMA_E_SIM
    force_nN = stress_per_unit * BOX_UM ** 2 / 1.0e3
    U = Units(length_um=BOX_UM, time_s=FRAME_S, force_nN=force_nN, declared=True)
    # the epithelium is solved in TISSUE units, not box units, so it declares its own length scale
    Ut = Units(length_um=TISSUE_UM, time_s=FRAME_S, force_nN=None, declared=True)
    return U, Ut


def rows(U, Ut):
    """One row per quantity a note quotes with a unit. `lo`/`hi` are the literature range."""
    dx_box = 1.0 / 64
    R = []

    def add(note, name, sim, phys, unit, lo, hi, src):
        R.append(dict(note=note, name=name, sim=sim, phys=float(phys), unit=unit,
                      lo=lo, hi=hi, ok=bool(lo <= phys <= hi), source=src))

    # ---- 01, the junction: everything here is a TIME
    add("01", "one frame", 1.0, U.time_s / 60.0, "min", 8.8, 17.7,
        "4.93 doublings at a 12-24 h cycle")
    add("01", "tau_jun = 20 frames", 20.0, 20 * U.time_s / 60.0, "min", 30.0, 600.0,
        "E-cadherin maturation, tens of min to h -- NOT myosin turnover (1-2 min)")
    add("01", "tau_med = 20 frames", 20.0, 20 * U.time_s / 60.0, "min", 30.0, 600.0,
        "same; the medioapical pool is not a faster pool in this model")
    # a T1 exchange per cell: 0.00373 per cell per frame -> the time between exchanges for one cell
    add("01", "time between T1s, per cell", 1 / 0.00373, U.time_s / 0.00373 / 3600.0, "h", 20.0,
        200.0, "neighbour exchange is rare per cell over a 2.5-5 day run")

    # ---- 02, the matrix: a LENGTH, a STRESS and a RATE
    add("02", "grid cell dx", dx_box, dx_box * U.length_um, "um", 5.0, 50.0,
        "must resolve the tissue (a cell is ~8.5 um) without resolving a fibril")
    add("02", "strand length 0.12 box", 0.12, 0.12 * U.length_um, "um", 20.0, 500.0,
        "collagen fibril bundles, tens to hundreds of um")
    add("02", "stroma E = 15", 15.0, 15.0 * U.stress_Pa, "Pa", 10.0, 1000.0,
        "collagen I gel at 1-3 mg/ml, Yang 2009 (this is the CALIBRATION row)")
    add("02", "drag 8, decay time 1/drag", 0.125, 0.125 * U.time_s, "s", 1.0, 1000.0,
        "a hydrated network dissipates fast; it must not ring over a frame")
    add("02", "substep dt_sub", 4.0e-4, 4.0e-4 * U.time_s, "s", 0.01, 10.0,
        "must resolve the elastic wave; dx/c sets the ceiling")

    # ---- 03/04, the interface: a LENGTH and a SPEED
    add("03", "penetration 0.82 cells", 0.82 * dx_box, 0.82 * dx_box * U.length_um, "um", 0.0, 10.0,
        "must be under a cell diameter or the surface is not a surface")
    add("03", "penetration 1.47 cells (03b)", 1.47 * dx_box, 1.47 * dx_box * U.length_um, "um",
        0.0, 30.0, "the flat rig presses harder; still under a few cells")
    add("04", "spheroid radius, final", 0.15, 0.15 * U.length_um, "um", 50.0, 400.0,
        "a 6,000-cell spheroid is 100-200 um in radius")
    add("04", "spheroid radius, initial", 0.0398, 0.0398 * U.length_um, "um", 20.0, 100.0,
        "200 cells")
    add("04", "matrix shell outer radius", 0.40, 0.40 * U.length_um, "um", 200.0, 2000.0,
        "the gel around a spheroid, hundreds of um")
    add("04", "surface speed", 5.5e-4, 5.5e-4 * U.length_um / U.time_s * 3600.0, "um/h", 0.5, 20.0,
        "a spheroid tripling its radius over days")
    # the epithelium's own scale
    add("01", "junction length, final", 0.46, 0.46 * Ut.length_um, "um", 2.0, 15.0,
        "a cell-cell contact, a few um")
    add("01", "cell diameter, final", 0.85, 0.85 * Ut.length_um, "um", 5.0, 15.0,
        "an epithelial cell (this is the CALIBRATION row)")
    return R


def main():
    d = os.path.join(LOG, "0u_units")
    os.makedirs(d, exist_ok=True)
    U, Ut = build()
    print(f"[units] box: {U.describe()}", flush=True)
    print(f"[units] tissue: {Ut.describe()}", flush=True)
    R = rows(U, Ut)
    for r in R:
        print(f"  {'PASS' if r['ok'] else 'FAIL'}  {r['note']}  {r['name']:32s} "
              f"{r['sim']:10.4g} -> {r['phys']:10.4g} {r['unit']:6s} "
              f"[{r['lo']:g}, {r['hi']:g}]", flush=True)
    n_fail = sum(1 for r in R if not r["ok"])
    out = dict(
        calibration=dict(tissue_um=TISSUE_UM, box_um=BOX_UM, frame_s=FRAME_S,
                         frame_s_range=list(FRAME_S_RANGE), force_nN=U.force_nN,
                         stress_Pa_per_unit=U.stress_Pa, tension_N_per_m=U.tension_N_per_m,
                         box_describe=U.describe(), tissue_describe=Ut.describe()),
        rows=R,
        gates=dict(
            U1=dict(note="01", threshold="every time quoted in the note lands in its range",
                    measured=sum(1 for r in R if r["note"] == "01" and not r["ok"]),
                    unit="failures"),
            U2=dict(note="02", threshold="the matrix's length, stress and rate land in range",
                    measured=sum(1 for r in R if r["note"] == "02" and not r["ok"]),
                    unit="failures"),
            U3=dict(note="03", threshold="penetration is under a cell diameter in um",
                    measured=sum(1 for r in R if r["note"] == "03" and not r["ok"]),
                    unit="failures"),
            U4=dict(note="04", threshold="the spheroid and its shell are the size of a spheroid",
                    measured=sum(1 for r in R if r["note"] == "04" and not r["ok"]),
                    unit="failures")))
    json.dump(out, open(os.path.join(d, "units.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6), facecolor="white")
    # a: every quantity against its literature range, on a log axis, normalised to the range
    lab = [f"{r['note']}  {r['name']}" for r in R]
    y = np.arange(len(R))
    for i, r in enumerate(R):
        ax[0].plot([r["lo"], r["hi"]], [i, i], color="#cfd8dc", lw=6, solid_capstyle="butt")
        ax[0].plot([r["phys"]], [i], "o", ms=6,
                   color="#1B7F4B" if r["ok"] else "#B03A2E")
    ax[0].set_yticks(y); ax[0].set_yticklabels(lab, fontsize=6.5)
    ax[0].set_xscale("log"); ax[0].invert_yaxis()
    ax[0].set_xlabel("physical value (um, min, h, Pa, s, um/h -- see the table)")
    _panel(ax[0], "a")
    # b: the calibration itself, and what it makes of the three base scales
    ax[1].axis("off")
    txt = [f"length   1 box unit = {BOX_UM:.0f} um     (1 tissue unit = {TISSUE_UM:g} um, by cell size)",
           f"time     1 frame    = {FRAME_S:g} s = {FRAME_S/60:.0f} min   "
           f"(range {FRAME_S_RANGE[0]/60:.1f}-{FRAME_S_RANGE[1]/60:.1f} min)",
           f"force    1 unit     = {U.force_nN:.4g} nN   (set so stroma E = {STROMA_E_PA:g} Pa)",
           "",
           f"stress   1 unit     = {U.stress_Pa:.4g} Pa",
           f"tension  1 unit     = {U.tension_N_per_m:.4g} N/m",
           f"energy   1 unit     = {U.energy_aJ:.4g} aJ",
           "",
           f"{len(R) - n_fail} of {len(R)} quantities land in their literature range."]
    ax[1].text(0.0, 0.98, "\n".join(txt), va="top", family="monospace", fontsize=8.5,
               transform=ax[1].transAxes)
    _panel(ax[1], "b")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(d, "units.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[units] {len(R) - n_fail}/{len(R)} in range -> {d}", flush=True)


if __name__ == "__main__":
    main()
