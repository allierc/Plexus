#!/usr/bin/env python
"""phase3 -- buy the authority back, ONE FACTOR AT A TIME, re-probing after each.

    python phase3.py --stage a --device cuda:0        # length
    python phase3.py --stage b --device cuda:0        # + pulley
    python phase3.py --stage c --device cuda:0        # + drive
    python phase3.py --compare                        # the table + the verdicts

Phase 2 measured that this plant is short of authority by 7x, so no controller recovers it. Phase 3
goes after the mechanics, using the Phase-2 probe as the instrument -- the first thing in this
campaign that passed a fidelity check rather than failing one.

The three interventions were found in Phase 0 by closed-loop guessing, which never learned what any
one of them was worth. Each is applied on top of the last, and the full six-muscle probe is repeated
after each, so every one gets a measured delta on the static gain matrix:

    a   LENGTH   frac 0.55 -> 0.95, sclera stand-off 0.020 -> 0.042
    b   PULLEY   + muscle_sleeve
    c   DRIVE    + A/E 0.25 -> 0.28 (the top of the range the pulley was measured to allow)

REGISTERED PREDICTIONS (eye_note.pdf, before running):
    a  gains roughly double: LR horizontal 4.25 -> 7-9, SO torsion 1.10 -> 1.9-2.4 deg/act
    b  THE SHARP ONE. If muscle_sleeve is a purely transverse anti-buckling constraint and nothing
       else, it must leave the STATIC gains under 20% changed on every entry. If they move a lot,
       the story this note tells about the sleeve is wrong.
    c  gains are linear in A: +12%, no more.
    headline  compounded, LR reaches ~8-9 deg, still nowhere near the 26 deg commanded -- i.e. all
       three Phase-0 fixes together do NOT make this plant able to reach its commands.

A NOTE ON THE BASELINE, which is a defect worth naming. `muscle_morphogenesis` reads the muscle
origins from `eye_anatomy.origins_world()` at RUN time, so an archived spec.yaml does not fully
determine its own run: t03_c_a was generated before ANNULUS_RING existed and recorded rest length
LR = 0.259, but re-running that same spec today gives 0.242. The origin-spreading intervention is
therefore already present in the Phase-2 baseline and is NOT a Phase-3 variable -- its effect was
never measured and cannot be recovered without reverting the module. Geometry that a spec does not
carry is geometry the spec cannot reproduce.
"""
from __future__ import annotations

import os
import sys
import json
import copy
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml

import probe_plant
import eye_anatomy as EA

BASE = os.path.join(HERE, "archive", "t03_c_a", "spec.yaml")
ARCH = os.path.join(HERE, "archive")

STAGES = {
    "a": dict(tag="p3a_length", frac=0.95, gap=0.042, sleeve=False, amplitude=60.0),
    "b": dict(tag="p3b_pulley", frac=0.95, gap=0.042, sleeve=True, amplitude=60.0),
    "c": dict(tag="p3c_drive", frac=0.95, gap=0.042, sleeve=True, amplitude=67.0),
}


def variant(base, st):
    """Apply one stage's interventions to the baseline spec."""
    s = copy.deepcopy(base)
    ops = []
    for o in s["operators"]:
        o = dict(o)
        if o["op"] == "muscle_morphogenesis":
            o["frac"] = float(st["frac"])
            o["gap"] = float(st["gap"])
        if o["op"] == "muscle_contract":
            o["amplitude"] = float(st["amplitude"])
        ops.append(o)
    if st["sleeve"] and not any(o["op"] == "muscle_sleeve" for o in ops):
        i = next(i for i, o in enumerate(ops) if o["op"] == "bone_anchor")
        ops.insert(i + 1, {"op": "muscle_sleeve", "at": "muscle_particle", "k": 2500.0,
                           "c": 30.0, "free_from": 0.70, "free_to": 0.88})
        j = s["schedule"].index("bone_anchor")
        s["schedule"] = s["schedule"][:j + 1] + ["muscle_sleeve"] + s["schedule"][j + 1:]
    s["operators"] = ops
    s["general"] = dict(s["general"])
    s["general"]["name"] = f"eye_{st['tag']}"
    return s


def run_stage(stage, device, muscles, frames, t_on, t_off, stride, movie):
    st = STAGES[stage]
    base = yaml.safe_load(open(BASE))
    spec = variant(base, st)
    tonic = float(next(o for o in base["operators"]
                       if o["op"] == "oculomotor_drive").get("tonic", 0.14))
    outdir = os.path.join(ARCH, f"phase3_{st['tag']}")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "variant_spec.yaml"), "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, width=100)
    print(f"[phase3-{stage}] {st['tag']}: frac={st['frac']} gap={st['gap']} "
          f"sleeve={st['sleeve']} A={st['amplitude']} (A/E={st['amplitude'] / 240:.3f})", flush=True)
    for m in muscles:
        probe_plant.run_probe(spec, m, device, 1.0, tonic, t_on, t_off, frames, outdir,
                              stride=stride, movie=movie)
    return outdir


def gain_matrix(outdir, t_on, t_off, dt):
    G = np.full((3, EA.N_MUSCLE), np.nan)
    rows = {}
    for m, key in enumerate(EA.MUSCLE_KEYS):
        f = os.path.join(outdir, f"probe_{key}.npz")
        if not os.path.exists(f):
            continue
        d = dict(np.load(f))
        d["_muscle"] = m
        r = probe_plant.fit_probe(d, t_on, t_off, dt)
        if r is None:
            continue
        rows[key] = r
        G[:, m] = r["gain_deg_per_act"]
    return G, rows


def compare(t_on, t_off, dt, tonic=0.14, command=26.0):
    """The Phase-3 table, and each registered prediction judged against it."""
    stages = [("phase 2 baseline", os.path.join(ARCH, "phase2_stepresponse"))]
    stages += [(STAGES[k]["tag"], os.path.join(ARCH, f"phase3_{STAGES[k]['tag']}"))
               for k in ("a", "b", "c")]
    mats, allrows = {}, {}
    for name, d in stages:
        if not os.path.isdir(d):
            continue
        G, rows = gain_matrix(d, t_on, t_off, dt)
        if np.all(np.isnan(G)):
            continue
        mats[name] = G
        allrows[name] = rows

    axes = ("horizontal", "vertical", "torsion")
    print("\n" + "=" * 100)
    print("PHASE 3 -- STATIC GAIN (deg per unit activation) after each intervention")
    for name, G in mats.items():
        print(f"\n  {name}")
        print(f"{'':14}" + "".join(f"{k:>11}" for k in EA.MUSCLE_KEYS))
        for k, nm in enumerate(axes):
            print(f"  {nm:12}" + "".join(f"{G[k, m]:11.2f}" for m in range(EA.N_MUSCLE)))
    print("=" * 100)

    base = mats.get("phase 2 baseline")
    verdicts = {}
    if base is not None:
        # (a) length: LR horizontal 7-9, SO torsion 1.9-2.4
        A = mats.get("p3a_length")
        if A is not None:
            lr, so = abs(A[0, 0]), abs(A[2, 4])
            verdicts["a_length"] = {
                "lr_horizontal": round(float(lr), 2), "predicted": [7.0, 9.0],
                "held": bool(7.0 <= lr <= 9.0),
                "so_torsion": round(float(so), 2), "so_predicted": [1.9, 2.4],
                "so_held": bool(1.9 <= so <= 2.4),
                "lr_ratio_to_baseline": round(float(lr / abs(base[0, 0])), 2)}
            print(f"\n(a) LENGTH   LR horizontal {lr:.2f} (predicted 7-9): "
                  f"{'HELD' if verdicts['a_length']['held'] else 'BROKEN'}"
                  f"   x{verdicts['a_length']['lr_ratio_to_baseline']} on baseline")
            print(f"             SO torsion    {so:.2f} (predicted 1.9-2.4): "
                  f"{'HELD' if verdicts['a_length']['so_held'] else 'BROKEN'}")
        # (b) the sharp one: sleeve must not move the STATIC gains more than 20%
        B = mats.get("p3b_pulley")
        if A is not None and B is not None:
            ref = np.abs(A)
            big = ref > 0.30                      # only judge entries big enough to be meaningful
            rel = np.abs(B - A) / np.maximum(ref, 1e-9)
            worst = float(np.nanmax(np.where(big, rel, np.nan)))
            verdicts["b_pulley"] = {"worst_relative_change": round(worst, 3),
                                    "threshold": 0.20, "held": bool(worst < 0.20)}
            print(f"\n(b) PULLEY   worst change on a meaningful entry {100 * worst:.0f}% "
                  f"(predicted <20%): {'HELD' if worst < 0.20 else 'BROKEN'}")
            if worst >= 0.20:
                print("             => the sleeve is NOT purely transverse. The story this note "
                      "tells about it is wrong,\n                and Phase 0's attribution of the "
                      "A/E ceiling to it needs re-examining.")
        # (c) drive: linear in A, +12%
        C = mats.get("p3c_drive")
        if B is not None and C is not None:
            ref = np.abs(B)
            big = ref > 0.30
            rel = np.where(big, (np.abs(C) - ref) / np.maximum(ref, 1e-9), np.nan)
            med = float(np.nanmedian(rel))
            verdicts["c_drive"] = {"median_relative_change": round(med, 3),
                                   "predicted": 0.117, "held": bool(0.03 <= med <= 0.22)}
            print(f"\n(c) DRIVE    median gain change {100 * med:+.0f}% (predicted +12%): "
                  f"{'HELD' if verdicts['c_drive']['held'] else 'BROKEN'}")
        # the headline
        final = mats.get("p3c_drive", mats.get("p3b_pulley", mats.get("p3a_length")))
        if final is not None:
            reach = abs(final[0, 0]) * (1.0 - tonic)
            verdicts["headline"] = {
                "lr_reach_deg": round(float(reach), 2), "command_deg": command,
                "predicted_band": [8.0, 9.0],
                "still_short": bool(reach < command),
                "prediction_held": bool(8.0 <= reach <= 9.0)}
            print(f"\nHEADLINE     LR reaches {reach:.1f} deg at full activation against a "
                  f"{command:.0f} deg command")
            print(f"             predicted 8-9 deg and still short: "
                  f"{'HELD' if verdicts['headline']['prediction_held'] else 'BROKEN'}"
                  f"; still short: {'YES' if reach < command else 'NO'}")
            if reach >= command:
                print("             => the plant CAN reach its commands after all. The Phase-2 "
                      "conclusion stands for\n                the baseline but not for the fixed "
                      "plant, and a controller is worth tuning again.")

    out = {"gain_matrices": {k: v.tolist() for k, v in mats.items()}, "verdicts": verdicts}
    with open(os.path.join(ARCH, "phase3_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[phase3] -> {os.path.join(ARCH, 'phase3_summary.json')}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--muscles", type=int, nargs="*", default=list(range(EA.N_MUSCLE)))
    ap.add_argument("--frames", type=int, default=560)
    ap.add_argument("--t_on", type=int, default=60)
    ap.add_argument("--t_off", type=int, default=480)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()

    dt = float(yaml.safe_load(open(BASE))["general"]["dt"])
    if a.stage:
        run_stage(a.stage, a.device, a.muscles, a.frames, a.t_on, a.t_off,
                  a.stride, not a.no_movie)
    if a.compare or not a.stage:
        compare(a.t_on, a.t_off, dt)


if __name__ == "__main__":
    main()
