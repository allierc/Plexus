#!/usr/bin/env python
"""p1c_refute -- PROBE C. An adversarial attempt to break "E is unidentifiable from this observable".

P0 concluded that a per-cell Young's modulus is nearly invisible, on the strength of ONE sweep
(uniform E over 40x), ONE instrument (`peak_excursion`, read as a raw amplitude), ONE configuration
(drag 30, period 150, amplitude 20) and ONE beat. Each of those is a place the conclusion could be
wrong, and each is attacked here separately.

Everything is scored through `accept.score_one`, i.e. through `cite()`, so only the four certified
instruments carry any of it. The objective is printed beside them and never used to conclude
anything.

  --do instruments   all four instruments across a uniform-E sweep, against the theta_true rollout
  --do dense         20 points over [100, 600]: is the turning point at E~234 real?
  --do regime        drag / pacemaker period / drive amplitude: is E invisible only HERE?
  --do long          three beats instead of one, scored as three ticks through accept()
  --do checker       THE MAIN ATTACK: a spatial CONTRAST in E against its best uniform impostor
  --do determinism   two identical rollouts, to prove a step is a step and not simulator noise
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p1c_lib as L                                                       # noqa: E402
import accept as ACC                                                      # noqa: E402

INS = L.INSTRUMENTS
T0 = time.time()


def log(s=""):
    print(f"[{time.time() - T0:7.1f}s] {s}" if s else "")
    sys.stdout.flush()


def dump(name, obj):
    p = os.path.join(HERE, f"p1c_{name}.json")
    json.dump(obj, open(p, "w"), indent=1, default=float)
    log(f"  -> {p}")


def _row(sim, ref):
    r = L.steps_row(sim, ref)
    r["amp_reading"] = L.amp_reading(sim)
    r["path_reading"] = L.path_reading(sim)
    return r


def _hdr(first="E"):
    return (f"  {first:>10s} " + "".join(f"{n[:11]:>12s}" for n in INS)
            + f"{'STAT':>9s}{'loopscore':>11s}{'peak_exc':>11s}")


def _line(k, r, fmt="{:10.2f}"):
    cells = "".join((f"{r[n]:12.2f}" if isinstance(r.get(n), float) else f"{'undef':>12s}")
                    for n in INS)
    ls = r.get("loopscore")
    return (f"  {fmt.format(k)} " + cells + f"{r['STAT']:9.2f}"
            + (f"{ls:11.4f}" if isinstance(ls, float) else f"{'--':>11s}")
            + f"{r['amp_reading']:11.6f}")


def _span(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), float)]
    if len(v) < 2:
        return None
    d = np.diff(v)
    return {"min": float(min(v)), "max": float(max(v)), "span": float(max(v) - min(v)),
            "monotone": bool(np.all(d > 0) or np.all(d < 0)),
            "n_turning_points": int((np.diff(np.sign(d)) != 0).sum())}


# ============================================================================================= #
def do_instruments(a):
    """ATTACK 1. peak_excursion is one of four. Read all four across the same sweep."""
    rig = L.Rig(L.default_args(device=a.device), quiet=False, log=log)
    ref = rig.roll(rig.theta_true)
    log(f"reference = theta_true rollout, {ref.shape[0]} frames x {ref.shape[1]} probes; "
        f"peak_excursion reading {L.amp_reading(ref):.6g}, path_length {L.path_reading(ref):.6g}")
    log(f"planted E: min {rig.E_true.min():.1f} mean {rig.E_true.mean():.1f} "
        f"median {rig.E_true.median():.1f} max {rig.E_true.max():.1f}")

    grid = np.geomspace(20.0, 800.0, 13)
    rows = []
    log("\n  UNIFORM E, gain = gain_true, scored against the theta_true rollout, in steps")
    log(_hdr("E"))
    for E in grid:
        r = _row(rig.roll(rig.theta(E=E)), ref)
        r["E"] = float(E)
        rows.append(r)
        log(_line(E, r))
    nul = L.null_row()
    log("  " + "-" * 96)
    log(f"  {'KNOWS NOTHING':>10s} " + "".join(f"{nul[n]:12.2f}" for n in INS)
        + f"{min(nul.values()):9.2f}")

    out = {"rows": rows, "null": nul,
           "span_steps": {n: _span(rows, n) for n in list(INS) + ["STAT"]},
           "span_readings": {"peak_excursion": _span(rows, "amp_reading"),
                             "path_length": _span(rows, "path_reading")},
           "planted": {"min": float(rig.E_true.min()), "max": float(rig.E_true.max()),
                       "mean": float(rig.E_true.mean()), "median": float(rig.E_true.median())},
           "reference_readings": {"peak_excursion": L.amp_reading(ref),
                                  "path_length": L.path_reading(ref)}}
    log("\n  SPAN over the 40x sweep, per instrument (max - min of the steps column):")
    for n in list(INS) + ["STAT"]:
        s = out["span_steps"][n]
        if s:
            log(f"    {n:<20s} span {s['span']:8.2f} steps   range [{s['min']:.2f}, {s['max']:.2f}]"
                f"   monotone {str(s['monotone']):<5s}  turning points {s['n_turning_points']}"
                f"   null {nul.get(n, float('nan')):.2f}")
    rig.free()
    dump("instruments", out)
    return out


# ============================================================================================= #
def do_dense(a):
    """ATTACK 2. Is the turning point near E = 234 real, and is it a resonance?

    A resonance of a driven sheet MOVES when the drive period or the drag changes. A numerical
    artefact does not. Three configurations, the same dense sweep in each.
    """
    grid = np.geomspace(100.0, 600.0, 20)
    configs = [("baseline  drag30 period150", {}),
               ("period 75 (drive x2 faster)", {"period": 75.0, "duration": 15.0}),
               ("period 300 (drive x2 slower)", {"period": 300.0, "duration": 60.0}),
               ("drag 3 (x10 less)", {"drag_k": 3.0}),
               ("drag 300 (x10 more)", {"drag_k": 300.0})]
    out = {"grid": [float(x) for x in grid], "configs": {}}
    for name, kw in configs:
        per = kw.get("period", 150.0)
        args = L.default_args(device=a.device, warmup=int(per) + 30, window=int(per))
        rig = L.Rig(args, **kw)
        ref = rig.roll(rig.theta_true)
        log(f"\n  {name}   (warmup {args.warmup}, window {args.window})")
        log(_hdr("E"))
        rows = []
        for E in grid:
            r = _row(rig.roll(rig.theta(E=E)), ref)
            r["E"] = float(E)
            rows.append(r)
            log(_line(E, r))
        amp = np.array([r["amp_reading"] for r in rows])
        # where the amplitude turns, read off the sweep itself
        k = int(np.argmax(amp))
        turn = float(grid[k]) if 0 < k < len(grid) - 1 else None
        d = np.diff(amp)
        out["configs"][name] = {
            "spec": kw, "rows": rows,
            "amp": [float(v) for v in amp],
            "turning_E": turn, "argmax_E": float(grid[k]),
            "n_sign_changes": int((np.diff(np.sign(d)) != 0).sum()),
            "monotone": bool(np.all(d > 0) or np.all(d < 0)),
            "amp_span_frac": float(amp.max() / amp.min() - 1.0),
            "span_steps": {n: _span(rows, n) for n in list(INS) + ["STAT"]}}
        log(f"    argmax of peak_excursion at E = {grid[k]:.1f}"
            + ("  (interior -> a real turning point)" if turn else "  (at an endpoint)")
            + f";  amplitude range {amp.min():.6g} .. {amp.max():.6g}"
              f" = {100 * (amp.max() / amp.min() - 1):.1f}%")
        rig.free()
    log("\n  DOES THE TURNING POINT MOVE?")
    for n, c in out["configs"].items():
        log(f"    {n:<32s} argmax E = {c['argmax_E']:7.1f}   sign changes {c['n_sign_changes']}"
            f"   amplitude range {100 * c['amp_span_frac']:5.1f}%")
    dump("dense", out)
    return out


# ============================================================================================= #
def do_regime(a):
    """ATTACK 3. Is E invisible in this SHEET, or only in this SPEC?"""
    grid = np.geomspace(20.0, 800.0, 8)
    configs = [("baseline", {}),
               ("drag 0.3", {"drag_k": 0.3}), ("drag 3", {"drag_k": 3.0}),
               ("drag 300", {"drag_k": 300.0}), ("drag 3000", {"drag_k": 3000.0}),
               ("amplitude 5", {"amplitude": 5.0}), ("amplitude 80", {"amplitude": 80.0}),
               ("period 75", {"period": 75.0, "duration": 15.0}),
               ("period 300", {"period": 300.0, "duration": 60.0}),
               ("duration 150 (always on)", {"duration": 150.0})]
    out = {"grid": [float(x) for x in grid], "configs": {}}
    for name, kw in configs:
        per = kw.get("period", 150.0)
        args = L.default_args(device=a.device, warmup=int(per) + 30, window=int(per))
        try:
            rig = L.Rig(args, **kw)
            ref = rig.roll(rig.theta_true)
        except Exception as e:
            log(f"  {name}: BUILD FAILED {type(e).__name__}: {e}")
            out["configs"][name] = {"spec": kw, "error": f"{type(e).__name__}: {e}"}
            continue
        rows = []
        log(f"\n  {name}   spec {kw}")
        log(_hdr("E"))
        for E in grid:
            r = _row(rig.roll(rig.theta(E=E)), ref)
            r["E"] = float(E)
            rows.append(r)
            log(_line(E, r))
        amp = np.array([r["amp_reading"] for r in rows])
        d = np.diff(amp)
        # the CONTRAST that decides the regime: how far the same sweep moves it on GAIN
        gref = _row(rig.roll(rig.theta(gain=0.5)), ref)
        gr2 = _row(rig.roll(rig.theta(gain=2.0)), ref)
        out["configs"][name] = {
            "spec": kw, "rows": rows,
            "ref_amp": L.amp_reading(ref),
            "amp_span_frac": float(amp.max() / amp.min() - 1.0),
            "amp_exponent": float(np.polyfit(np.log(grid), np.log(amp), 1)[0]),
            "monotone": bool(np.all(d > 0) or np.all(d < 0)),
            "argmax_E": float(grid[int(np.argmax(amp))]),
            "span_steps": {n: _span(rows, n) for n in list(INS) + ["STAT"]},
            "gain_x0.5_STAT": gref["STAT"], "gain_x2_STAT": gr2["STAT"],
            "gain_amp_ratio": float(gr2["amp_reading"] / gref["amp_reading"])}
        c = out["configs"][name]
        log(f"    E over x40: amplitude {100 * c['amp_span_frac']:.1f}% "
            f"(exponent {c['amp_exponent']:+.3f}, monotone {c['monotone']}), "
            f"STAT span {c['span_steps']['STAT']['span']:.1f} steps"
            f"   |   gain 0.5 -> 2.0 moves amplitude x{c['gain_amp_ratio']:.2f}")
        rig.free()
    log(f"\n{'=' * 112}\n  REGIME TABLE -- does E become visible anywhere?\n{'=' * 112}")
    log(f"  {'config':<26s}{'E span (steps)':>16s}{'E amp %':>10s}{'exponent':>10s}"
        f"{'monotone':>10s}{'argmax E':>10s}{'gain x4 amp':>13s}")
    for n, c in out["configs"].items():
        if "error" in c:
            log(f"  {n:<26s}  {c['error']}")
            continue
        log(f"  {n:<26s}{c['span_steps']['STAT']['span']:16.1f}"
            f"{100 * c['amp_span_frac']:10.1f}{c['amp_exponent']:+10.3f}"
            f"{str(c['monotone']):>10s}{c['argmax_E']:10.1f}{c['gain_amp_ratio']:13.2f}")
    dump("regime", out)
    return out


# ============================================================================================= #
def do_long(a):
    """ATTACK 4. Does E become identifiable over several beats rather than one?"""
    beats = 3
    per = 150
    args = L.default_args(device=a.device, warmup=per + 30, window=per * beats)
    rig = L.Rig(args)
    cands = {"uniform_E=planted_mean": rig.theta(E=float(rig.E_true.mean())),
             "uniform_E=40": rig.theta(E=40.0), "uniform_E=220": rig.theta(E=220.0),
             "uniform_E=800": rig.theta(E=800.0),
             "E_true, gain x1.05": rig.theta(gain=(rig.gain_true * 1.05).cpu().numpy())}
    ref = rig.roll(rig.theta_true, G=per * beats)
    out = {"beats": beats, "period": per, "rows": {}}
    log(f"\n  {beats} beats ({per * beats} frames). Each beat is one TICK, so accept() applies.")
    log(f"  {'candidate':<26s}{'1 beat STAT':>13s}{'3 beats STAT':>14s}"
        f"{'accept() 3-tick':>17s}{'limiting':>20s}")
    for name, th in cands.items():
        sim = rig.roll(th, G=per * beats)
        one = L.steps_row(sim[:per], ref[:per])
        three = L.steps_row(sim, ref)
        pairs = [(sim[k * per:(k + 1) * per], ref[k * per:(k + 1) * per]) for k in range(beats)]
        acc = ACC.accept(pairs, L.floors())
        out["rows"][name] = {"one_beat": {k: v for k, v in one.items() if k != "_values"},
                             "three_beats": {k: v for k, v in three.items() if k != "_values"},
                             "accept": {k: v for k, v in acc.items() if k != "per_tick"},
                             "per_beat_STAT": [float(max(
                                 s for s in (L.steps_row(p[0], p[1])[n] for n in INS)
                                 if isinstance(s, float))) for p in pairs]}
        log(f"  {name:<26s}{one['STAT']:13.2f}{three['STAT']:14.2f}"
            f"{acc['statistic']:17.2f}{acc['limiting_instrument']:>20s}")
    log(f"\n  the null, for reference: {min(L.null_row().values()):.2f} steps")
    rig.free()
    dump("long", out)
    return out


# ============================================================================================= #
def do_checker(a):
    """ATTACK 5, THE MAIN ONE. A spatial CONTRAST in E, against its best uniform impostor.

    Uniform E is the least favourable case for identifiability: it moves the global scale, which
    the drive also moves. The question an inverse problem actually asks is whether the SPATIAL
    PATTERN of E is visible. Three things are measured, and the third is the one that decides it:

      1. checkerboard vs uniform at the same arithmetic mean -- confounded, because a mixture's
         effective stiffness is not the mean of its parts, so this can be a scale effect
      2. checkerboard vs ANTI-checkerboard -- identical composition, identical mean, identical
         histogram, only the ARRANGEMENT differs. Whatever this reads is spatial structure and
         nothing else.
      3. checkerboard vs the BEST uniform E, found by scanning. If some uniform sheet imitates the
         checkerboard to within a step, then no per-cell E is recoverable no matter how good the
         estimator is. If the best impostor is still several steps away, E is identifiable.
    """
    rig = L.Rig(L.default_args(device=a.device), quiet=False, log=log)
    out = {"contrasts": {}}
    Ebar = 132.5

    for tag, (lo, hi) in {"planted 45/220 (x4.9)": (45.0, 220.0),
                          "wide 20/800 (x40)": (20.0, 800.0)}.items():
        mid = 0.5 * (lo + hi)
        chk = rig.checker(lo, hi, block=0.10, by="space")
        anti = rig.checker(hi, lo, block=0.10, by="space")
        idx = rig.checker(lo, hi, by="index")
        log(f"\n{'=' * 112}\n  CONTRAST {tag}   arithmetic mean {mid:g};  "
            f"checkerboard {int((chk == hi).sum())} stiff / {int((chk == lo).sum())} soft cells"
            f"\n{'=' * 112}")

        r_chk = rig.roll(rig.theta(E=chk))
        r_anti = rig.roll(rig.theta(E=anti))
        r_idx = rig.roll(rig.theta(E=idx))
        r_uni = rig.roll(rig.theta(E=mid))

        res = {"lo": lo, "hi": hi, "mean": mid,
               "n_stiff": int((chk == hi).sum()), "n_soft": int((chk == lo).sum())}
        pairs = {"checker vs uniform(mean)": (r_chk, r_uni),
                 "checker vs ANTI-checker": (r_chk, r_anti),
                 "checker(space) vs checker(index)": (r_chk, r_idx),
                 "index-checker vs uniform(mean)": (r_idx, r_uni)}
        log(f"  {'pair':<36s}" + "".join(f"{n[:11]:>12s}" for n in INS) + f"{'STAT':>9s}")
        for pn, (s, r) in pairs.items():
            row = L.steps_row(s, r)
            res[pn] = {k: v for k, v in row.items() if k != "_values"}
            log(f"  {pn:<36s}" + "".join(
                (f"{row[n]:12.2f}" if isinstance(row[n], float) else f"{'undef':>12s}")
                for n in INS) + f"{row['STAT']:9.2f}")

        # 3. the best uniform impostor for the checkerboard
        log(f"\n  the best UNIFORM E impostor for the checkerboard "
            f"(if one gets under a step, the pattern is invisible):")
        log(_hdr("uniform E"))
        scan, best = [], None
        for E in np.geomspace(lo * 0.7, hi * 1.4, 11):
            row = _row(rig.roll(rig.theta(E=E)), r_chk)
            row["E"] = float(E)
            scan.append(row)
            log(_line(E, row))
            if best is None or row["STAT"] < best["STAT"]:
                best = row
        res["uniform_impostor_scan"] = [{k: v for k, v in r.items() if k != "_values"}
                                        for r in scan]
        res["best_uniform"] = {k: v for k, v in best.items() if k != "_values"}
        log(f"    BEST uniform impostor: E = {best['E']:.1f} at {best['STAT']:.2f} steps "
            f"(limiting {max((n for n in INS if isinstance(best[n], float)), key=lambda n: best[n])})")
        out["contrasts"][tag] = res

    # ---- the same experiment on GAIN, as the calibration ------------------------------------- #
    log(f"\n{'=' * 112}\n  THE SAME EXPERIMENT ON GAIN -- the parameter P0 says the data DOES "
        f"constrain\n{'=' * 112}")
    g_chk = rig.checker(0.5, 1.5, block=0.10, by="space")
    g_anti = rig.checker(1.5, 0.5, block=0.10, by="space")
    rg_chk = rig.roll(rig.theta(E=Ebar, gain=g_chk))
    rg_anti = rig.roll(rig.theta(E=Ebar, gain=g_anti))
    rg_uni = rig.roll(rig.theta(E=Ebar, gain=1.0))
    gout = {}
    for pn, (s, r) in {"gain checker vs uniform(1.0)": (rg_chk, rg_uni),
                       "gain checker vs ANTI-checker": (rg_chk, rg_anti)}.items():
        row = L.steps_row(s, r)
        gout[pn] = {k: v for k, v in row.items() if k != "_values"}
        log(f"  {pn:<36s}" + "".join(
            (f"{row[n]:12.2f}" if isinstance(row[n], float) else f"{'undef':>12s}")
            for n in INS) + f"{row['STAT']:9.2f}")
    scan = []
    log(f"\n  the best UNIFORM GAIN impostor for the gain checkerboard:")
    log(_hdr("uniform g"))
    bestg = None
    for g in np.linspace(0.5, 1.6, 12):
        row = _row(rig.roll(rig.theta(E=Ebar, gain=float(g))), rg_chk)
        row["gain"] = float(g)
        scan.append(row)
        log(_line(g, row, fmt="{:10.3f}"))
        if bestg is None or row["STAT"] < bestg["STAT"]:
            bestg = row
    gout["uniform_impostor_scan"] = [{k: v for k, v in r.items() if k != "_values"} for r in scan]
    gout["best_uniform"] = {k: v for k, v in bestg.items() if k != "_values"}
    log(f"    BEST uniform-gain impostor: gain = {bestg['gain']:.3f} at {bestg['STAT']:.2f} steps")
    out["gain_control"] = gout

    # ---- and the planted field itself, which is what an estimator is asked for ---------------- #
    log(f"\n{'=' * 112}\n  THE PLANTED FIELD -- flatten it and see what it costs\n{'=' * 112}")
    ref = rig.roll(rig.theta_true)
    pl = {}
    for nm, th in {"uniform E at planted mean": rig.theta(E=float(rig.E_true.mean())),
                   "uniform E at planted median": rig.theta(E=float(rig.E_true.median())),
                   "planted E SHUFFLED across cells":
                       rig.theta(E=np.random.default_rng(7).permutation(
                           rig.E_true.cpu().numpy())),
                   "uniform gain at planted mean (control)":
                       rig.theta(gain=float(rig.gain_true.mean()))}.items():
        row = L.steps_row(rig.roll(th), ref)
        pl[nm] = {k: v for k, v in row.items() if k != "_values"}
        log(f"  {nm:<40s}" + "".join(
            (f"{row[n]:12.2f}" if isinstance(row[n], float) else f"{'undef':>12s}")
            for n in INS) + f"{row['STAT']:9.2f}")
    out["planted"] = pl
    out["null"] = L.null_row()
    log(f"\n  the null (knowing nothing): "
        + ", ".join(f"{n} {v:.2f}" for n, v in out['null'].items()))
    rig.free()
    dump("checker", out)
    return out


# ============================================================================================= #
def do_determinism(a):
    """A step must be a step. Two identical rollouts, and two nearly-identical parameter sets."""
    rig = L.Rig(L.default_args(device=a.device))
    r1 = rig.roll(rig.theta_true)
    r2 = rig.roll(rig.theta_true)
    same = float(np.abs(r1 - r2).max())
    out = {"max_abs_diff_identical_rollouts": same,
           "bitwise_identical": bool(same == 0.0)}
    log(f"  two rollouts of theta_true differ by {same:.3g} (bitwise identical: {same == 0.0})")
    for f in (1.001, 1.01, 1.1):
        row = L.steps_row(rig.roll(rig.theta(E=(rig.E_true * f).cpu().numpy())), r1)
        out[f"E x{f}"] = {k: v for k, v in row.items() if k != "_values"}
        log(f"  E x{f:<6g}: STAT {row['STAT']:8.3f} steps")
        row = L.steps_row(rig.roll(rig.theta(gain=(rig.gain_true * f).cpu().numpy())), r1)
        out[f"gain x{f}"] = {k: v for k, v in row.items() if k != "_values"}
        log(f"  gain x{f:<3g}: STAT {row['STAT']:8.3f} steps")
    rig.free()
    dump("determinism", out)
    return out


DOERS = {"instruments": do_instruments, "dense": do_dense, "regime": do_regime,
         "long": do_long, "checker": do_checker, "determinism": do_determinism}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--do", nargs="+", default=["checker"], choices=list(DOERS) + ["all"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    todo = list(DOERS) if "all" in args.do else args.do
    for k in todo:
        log(f"\n{'#' * 112}\n#  {k.upper()}\n{'#' * 112}")
        DOERS[k](args)
    log("done")
