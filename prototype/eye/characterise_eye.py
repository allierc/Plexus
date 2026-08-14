"""characterise_eye -- run the whole characterisation protocol for one eye on the L4 partition.

    python characterise_eye.py archive/eye_F --stage 0        # span gate first, always
    python characterise_eye.py archive/eye_F --stage 1
    python characterise_eye.py archive/eye_F --collect        # assemble the protocol's files
    python characterise_eye.py archive/eye_F --status

Takes a MODEL FOLDER, not a model letter, so a new eye is characterised by pointing at
its archive directory -- the folder must contain a `baseline_spec.yaml` (or any
`*_spec.yaml`, which is what the older eyes kept).

The protocol is PROTOCOL_eye_characterisation.md; this is its executable form. Three
things in it are load-bearing and are enforced here rather than left to the operator:

  THE SPAN GATE IS A GATE. Stage 0 measures what the eye can reach; if horizontal span
  is under 25 deg or vertical under 10, `--stage 1` REFUSES to submit and says so.
  Characterising an eye that cannot do the task produces a perfect description of a
  useless plant, and eye F was heading for exactly that at 7.9 deg.

  THE HOLD IS DERIVED, NOT ASSUMED. `T_hold = max(2.0 s, 1.5 x slowest settling)` comes
  out of stage 0 and is written into `characterise_<eye>/T_hold.json`, which every later
  stage reads. Eyes A-E were fitted from holds of 0.19-1.27 s against a settling time of
  1.28 s -- none had stopped moving -- and a hard-coded 2.0 s would quietly repeat that
  on any eye softer than F.

  THE RAW RUNS ARE KEPT. Every job writes its own curves.npz and spec.yaml under the
  eye's folder. A-E's raw runs were deleted and their fit can no longer be re-derived.

Sharding: every hold is an independent run, so the stages shard by run across the
partition, one bsub per job, `PG_PARALLEL` at a time. Stage 1 is 30 jobs; on 8 L4s
that is four waves.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np

import eye_anatomy as EA
import eye_cluster as CL

LEVELS = [0.10, 0.25, 0.50, 0.75, 1.00]      # not four even ones: the low end carries the shape
SCREEN_LEVEL = 0.5                            # stage 2a, every pair at one point
GRID_LEVELS = [0.25, 0.50, 0.75]              # stage 2b, flagged pairs only
PAIR_TOL = 0.20                               # deg: four times the settled tolerance
GATE_H, GATE_V = 25.0, 10.0                   # deg, from the task
SETTLED_TOL = 0.05                            # deg peak-to-peak


def eye_name(folder):
    return os.path.basename(os.path.normpath(folder)).replace("eye_", "")


def outdir(folder):
    d = os.path.join(HERE, "archive", f"characterise_{eye_name(folder)}")
    os.makedirs(d, exist_ok=True)
    return d


def _spec_of(folder):
    p = os.path.join(folder, "baseline_spec.yaml")
    if os.path.exists(p):
        return p
    c = sorted(glob.glob(os.path.join(folder, "*_spec.yaml")))
    if not c:
        raise FileNotFoundError(f"{folder} holds no spec.yaml -- not an eye archive")
    return c[0]


def t_hold(folder, default=2.0):
    """The hold length stage 0 derived for THIS eye, or the floor if stage 0 has not run."""
    p = os.path.join(outdir(folder), "T_hold.json")
    if os.path.exists(p):
        return float(json.load(open(p))["T_hold"])
    return default


def gate(folder, quiet=False):
    """(passes, span_h, span_v) from stage 0's results, or (None, ...) if it has not run."""
    p = os.path.join(outdir(folder), "stage0.json")
    if not os.path.exists(p):
        return None, None, None
    rows = json.load(open(p))
    h = [r["pose_deg"][0] for r in rows]
    v = [r["pose_deg"][1] for r in rows]
    span_h, span_v = max(h) - min(h), max(v) - min(v)
    ok = span_h >= GATE_H and span_v >= GATE_V
    if not quiet:
        print(f"[gate] horizontal {span_h:.1f} deg (need {GATE_H}), "
              f"vertical {span_v:.1f} deg (need {GATE_V}) -> {'PASS' if ok else 'FAIL'}")
    return ok, span_h, span_v


# --------------------------------------------------------------------------- the stages
def jobs_stage0(folder, model):
    """6 runs: every muscle at full drive, held 3 s, from rest. Gate + settling time."""
    return [(f"{model}_s0_{m}",
             f"python run_hold.py --folder {folder} --muscles {m} --level 1.0 "
             f"--hold-s 3.0 --stage 0 --device cuda:0")
            for m in EA.MUSCLE_KEYS]


def jobs_stage1(folder, model):
    """30 holds: each muscle alone at five levels, from rest each time."""
    T = t_hold(folder)
    return [(f"{model}_s1_{m}_{u:g}".replace("0.", "p"),
             f"python run_hold.py --folder {folder} --muscles {m} --level {u} "
             f"--hold-s {T} --stage 1 --device cuda:0")
            for m in EA.MUSCLE_KEYS for u in LEVELS]


def jobs_stage2a(folder, model):
    """15 holds: every unordered pair together at 0.5, to decide which pairs interact."""
    T = t_hold(folder)
    return [(f"{model}_s2a_{i}_{j}",
             f"python run_hold.py --folder {folder} --muscles {i} {j} "
             f"--level {SCREEN_LEVEL} {SCREEN_LEVEL} --hold-s {T} --stage 2a --device cuda:0")
            for i, j in itertools.combinations(EA.MUSCLE_KEYS, 2)]


def flagged_pairs(folder):
    """Pairs whose measured pose departs from the additive prediction by > PAIR_TOL."""
    o = outdir(folder)
    s1 = {(r["muscles"][0], r["level"][0]): r for r in _load(o, "stage1.json")}
    out = []
    for r in _load(o, "stage2a.json"):
        a, b = r["muscles"]
        pa, pb = s1.get((a, SCREEN_LEVEL)), s1.get((b, SCREEN_LEVEL))
        if not (pa and pb):
            continue
        pred = np.array(pa["pose_deg"]) + np.array(pb["pose_deg"])
        resid = np.abs(np.array(r["pose_deg"]) - pred)
        out.append(dict(pair=[a, b], residual_deg=[round(float(v), 4) for v in resid],
                        flagged=bool(resid.max() > PAIR_TOL)))
    return out


def jobs_stage2b(folder, model):
    """9 holds per flagged pair: the 3x3 interior. Unflagged pairs get nothing."""
    T = t_hold(folder)
    rows = [r for r in flagged_pairs(folder) if r["flagged"]]
    if len(rows) > 8:
        print(f"[stage2b] {len(rows)} pairs flagged (>8). The plant is not close to additive; "
              f"per the protocol, stop and re-plan the sampling rather than expanding it.")
        return []
    jobs = []
    for r in rows:
        a, b = r["pair"]
        for ua in GRID_LEVELS:
            for ub in GRID_LEVELS:
                jobs.append((f"{model}_s2b_{a}{b}_{ua:g}_{ub:g}".replace("0.", "p"),
                             f"python run_hold.py --folder {folder} --muscles {a} {b} "
                             f"--level {ua} {ub} --hold-s {T} --stage 2b --device cuda:0"))
    return jobs


def _load(o, name):
    p = os.path.join(o, name)
    return json.load(open(p)) if os.path.exists(p) else []


STAGES = {"0": jobs_stage0, "1": jobs_stage1, "2a": jobs_stage2a, "2b": jobs_stage2b}


def collect(folder):
    """Assemble the protocol's output files from whatever runs have landed."""
    o = outdir(folder)
    rows = []
    for name in ("stage0.json", "stage1.json", "stage2a.json", "stage2b.json"):
        rows += _load(o, name)
    if not rows:
        print(f"[collect] nothing in {o} yet")
        return
    m = np.zeros((len(rows), EA.N_MUSCLE), np.float32)
    for k, r in enumerate(rows):
        for name, u in zip(r["muscles"], r["level"]):
            m[k, EA.MUSCLE_KEYS.index(name)] = u
    np.savez_compressed(
        os.path.join(o, "holds.npz"),
        muscles=np.array(EA.MUSCLE_KEYS),
        m=m,
        pose=np.array([r["pose_deg"] for r in rows], np.float32),
        p2p=np.array([r["settle_ptp_deg"] for r in rows], np.float32),
        settled=np.array([r["settled"] for r in rows], bool),
        stage=np.array([r["stage"] for r in rows]),
        T_hold=np.float32(t_hold(folder)))
    ok, sh, sv = gate(folder, quiet=True)
    report = dict(eye=eye_name(folder), n_holds=len(rows),
                  span_h=sh, span_v=sv, gate_pass=ok,
                  gate_required=[GATE_H, GATE_V], T_hold=t_hold(folder),
                  settled_tolerance_deg=SETTLED_TOL,
                  n_unsettled=int(sum(not r["settled"] for r in rows)),
                  settling_time_s={r["muscles"][0]: r.get("settling_s")
                                   for r in _load(o, "stage0.json")},
                  pair_residuals=flagged_pairs(folder))
    with open(os.path.join(o, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[collect] {len(rows)} holds, {report['n_unsettled']} unsettled -> {o}")
    print(f"[collect] span {sh} / {sv} deg, gate {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="the eye's archive folder, e.g. archive/eye_F")
    ap.add_argument("--stage", choices=list(STAGES), default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true", help="submit even if the span gate failed")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    folder = a.folder if os.path.isabs(a.folder) else os.path.join(HERE, a.folder)
    _spec_of(folder)                                    # fail early if it is not an eye archive
    model = eye_name(folder)
    if a.status:
        return CL.status()
    if a.collect:
        return collect(folder)
    if a.stage is None:
        ap.error("give --stage {0,1,2a,2b} or --collect")

    if a.stage != "0":
        ok, sh, sv = gate(folder)
        if ok is None:
            print("[gate] stage 0 has not been run for this eye. Run --stage 0 first: the hold "
                  "length for every later stage is derived from its settling time.")
            return
        if not ok and not a.force:
            print(f"[gate] REFUSING to submit stage {a.stage}. The task needs {GATE_H} deg "
                  f"horizontal and {GATE_V} vertical; this eye reaches {sh:.1f} and {sv:.1f}. "
                  f"Change the eye, not the measurement. (--force overrides.)")
            return
    jobs = STAGES[a.stage](folder, model)
    if not jobs:
        print(f"[stage {a.stage}] nothing to submit")
        return
    print(f"[stage {a.stage}] {len(jobs)} jobs, T_hold = {t_hold(folder):.2f} s")
    if a.dry_run:
        for n, c in jobs:
            print("   ", n, "|", c)
        return
    CL.submit(jobs)


if __name__ == "__main__":
    main()
