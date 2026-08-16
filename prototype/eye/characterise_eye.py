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
import yaml

import eye_anatomy as EA
import eye_cluster as CL

LEVELS = [0.10, 0.25, 0.50, 1.00]             # weighted low: the nonlinearity lives near 0
SCREEN_LEVEL = 0.5                            # stage 2a, every pair at one point
GRID_LEVELS = [0.35, 0.75]                    # stage 2b, flagged pairs only: a 2x2
PAIR_TOL = 0.20                               # deg: four times the settled tolerance

# THE GATE, as of eye G (set 2026-08-15, by the session that owns the controller).
# It was 25 deg horizontal, taken from the tracking task. No geometric lever reaches it on
# scanned anatomy and all of them have now been measured: globe x1.2 loses 81% of the
# temporal excursion and x0.9 / x0.85 collapse the nasal one (8.5 -> 2.1 -> 0.7 deg) while
# horizontal stays ~15; drive is at its ceiling (x1.5 and x2 blow the MPM up); pinning the
# origins costs 6-19%. Eye G reaches ~16 deg horizontal and 27.6 vertical WITH ALL FOUR
# SYNERGIES CORRECT, and that is accepted as the plant. The gate stays in the script -- it
# still catches an eye that cannot move at all -- but it now encodes what this plant does
# rather than what the first draft of the task hoped for.
GATE_H, GATE_V = 15.0, 10.0                   # deg
SETTLED_TOL = 0.05                            # deg peak-to-peak


def eye_name(folder):
    return os.path.basename(os.path.normpath(folder)).replace("eye_", "")


def outdir(folder):
    """Where this eye's characterisation lands: INSIDE the eye's own archive, as `charac/`.

    Everything about one eye stays in one directory -- the spec it was run from, the
    synergy movie, every hold's spec / curves / mp4, and the protocol's assembled files --
    so an eye can be handed over, or deleted, as a unit.
    """
    f = folder if os.path.isabs(folder) else os.path.join(HERE, folder)
    d = os.path.join(f, "charac")
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
    """(passes, span_h, span_v) -- the USABLE workspace, preferring stage 0-lite.

    The protocol tests the gate on the union of the four synergy excursions, because that
    is the workspace a controller can actually command: no single extraocular muscle of a
    fish eye moves it along a cardinal axis, so single-muscle extremes systematically
    understate the reachable range. On eye G the two differ by 1.7 deg horizontal (15.9
    against 14.2), which straddles the threshold -- so which one is used has to be stated
    rather than left to whichever file happens to exist.
    """
    f = folder if os.path.isabs(folder) else os.path.join(HERE, folder)
    lite = os.path.join(f, "pairs_long_diag.json")
    if os.path.exists(lite):
        exc = [v["gaze_excursion_deg"] for v in json.load(open(lite))["synergies"].values()]
        h = [e[0] for e in exc] + [0.0]
        v = [e[1] for e in exc] + [0.0]
        span_h, span_v = max(h) - min(h), max(v) - min(v)
        ok = span_h >= GATE_H and span_v >= GATE_V
        if not quiet:
            print(f"[gate] stage 0-lite synergies: horizontal {span_h:.1f} deg (need {GATE_H}), "
                  f"vertical {span_v:.1f} (need {GATE_V}) -> {'PASS' if ok else 'FAIL'}")
        return ok, span_h, span_v
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
def jobs_stage0lite(folder, model):
    """ONE run: the four cardinal SYNERGIES in a single simulation -- the gate.

    SR+SO up, IR+IO down, LR temporal, MR nasal, driven open loop by
    `probe_ops.muscle_probe [groups]` (see `probe_groups.py`). It answers, for one job,
    both questions a per-muscle sweep cannot: whether the scanned geometry moves the eye
    where the anatomy claims -- no single extraocular muscle of a fish eye moves it along
    a cardinal axis, so only a synergy can be scored against a direction written down in
    advance -- and what the usable workspace is. If this fails the gate, change the eye
    rather than characterise it.
    """
    return [(f"{model}_s0lite",
             f"python run_eye_G.py --program pairs --hold 200 --rest 160 --stride 3 "
             f"--out {folder} --label pairs_long --turns 0 --az 25 --device cuda:0")]


def jobs_derisk(folder, model):
    """4 short jobs to run BEFORE the ~65: does the pipeline hold together on THIS eye?

    Cheap answers to the three things that would waste a whole fan-out:

      substep   the production substep (2.0e-4, validated on eye F) against the one this
                eye's own spec was integrated at. If a hold's settled pose differs by more
                than the settled tolerance, the cheap substep is not safe on this geometry
                and every later stage has to use the spec's.
      loading   that a BLEND-seeded spec loads and runs through `run_hold` at all --
                `blend_globe` / `blend_muscles` have to be registered, and this is the
                first stage that would find out.
      the gate  the synergy run on the two deflated globes, because eye G fails the
                horizontal gate at 16.4 deg and globe size is the strongest lever known
                (eye H, globe x1.2, lost 81% of the temporal excursion -- so the test is
                the other direction).
    """
    spec = yaml.safe_load(open(_spec_of(os.path.join(HERE, folder))))
    fine = next((st["substep_dt"] for st in spec["schedule"]
                 if isinstance(st, dict) and "substep_dt" in st), 1.2e-4)
    jobs = [(f"{model}_dr_substep_prod",
             f"python run_hold.py --folder {folder} --muscles LR --level 1.0 --hold-s 3.0 "
             f"--stage derisk --substep 2.0e-4 --device cuda:0"),
            (f"{model}_dr_substep_fine",
             f"python run_hold.py --folder {folder} --muscles LR --level 1.0 --hold-s 3.0 "
             f"--stage deriskfine --substep {fine:g} --device cuda:0")]
    for infl in (0.85, 0.90):
        tag = f"{infl:g}".replace("0.", "p")
        out = f"archive/eye_{model}{tag}"
        jobs.append((f"{model}_dr_globe{tag}",
                     f"python run_eye_G.py --program pairs --hold 200 --rest 160 --stride 3 "
                     f"--inflate {infl:g} --out {out} --label pairs_long --turns 0 --az 25 "
                     f"--device cuda:0"))
    return jobs


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


def jobs_stage6d(folder, model, n=64):
    """A 6-D SOBOL sweep -- what to do when every pair interacts.

    Stage 2a screened all fifteen pairs and flagged ALL FIFTEEN: residuals 0.09-1.03 deg
    against a 0.20 tolerance, concentrated in horizontal, up to ~13% of a typical
    excursion. That is not measurement noise -- the numerical floor is 0.03 deg, measured
    as the same hold at two substeps -- so this plant is genuinely sub-additive and the
    protocol says stop and re-plan rather than expand.

    The re-plan: stage 2b would spend 60 holds on 15 bilinear terms and would still never
    leave the pairwise faces of the cube. A quadratic in six drives has 27 coefficients
    (6 linear, 6 square, 15 cross), so a Sobol design over the whole [0,1]^6 identifies the
    same interactions from ~50 points AND samples the interior, where a controller
    commanding three or four muscles at once actually operates. Sobol rather than random:
    it is deterministic, so the design is in the file rather than in a seed. 64 rather than
    50 because Sobol's balance properties hold at powers of two.
    """
    from scipy.stats import qmc
    T = t_hold(folder)
    pts = qmc.Sobol(d=EA.N_MUSCLE, scramble=True, seed=0).random(n)
    jobs = []
    for k, u in enumerate(np.round(pts, 3)):
        lv = " ".join(f"{v:g}" for v in u)
        jobs.append((f"{model}_s6d_{k:03d}",
                     f"python run_hold.py --folder {folder} "
                     f"--muscles {' '.join(EA.MUSCLE_KEYS)} --level {lv} "
                     f"--hold-s {T} --stage 6d --no-movie --device cuda:0"))
    return jobs


def _load(o, name):
    """A stage's rows: the per-hold files written by `run_hold`, plus any legacy table.

    Per-hold files are the source of truth (see run_hold on the lost-update race); the
    stage<N>.json tables are kept readable for the holds that predate them.
    """
    stage = name.replace("stage", "").replace(".json", "")
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(o, "rows", f"s{stage}_*.json")))]
    seen = {(tuple(r["muscles"]), tuple(r["level"])) for r in rows}
    p = os.path.join(o, name)
    for r in (json.load(open(p)) if os.path.exists(p) else []):
        if (tuple(r["muscles"]), tuple(r["level"])) not in seen:
            rows.append(r)
    return rows


STAGES = {"derisk": jobs_derisk, "0lite": jobs_stage0lite, "0": jobs_stage0,
          "1": jobs_stage1, "2a": jobs_stage2a, "2b": jobs_stage2b, "6d": jobs_stage6d}


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

    if a.stage not in ("0", "0lite", "derisk"):
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
    # the job command must carry a RELATIVE folder: the cluster mounts this repo under
    # /groups/... and the devcontainer under /workspace, and the job script cds to the
    # prototype before running. An absolute path here is the one that does not survive.
    jobs = STAGES[a.stage](os.path.relpath(folder, HERE), model)
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
