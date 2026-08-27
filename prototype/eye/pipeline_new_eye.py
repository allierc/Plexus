#!/usr/bin/env python
"""pipeline_new_eye -- one command from a new .blend to a characterised MPM eye.

    python pipeline_new_eye.py --blend "PATH/TO/some_eye.blend" --name H

Everything the earlier eyes needed a human running six or seven commands by hand for
(cut the blend, build the plant, find a safe drive amplitude, run the ~110-hold
characterisation protocol on the L4 partition, assemble the report) is ONE call here,
because the premise of this script is that MORE blend files are coming and nobody should
have to re-type that sequence, or re-discover its ordering, for each one.

WHAT THE EXISTING CODE ALREADY DOES, so this script does not reimplement it:

    archive/run_03/read_blend.py   .blend -> named watertight parts (bpy subprocess)
    blend_mpm_ops.py                parts -> seeded MPM particles (blend_globe/blend_muscles)
    run_eye_G.py                    the plant: build the spec, run it, render it
    probe_groups.py / probe_ops.py  open-loop single-muscle / synergy driving
    characterise_eye.py             the staged protocol (derisk/0lite/0/1/2a/2b or 6d),
                                    sharded one hold per L4 job via run_hold.py
    fig_eyeG_charac.py              the four-panel characterisation summary figure

This script is the ORCHESTRATOR over those, plus the one piece that did not exist yet:
the per-muscle drive-amplitude break scan (stage 2). Everything here is a subprocess or
import call into the files above; nothing physical is reimplemented.

STAGE 1 -- blend to MPM, viz.  Cuts the blend into `archive/eye_<NAME>/blend_parts/`, builds
    the baseline spec, and runs the four-synergy open-loop probe (`--program pairs`) at the
    PLANT'S OWN drive amplitude to produce `pairs_long.mp4` (the VTK surface render) and
    `pairs_long.png`. This run is not thrown away once stage 2 picks a better amplitude: it
    IS stage 0-lite (same label, same file), so it is not repeated in stage 3.

STAGE 2 -- the amplitude scan, the piece this script adds. For each muscle alone, at full
    activation, the peak active stress A is walked up a geometric ladder
    (`--amp-ladder`) until the globe's own diagnostics say the plant has broken --
    `radius_worst_pct` past a few percent, `strain_p99` turning `NaN`, or shortening past
    half the muscle's rest length, the same three indicators eye G's own `x1.5`/`x2` drive
    tests were judged by. The scan STOPS AT THE FIRST BREAK per muscle (no point measuring
    further into a blown-up simulation), plots all six muscles' three indicators against
    amplitude in one figure, and renders one movie of the failure itself for whichever
    muscle broke first (the most informative single failure to look at). The amplitude
    carried into stage 3 is `--amp-margin` (default 0.85) times the SMALLEST last-safe
    amplitude across all six muscles -- the plant is only as strong as its weakest muscle.

STAGE 3 -- the same protocol eye G ran, unchanged: derisk (2 jobs), the synergy gate,
    stage 0 (6 holds, derives T_hold), stage 1 (24 holds, the marginals), stage 2a (15
    holds, the pair screen). If 8 or fewer pairs flag, stage 2b grids them (the protocol's
    own branch); otherwise -- as it was for eye G, all fifteen flagged -- stage 6d replaces
    it with the 64-point Sobol design over all six drives at once. `characterise_eye.py`'s
    own gate REFUSES stage 1 onward if the synergy span fails; this script reports that and
    stops rather than forcing it through.

STAGE 4 -- cleanup. Deletes only what is CHEAPLY REGENERABLE and was never meant to be kept:
    the cut's `blend_parts/meshes/*.ply` exports (regenerate with `read_blend.py --ply`) and
    stray `__pycache__` directories this run created. It does NOT delete any raw run --
    `curves.npz`/`spec.yaml` for every hold -- because that is the exact failure the
    characterisation protocol was written to stop repeating (eyes A-E's raw runs were
    deleted and their fit can no longer be re-derived). It does not touch the shared source
    files this script calls into: `run_eye_G.py`, `characterise_eye.py`, `run_hold.py`,
    `blend_mpm_ops.py` and the rest are load-bearing for every eye, including this one.

OUTPUT, all under `archive/eye_<NAME>/`:

    blend_parts/                 the cut (parts.npz, parts.json, blend_parts.png)
    baseline_spec.yaml           the plant, at the DISCOVERED safe amplitude
    pairs_long.{mp4,png,...}     the viz + the synergy gate (stage 1 = stage 0-lite)
    ampscan/                     amplitude_scan.json, amplitude_scan.png, break_<M>.mp4
    charac/                      holds.npz, report.json, T_hold.json, every raw hold
    charac/fig_characterisation.png   the four-panel summary
    pipeline_report.json         what this script did, in order, with timings
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np

import eye_anatomy as EA
import characterise_eye as CE
import eye_cluster as CL

PY = sys.executable
READ_BLEND = os.path.join(HERE, "archive", "run_03", "read_blend.py")
RUN_EYE_G = os.path.join(HERE, "run_eye_G.py")
FIG_CHARAC = os.path.join(HERE, "fig_eyeG_charac.py")

# the three ways a run says "the plant broke", read off its OWN diag.json -- the same
# indicators eye G's x1.5/x2 drive tests were judged by (6.4% radius loss, NaN strain,
# 74% shortening at a confirmed blow-up; 0.07-0.09% radius loss at the working point)
RADIUS_FAIL_PCT = 3.0
SHORTEN_FAIL_PCT = 55.0
DEFAULT_LADDER = [20.0, 30.0, 45.0, 67.0, 100.0, 150.0, 225.0, 337.0, 500.0]


def _run(cmd, **kw):
    print(f"[pipeline] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def _wait_jobs(name, poll=45, timeout_s=5400):
    """Block until every LSF job named `eye_<name>_*` has drained -- SCOPED to this eye,
    not `eye_cluster.wait()`'s blanket `eye_*` (another session's eye_G jobs are routinely
    running concurrently on this partition; waiting on those too would be wrong)."""
    ssh = os.environ.get("PG_SSH", "allierc@login1")
    t0 = time.time()
    while True:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", ssh,
             f"bjobs -J 'eye_{name}_*' 2>&1 | grep -c -E 'RUN|PEND' || true"],
            capture_output=True, text=True, timeout=60)
        n = int((r.stdout or "0").strip() or 0) if r.returncode == 0 else -1
        print(f"[pipeline] {max(n, 0)} eye_{name}_* job(s) in flight", flush=True)
        if n == 0:
            return
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"eye_{name}_* jobs did not drain within {timeout_s}s")
        time.sleep(poll)


# --------------------------------------------------------------------------- #
#  stage 1 -- blend -> MPM, viz
# --------------------------------------------------------------------------- #
def stage1_cut_and_viz(blend, name, out, device, particles, contract):
    parts_dir = os.path.join(out, "blend_parts")
    print(f"\n[pipeline] === stage 1: cut + baseline spec + viz ({name}) ===", flush=True)
    _run([PY, READ_BLEND, "--blend", blend, "--out", parts_dir, "--figure", "--ply"])

    # the baseline spec, and the synergy probe that IS stage 0-lite -- one run serves both
    # the viz this stage promises and the gate stage 3 will read straight off its diag.
    _run([PY, RUN_EYE_G, "--program", "pairs", "--blend", blend, "--parts", parts_dir,
         "--out", out, "--label", "pairs_long", "--particles", str(particles),
         "--contract", str(contract), "--hold", "200", "--rest", "160",
         "--turns", "0", "--az", "25", "--device", device])
    return parts_dir


# --------------------------------------------------------------------------- #
#  stage 2 -- the amplitude break scan (the new piece)
# --------------------------------------------------------------------------- #
def _diag_verdict(diag):
    """(broken: bool, reason: str) from one pairs-probe diag.json."""
    r = diag.get("radius_worst_pct")
    s = diag.get("strain_p99")
    p = diag.get("peak_shortening_pct")
    if r is not None and r > RADIUS_FAIL_PCT:
        return True, f"radius_worst {r:.2f}% > {RADIUS_FAIL_PCT}%"
    if s is None or not np.isfinite(s):
        return True, "strain_p99 is NaN"
    if p is not None and p > SHORTEN_FAIL_PCT:
        return True, f"peak_shortening {p:.1f}% > {SHORTEN_FAIL_PCT}%"
    return False, "ok"


def stage2_amplitude_scan(blend, parts_dir, name, out, device, particles, ladder,
                          amp_hold, amp_rest, amp_margin):
    print(f"\n[pipeline] === stage 2: per-muscle amplitude scan ({name}) ===", flush=True)
    scan_dir = os.path.join(out, "ampscan")
    os.makedirs(scan_dir, exist_ok=True)
    results = {}                                        # key -> list of (A, diag)
    first_break = None                                  # (key, A) -- the earliest, lowest A

    for key in EA.MUSCLE_KEYS:
        rows = []
        for A in ladder:
            label = f"amp_{key}_{A:g}"
            _run([PY, RUN_EYE_G, "--program", "pairs", "--groups", key,
                 "--blend", blend, "--parts", parts_dir, "--out", scan_dir,
                 "--label", label, "--particles", str(particles),
                 "--contract", str(A), "--hold", str(amp_hold), "--rest", str(amp_rest),
                 "--no-movie", "--device", device])
            diag = json.load(open(os.path.join(scan_dir, f"{label}_diag.json")))
            broken, reason = _diag_verdict(diag)
            rows.append(dict(amplitude=A, broken=broken, reason=reason,
                             radius_worst_pct=diag.get("radius_worst_pct"),
                             strain_p99=diag.get("strain_p99"),
                             peak_shortening_pct=diag.get("peak_shortening_pct")))
            print(f"[pipeline]   {key} A={A:g}: {'BROKEN' if broken else 'ok'} ({reason})",
                 flush=True)
            if broken:
                if first_break is None or A < first_break[1]:
                    first_break = (key, A)
                break                                    # no point scanning further up
        results[key] = rows

    safe = {k: max([r["amplitude"] for r in rows if not r["broken"]], default=0.0)
           for k, rows in results.items()}
    weakest = min(safe, key=safe.get)
    chosen = round(amp_margin * safe[weakest], 1)
    print(f"[pipeline] last-safe amplitude per muscle: "
          + "  ".join(f"{k}={v:g}" for k, v in safe.items()))
    print(f"[pipeline] weakest muscle {weakest} (safe to {safe[weakest]:g}); "
          f"chosen amplitude = {amp_margin:g} x {safe[weakest]:g} = {chosen:g}")

    summary = dict(ladder=ladder, results=results, safe_amplitude_per_muscle=safe,
                   weakest_muscle=weakest, margin=amp_margin, chosen_amplitude=chosen)
    with open(os.path.join(scan_dir, "amplitude_scan.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    _plot_amplitude_scan(results, chosen, os.path.join(scan_dir, "amplitude_scan.png"))

    if first_break is not None:
        bkey, bA = first_break
        _run([PY, RUN_EYE_G, "--program", "pairs", "--groups", bkey,
             "--blend", blend, "--parts", parts_dir, "--out", scan_dir,
             "--label", f"break_{bkey}", "--particles", str(particles),
             "--contract", str(bA), "--hold", str(amp_hold), "--rest", str(amp_rest),
             "--turns", "0", "--az", "25", "--device", device])
        print(f"[pipeline] failure movie: {scan_dir}/break_{bkey}.mp4 (A={bA:g})")

    return chosen, summary


def _plot_amplitude_scan(results, chosen, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), facecolor="white")
    metrics = [("radius_worst_pct", "globe radius error (%)", RADIUS_FAIL_PCT),
              ("strain_p99", "strain p99", None),
              ("peak_shortening_pct", "peak shortening (%)", SHORTEN_FAIL_PCT)]
    colors = {m["key"]: m["color"] for m in EA.MUSCLES}
    for ax, (field, ylabel, thresh) in zip(axes, metrics):
        for key, rows in results.items():
            A = [r["amplitude"] for r in rows]
            y = [r[field] if r[field] is not None and np.isfinite(r[field]) else np.nan
                for r in rows]
            ax.plot(A, y, "o-", color=colors.get(key, "black"), label=key, ms=5)
            broke = next((r["amplitude"] for r in rows if r["broken"]), None)
            if broke is not None:
                ax.axvline(broke, color=colors.get(key, "black"), lw=0.8, alpha=0.35, ls=":")
        if thresh is not None:
            ax.axhline(thresh, color="#d23b3b", lw=1.2, ls="--")
        ax.axvline(chosen, color="#1f6feb", lw=1.6, ls="-", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("peak active stress A")
        ax.set_ylabel(ylabel)
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    axes[-1].text(chosen, axes[-1].get_ylim()[1], f" chosen A={chosen:g}",
                 color="#1f6feb", fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[pipeline] {out_png}")


# --------------------------------------------------------------------------- #
#  stage 3 -- the characterisation protocol, unchanged
# --------------------------------------------------------------------------- #
def stage3_characterise(name, out, chosen_amplitude, particles, side="R"):
    print(f"\n[pipeline] === stage 3: characterisation protocol ({name}) ===", flush=True)
    rel = os.path.relpath(out, HERE)

    ok, sh, sv = CE.gate(rel)
    if ok is None:
        raise RuntimeError("stage 0-lite (pairs_long) diag not found; stage 1 must have "
                           "failed silently")
    print(f"[pipeline] gate: {sh:.1f} x {sv:.1f} deg -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("[pipeline] GATE FAILED. Per the protocol: change the eye, not the "
             "measurement. Stopping stage 3 here -- stage 4 cleanup still runs.")
        return False

    for stage in ("derisk", "0"):
        jobs = CE.STAGES[stage](rel, name)
        CL.submit(jobs)
        _wait_jobs(name)
    CE.collect(rel)

    for stage in ("1", "2a"):
        jobs = CE.STAGES[stage](rel, name)
        CL.submit(jobs)
        _wait_jobs(name)
    CE.collect(rel)

    flagged = [r for r in CE.flagged_pairs(rel) if r["flagged"]]
    print(f"[pipeline] stage 2a: {len(flagged)}/15 pairs flagged")
    if 0 < len(flagged) <= 8:
        jobs = CE.jobs_stage2b(rel, name)
        CL.submit(jobs)
        _wait_jobs(name)
    else:
        if len(flagged) > 8:
            print("[pipeline] >8 pairs flagged (as for eye G): the plant is not close to "
                 "additive, running the 6D Sobol design instead of the pairwise grid.")
        jobs = CE.jobs_stage6d(rel, name)
        CL.submit(jobs)
        _wait_jobs(name)
    CE.collect(rel)

    fig_out = os.path.join(out, "charac", "fig_characterisation.png")
    _run([PY, FIG_CHARAC, "--eye", out, "--out", fig_out])
    return True


# --------------------------------------------------------------------------- #
#  stage 4 -- cleanup (only the cheaply regenerable)
# --------------------------------------------------------------------------- #
def stage4_cleanup(out):
    print(f"\n[pipeline] === stage 4: cleanup ({out}) ===", flush=True)
    removed = []
    meshes = os.path.join(out, "blend_parts", "meshes")
    if os.path.isdir(meshes):
        shutil.rmtree(meshes)
        removed.append(meshes)
    for root, dirs, _files in os.walk(out):
        if "__pycache__" in dirs:
            p = os.path.join(root, "__pycache__")
            shutil.rmtree(p)
            removed.append(p)
    for p in removed:
        print(f"[pipeline] removed {p}")
    if not removed:
        print("[pipeline] nothing to remove")
    return removed


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blend", required=True, help="path to the .blend")
    ap.add_argument("--name", required=True, help="eye name, e.g. H -> archive/eye_H")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--particles", type=int, default=45000)
    ap.add_argument("--side", default="R", choices=("L", "R"))
    ap.add_argument("--amp-ladder", default=",".join(str(v) for v in DEFAULT_LADDER))
    ap.add_argument("--amp-hold", type=int, default=150)
    ap.add_argument("--amp-rest", type=int, default=30)
    ap.add_argument("--amp-margin", type=float, default=0.85)
    ap.add_argument("--skip-scan", action="store_true",
                    help="skip stage 2, use --contract as the amplitude directly")
    ap.add_argument("--contract", type=float, default=67.0,
                    help="starting amplitude for stage 1's viz run, and the amplitude used "
                         "outright if --skip-scan")
    ap.add_argument("--skip-charac", action="store_true", help="stop after stage 2")
    args = ap.parse_args()

    blend = os.path.abspath(args.blend)
    if not os.path.exists(blend):
        raise SystemExit(f"no such file: {blend}")
    out = os.path.join(HERE, "archive", f"eye_{args.name}")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    report = dict(blend=blend, name=args.name, out=out, started=t0)

    parts_dir = stage1_cut_and_viz(blend, args.name, out, args.device, args.particles,
                                   args.contract)
    report["stage1_seconds"] = round(time.time() - t0, 1)

    if args.skip_scan:
        chosen = args.contract
        print(f"[pipeline] --skip-scan: using amplitude {chosen:g} directly")
    else:
        t1 = time.time()
        ladder = [float(v) for v in args.amp_ladder.split(",")]
        chosen, scan_summary = stage2_amplitude_scan(
            blend, parts_dir, args.name, out, args.device, args.particles, ladder,
            args.amp_hold, args.amp_rest, args.amp_margin)
        report["stage2_seconds"] = round(time.time() - t1, 1)
        report["amplitude_scan"] = scan_summary["safe_amplitude_per_muscle"]
        report["chosen_amplitude"] = chosen

    if chosen != args.contract:
        # rebuild the baseline spec (and the gate's own pairs_long) at the DISCOVERED
        # amplitude -- run_hold.py reads amplitude from THIS file for every later hold,
        # so stage 3 never has to know the amplitude was chosen rather than assumed.
        print(f"\n[pipeline] rebuilding the plant at the chosen amplitude {chosen:g}")
        stage1_cut_and_viz(blend, args.name, out, args.device, args.particles, chosen)

    if not args.skip_charac:
        t2 = time.time()
        passed = stage3_characterise(args.name, out, chosen, args.particles, args.side)
        report["stage3_seconds"] = round(time.time() - t2, 1)
        report["gate_passed"] = passed

    report["removed"] = stage4_cleanup(out)
    report["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "pipeline_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[pipeline] done in {report['total_seconds']:.0f}s -> {out}")
    print(f"[pipeline] {os.path.join(out, 'pipeline_report.json')}")


if __name__ == "__main__":
    main()
