"""run_hold -- ONE hold: drive a set of muscles at fixed levels, wait until it stops, record.

    python run_hold.py --folder archive/eye_F --muscles LR --level 1.0 --hold-s 2.0 --stage 1
    python run_hold.py --folder archive/eye_F --muscles LR MR --level 0.5 0.5 --stage 2a

The unit of work for `characterise_eye.py`. One hold is one cluster job, which is what
makes the protocol shard: 30 stage-1 holds are 30 independent runs with nothing to
share but the spec they start from.

A hold is not a step. The drive ramps over 0.2 s (an instantaneous jump excites the MPM
substep far harder than anything the controller does), holds, and the pose is averaged
over the LAST QUARTER of the hold -- with the peak-to-peak recorded beside it, because
a mean over a window that is still ringing is not a steady state, and eyes A-E were
fitted from exactly such means. `settled` is that test, not a promise.

Stage 0 additionally measures the SETTLING TIME: the first moment the trace stays inside
0.05 deg of its final value. That number sets `T_hold` for every later stage of this eye.
"""
from __future__ import annotations
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators, eye_ops, muscle_ops, probe_ops   # noqa: F401
# BLEND-SEEDED EYES (G and later) name `blend_globe` / `blend_muscles` in their spec, so the
# registry has to know them before `load_spec` -- without this import a hold on eye_G fails
# at schema validation, which is the one thing that made this script F-only.
import blend_mpm_ops as BM                               # noqa: F401
import render_surface_vtk                                # noqa: F401  the per-hold movie
import eye_anatomy as EA, run_eye
from plexus.schema import load as load_spec
from plexus.models.registry import register_operator
from characterise_eye import outdir, eye_name, _spec_of, SETTLED_TOL

SUBSTEP = 2.0e-4        # validated by derisk_tests: 0.007 deg from the reference, 2.6x cheaper


def settling_time(t, g, tol=SETTLED_TOL):
    """First time after which every angle stays within `tol` of its final value."""
    final = g[-1]
    inside = (np.abs(g - final) <= tol).all(1)
    bad = np.nonzero(~inside)[0]
    return float(t[bad[-1] + 1] - t[0]) if bad.size and bad[-1] + 1 < len(t) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--muscles", nargs="+", required=True)
    ap.add_argument("--level", type=float, nargs="+", required=True)
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--lead-s", type=float, default=0.3)
    ap.add_argument("--ramp-s", type=float, default=0.2)
    ap.add_argument("--tail-s", type=float, default=0.2)
    ap.add_argument("--stage", default="1")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--substep", type=float, default=SUBSTEP,
                    help="MLS-MPM substep; the derisk pair compares this against the spec's")
    ap.add_argument("--movie", dest="movie", action="store_true", default=True,
                    help="write a VTK surface mp4 of the hold (default)")
    ap.add_argument("--no-movie", dest="movie", action="store_false")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if len(a.level) != len(a.muscles):
        ap.error("one --level per muscle")

    folder = a.folder if os.path.isabs(a.folder) else os.path.join(HERE, a.folder)
    spec = yaml.safe_load(open(_spec_of(folder)))
    dt = float(spec["general"]["dt"])
    hold, lead, tail = (int(round(s / dt)) for s in (a.hold_s, a.lead_s, a.tail_s))
    tonic = float(next((o for o in spec["operators"]
                        if o["op"] in ("oculomotor_drive", "muscle_probe")), {}).get("tonic", 0.14))
    idx = [EA.MUSCLE_KEYS.index(m) for m in a.muscles]

    # a multi-muscle hold: the staircase probe with a single level, one per driven muscle.
    # `muscle_probe [tour]` already drives all six from a vector, so the hold is that with a
    # constant vector rather than a schedule.
    s = probe_ops.staircase_spec(spec, idx[0], levels=(a.level[0],), hold=hold, lead=lead,
                                 tail=tail, tonic=tonic)
    for o in s["operators"]:
        if o["op"] == "muscle_probe":
            o["implementation"] = "hold_vector"
            o["muscles"] = idx
            o["levels_vec"] = [float(v) for v in a.level]
            o["step_frames"] = int(round(a.ramp_s / dt))
    for step in s["schedule"]:
        if isinstance(step, dict) and "substep_dt" in step:
            step["substep_dt"] = a.substep

    o = outdir(folder)
    tag = "_".join(f"{m}{u:g}" for m, u in zip(a.muscles, a.level))
    runs = os.path.join(o, "runs"); os.makedirs(runs, exist_ok=True)
    path = os.path.join(runs, f"s{a.stage}_{tag}_spec.yaml")
    with open(path, "w") as fh:
        fh.write(f"# stage {a.stage} hold: {tag}, hold {a.hold_s}s, ramp {a.ramp_s}s\n")
        yaml.safe_dump(s, fh, sort_keys=False, width=100)

    sim = load_spec(path)
    t0 = time.time()
    _, cap = run_eye.capture_run(sim, a.device, stride=a.stride)
    t = np.asarray(cap["frame"]) * dt
    g = np.asarray(cap["gaze"])
    # KEEP THE RAW RUN. A-E's were deleted and their fit can no longer be re-derived.
    np.savez_compressed(os.path.join(runs, f"s{a.stage}_{tag}_curves.npz"),
                        frame=np.asarray(cap["frame"]), t=t.astype(np.float32),
                        gaze=g.astype(np.float32), act=np.asarray(cap["act"], np.float32),
                        length=np.asarray(cap["length"], np.float32),
                        rest_length=np.asarray(cap["rest_length"], np.float32),
                        centre=np.asarray(cap["centre"], np.float32),
                        muscles=np.array(EA.MUSCLE_KEYS),
                        level=np.array(a.level, np.float32),
                        driven=np.array(a.muscles))
    if a.movie:
        # one movie per hold, drawn through the .blend geometry. The eye is scanned anatomy
        # now, so "is this hold sane" is a question you answer by LOOKING at the strap that
        # was driven -- and a blown-up hold is obvious in a frame and invisible in a mean.
        infl = float(next((o for o in s["operators"] if o["op"] == "blend_globe"),
                          {}).get("inflate", 1.0))
        side = str(next((o for o in s["operators"] if o["op"] == "blend_globe"),
                        {}).get("side", "R"))
        try:
            render_surface_vtk.render(cap, dt, os.path.join(runs, f"s{a.stage}_{tag}.mp4"),
                                      os.path.join(runs, f"s{a.stage}_{tag}.png"),
                                      turns=0.0, az0=25.0, side=side, inflate=infl)
        except Exception as exc:                      # a movie must never lose a measurement
            print(f"[hold {tag}] movie skipped: {exc}", flush=True)

    win = (t >= (lead + hold) * dt - 0.25 * a.hold_s) & (t <= (lead + hold) * dt)
    base = g[t <= lead * dt]
    base = base[-1] if len(base) else g[0]
    mean, ptp = g[win].mean(0), (g[win].max(0) - g[win].min(0))
    row = dict(stage=a.stage, muscles=a.muscles, level=[float(v) for v in a.level],
               pose_deg=[round(float(v), 4) for v in (mean - base)],
               settle_ptp_deg=[round(float(v), 4) for v in ptp],
               settled=bool(ptp.max() <= SETTLED_TOL),
               hold_s=a.hold_s, seconds=round(time.time() - t0, 1))
    if a.stage == "0":
        # WITHIN THE HOLD, not to the end of the run: the tail releases the drive and the eye
        # returns to rest, so a window that includes it takes the RELEASED pose as `final`
        # and reports the whole hold as "still settling" -- which is what it did, giving all
        # six muscles an identical 3.195 s. T_hold is derived from this number, so the error
        # propagates into every later stage as holds twice as long as they need to be.
        hw = (t >= lead * dt) & (t <= (lead + hold) * dt)
        row["settling_s"] = round(settling_time(t[hw], g[hw]), 3)
    # ONE FILE PER HOLD, not one per stage. Read-modify-write on a shared table is a
    # lost-update race the moment the stage is wider than the partition: 64 Sobol jobs ran
    # concurrently and 63 rows survived. Each job now owns its own row file and `collect`
    # assembles them, so the number of rows equals the number of jobs by construction.
    rows_dir = os.path.join(o, "rows"); os.makedirs(rows_dir, exist_ok=True)
    with open(os.path.join(rows_dir, f"s{a.stage}_{tag}.json"), "w") as fh:
        json.dump(row, fh, indent=2)
    print(f"[hold {tag}] pose {row['pose_deg']} settled={row['settled']} "
          f"p-p {row['settle_ptp_deg']} [{row['seconds']}s]", flush=True)
    if a.stage == "0":
        print(f"[hold {tag}] settling {row['settling_s']} s", flush=True)


if __name__ == "__main__":
    main()
