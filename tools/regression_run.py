"""Run the regression series and append the result to the archive. The overnight entry point.

    python tools/regression_run.py --device cuda:0                 the whole series (~20 min on an A6000)
    python tools/regression_run.py --device cuda:1 --quick         three short working points + the rest
    python tools/regression_run.py --only cvd2_adder_tension       one working point
    python tools/regression_run.py ... --open                      open the dashboard when done

    nohup python tools/regression_run.py --device cuda:0 > log/regression/nightly.log 2>&1 &

Every run appends ONE JSON line to `log/regression/archive.jsonl`: date, commit, host, GPU, and for
each working point its status, the violations, the frame time and every checkpoint metric of both
the reference and the rerun -- so two runs can be compared metric by metric later without rerunning
anything. `tools/regression_dashboard.py` reads that file.

What is run, in order:
  1. the seed table  (tools/seed_conventions.py --check; CPU, ~2 min)
  2. the invariants and scaling tests  (tests/regression/test_invariants.py, test_scaling.py)
  3. every registered working point  (tests/regression/fingerprints/*.json), via regression_lib
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import regression_lib as R  # noqa: E402

ARCHIVE = os.path.join(ROOT, "log", "regression", "archive.jsonl")
QUICK = {"cvd2_adder_tension": 60, "cv_kv_double": 60, "apop2_ks0p1": 60}


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


def _pytest_file(path: str, device: str) -> dict:
    """Run one test file under the regression marker; return {passed, failed, cases: {name: status/msg}}."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        xml = f.name
    env = dict(os.environ, PLEXUS_REGRESSION_DEVICE=device)
    subprocess.run([sys.executable, "-m", "pytest", path, "-m", "regression", "-q", "-p", "no:cacheprovider",
                    f"--junitxml={xml}"], cwd=ROOT, env=env, capture_output=True, text=True)
    out = {"passed": 0, "failed": 0, "skipped": 0, "cases": {}}
    try:
        for tc in ET.parse(xml).getroot().iter("testcase"):
            name = tc.get("name")
            fail = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
            skip = tc.find("skipped")
            if fail is not None:
                out["failed"] += 1
                out["cases"][name] = "FAIL: " + (fail.get("message") or "")[:300]
            elif skip is not None:
                out["skipped"] += 1
                out["cases"][name] = "SKIP: " + (skip.get("message") or "")[:120]
            else:
                out["passed"] += 1
                out["cases"][name] = "PASS"
    finally:
        os.remove(xml)
    return out


def _seed_table() -> dict:
    import seed_conventions as S
    try:
        ref = json.load(open(S.TABLE))
        now = S.build_table()
        bad = S.compare(ref, now)
        return {"status": "PASS" if not bad else "DIFFER", "specs": len(now), "drifts": bad[:60]}
    except Exception as e:                               # noqa: BLE001
        return {"status": "ERROR", "specs": 0, "drifts": [f"{type(e).__name__}: {e}"[:300]]}


def _working_point(name: str, device: str, quick: bool) -> dict:
    ref = R.read_fingerprint(name)
    cut = ref["cut"]["n_frames"]
    if quick:
        if name not in QUICK:
            return {"status": "SKIP", "violations": ["not in the quick set"], "ref": ref}
        cut = min(cut, QUICK[name])
    cps = [f for f in ref["cut"]["checkpoints"] if f <= cut]
    t0 = time.time()
    try:
        import yaml
        if ref.get("archive") and os.path.exists(os.path.join(ref["archive"], "spec.yaml")):
            spec = yaml.safe_load(open(os.path.join(ref["archive"], "spec.yaml")))
        else:
            spec = yaml.safe_load(open(os.path.join(ROOT, "config", ref["family"], name + ".yaml")))
        got = R.fingerprint_fresh(spec, ref["family"], cut, device, "nightly", checkpoints=cps,
                                  stride=int(ref["cut"].get("record_stride", 1)))
    except Exception as e:                               # noqa: BLE001
        return {"status": "ERROR", "violations": [f"{type(e).__name__}: {e}"[-600:]], "ref": ref,
                "seconds": round(time.time() - t0, 1)}
    ref_cut = dict(ref, cut=dict(ref["cut"], checkpoints=cps))
    bad = R.compare(ref_cut, got)
    tv, note = R.compare_timing(ref, got)
    return {
        "status": "PASS" if not bad and tv is None else "DIFFER",
        "violations": bad + ([tv] if tv else []),
        "timing_note": note,
        "cut": cut,
        "ms_per_frame": got.get("timing", {}).get("ms_per_frame"),
        "ref_ms_per_frame": ref.get("timing", {}).get("ms_per_frame"),
        "ref_device": ref.get("timing", {}).get("device"),
        "checkpoints": got["checkpoints"],
        "ref_checkpoints": {k: v for k, v in ref["checkpoints"].items() if int(k) in cps},
        "ref_archived_on": ref.get("archived_on"),
        "ref_commit": ref.get("generated_at_commit"),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", action="append", default=[], help="working point name (repeatable)")
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--skip-tests", action="store_true", help="working points only")
    ap.add_argument("--open", action="store_true", help="open the dashboard when done")
    ap.add_argument("--note", default="", help="free text stored with the run")
    args = ap.parse_args()

    t_start = time.time()
    rec = {
        "run_id": time.strftime("%Y-%m-%d_%H%M%S"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": bool(_git("status", "--porcelain", "--", "src", "config", "tools", "tests")),
        "host": platform.node(),
        "device": args.device,
        "gpu": R.gpu_name(args.device),
        "quick": args.quick,
        "note": args.note,
        "seed_table": None, "invariants": None, "scaling": None, "working_points": {},
    }
    print(f"[regression] {rec['run_id']}  commit {rec['commit']}{' (dirty)' if rec['dirty'] else ''}  "
          f"{rec['host']}  {rec['gpu']}", flush=True)

    if not args.skip_tests and not args.only:
        rec["seed_table"] = _seed_table()
        print(f"[regression] seed table: {rec['seed_table']['status']} ({rec['seed_table']['specs']} specs)", flush=True)
        for key, path in (("invariants", "tests/regression/test_invariants.py"),
                          ("scaling", "tests/regression/test_scaling.py")):
            rec[key] = _pytest_file(path, args.device)
            print(f"[regression] {key}: {rec[key]['passed']} passed, {rec[key]['failed']} failed, "
                  f"{rec[key]['skipped']} skipped", flush=True)

    names = args.only or R.registered()
    for name in names:
        r = _working_point(name, args.device, args.quick)
        rec["working_points"][name] = r
        ms = f"{r['ms_per_frame']:.0f} ms/frame" if r.get("ms_per_frame") else ""
        print(f"[regression] {name:30s} {r['status']:6s} {ms:>14s}  {r.get('seconds', 0):6.0f} s"
              + ("".join("\n      " + v for v in r["violations"]) if r["status"] != "PASS" else ""), flush=True)

    rec["duration_s"] = round(time.time() - t_start, 1)
    wp = rec["working_points"]
    rec["summary"] = {
        "pass": sum(1 for v in wp.values() if v["status"] == "PASS"),
        "differ": sum(1 for v in wp.values() if v["status"] == "DIFFER"),
        "error": sum(1 for v in wp.values() if v["status"] == "ERROR"),
        "skip": sum(1 for v in wp.values() if v["status"] == "SKIP"),
        "tests_failed": sum((rec[k] or {}).get("failed", 0) for k in ("invariants", "scaling"))
                        + (0 if (rec["seed_table"] or {}).get("status", "PASS") == "PASS" else 1),
    }
    os.makedirs(os.path.dirname(args.archive), exist_ok=True)
    with open(args.archive, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    s = rec["summary"]
    print(f"[regression] done in {rec['duration_s']/60:.1f} min: {s['pass']} pass, {s['differ']} differ, "
          f"{s['error']} error, {s['skip']} skipped; {s['tests_failed']} test file/table failures  "
          f"-> {os.path.relpath(args.archive, ROOT)}", flush=True)
    if args.open:
        subprocess.Popen([sys.executable, os.path.join(HERE, "regression_dashboard.py"), "--archive", args.archive])
    sys.exit(0 if s["differ"] == 0 and s["error"] == 0 and s["tests_failed"] == 0 else 1)


if __name__ == "__main__":
    main()
