#!/usr/bin/env python
"""certify_apparatus -- the Phase 0 gate. Nothing in this campaign runs until it prints PASS.

WHAT THIS GATE IS FOR
================================================================================================
Six of the previous campaign's sixty batches -- one in ten -- were destroyed by faults that cost
seconds to check: an operator renamed in the library while the campaign code kept the old name,
three consecutive crashes while wiring a new mechanism, and a data file that moved while the
loader kept a stale path. Not one of those was a scientific question. Every one of them burned a
batch of GPU time and a day of somebody's attention.

None of the checks below is science either. Each one makes a question askable.

A GATE THAT CANNOT FAIL IS NOT A GATE
------------------------------------------------------------------------------------------------
So this script has a `--canary` mode that BREAKS the apparatus on purpose, six ways, and demands
that the gate catch all six. A check nobody has watched fail is a check nobody should trust: the
previous campaign's own instrument documentation asserted that a do-nothing model scores zero,
and it scores +0.075.

    python certify_apparatus.py            # the gate
    python certify_apparatus.py --canary   # break it six ways; all six must be caught
    python certify_apparatus.py --fit      # also run a real short fit (slow; needs the engine)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(REPO, "src")
PY = sys.executable
SPEC = "material/material_aniso_cardio"

sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Phrases from the retired ledger. If any of these is still compiled into the campaign's source,
# every agent that reads the code inherits a conclusion we have decided not to hold.
RETIRED_LEDGER = [
    r"Falsified#", r"net-harmful", r"ledger", r"Est\.#", r"dose-confirm",
    r"campaign law", r"peak_ratio\s*~?0\.5", r"REJECTED \(", r"B\d\d (total )?loss",
]
# Scanned for the retired ledger. `certify_apparatus.py` is NOT here, and cannot be: it has to
# name the phrases in order to search for them. That is the only exclusion, and canary 6 proves
# the scan still fires on every other file.
SOURCE_FILES = ["train.py", "data.py", "ingest.py", "determinism.py", "provenance.py"]


class Check:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append({"check": name, "pass": bool(ok), "detail": str(detail)})
        return ok

    def report(self, title):
        print(f"\n{'=' * 92}\n  {title}\n{'=' * 92}")
        for r in self.rows:
            mark = "  ok  " if r["pass"] else " FAIL "
            print(f"  [{mark}] {r['check']:<44s} {r['detail']}")
        ok = all(r["pass"] for r in self.rows)
        print(f"\n  GATE: {'PASS' if ok else 'FAIL'}  ({sum(r['pass'] for r in self.rows)}"
              f"/{len(self.rows)} checks)")
        print("=" * 92)
        return ok


# ---------------------------------------------------------------------------------------------
# 1. OPERATORS. The list is derived FROM THE SPEC, never typed by hand -- a hand-typed list is
#    the same defect it is meant to catch, one indirection later.
# ---------------------------------------------------------------------------------------------
def check_operators(c):
    try:
        import plexus.operators                      # importing is what REGISTERS them
        from plexus.schema import load
        from plexus.paths import resolve_config
        from plexus.models.registry import get_operator
        spec = load(resolve_config(SPEC)[0])
        names = sorted({o.op for o in spec.operators})
        missing = [n for n in names if _resolve_fails(get_operator, n)]
        return c.add("operators in the spec resolve in the registry",
                     not missing,
                     f"{len(names) - len(missing)}/{len(names)} resolve"
                     + (f"; MISSING {missing}" if missing else f": {', '.join(names)}"))
    except Exception as e:
        return c.add("operators in the spec resolve in the registry", False, f"{type(e).__name__}: {e}")


def _resolve_fails(get_operator, name):
    try:
        get_operator(name)
        return False
    except Exception:
        return True


# ---------------------------------------------------------------------------------------------
# 2. DATA. One path, checked content, and refusals that actually fire.
# ---------------------------------------------------------------------------------------------
def check_data(c):
    import data as D
    ok = True
    try:
        z = D.open_npz(expect_sha256=D.HEALTHY_POS_SHA256)
        b = D.beats(z["pos"])
        ok &= c.add("recording opens and matches its declared content",
                    True, f"{b['n_frames']} frames, {b['n_nodes']} nodes, onsets {b['onsets']}")
        ok &= c.add("beat detector is reproducible",
                    b["onsets"] == [2, 51, 101, 152, 204] and b["gaps"] == [49, 50, 51, 52],
                    f"gaps {b['gaps']}, mean {b['mean_gap']}, reported period {b['period']}")
    except Exception as e:
        ok &= c.add("recording opens and matches its declared content", False, f"{type(e).__name__}: {e}")
    # refusals
    for label, fn in (("missing file", lambda: D.open_npz("/nonexistent/nope.npz")),
                      ("wrong content", lambda: D.open_npz(expect_sha256="0" * 64))):
        try:
            fn(); ok &= c.add(f"loader refuses: {label}", False, "NO REFUSAL -- it guessed")
        except D.DataRefusal:
            ok &= c.add(f"loader refuses: {label}", True, "raised DataRefusal")
        except Exception as e:
            ok &= c.add(f"loader refuses: {label}", False, f"wrong exception {type(e).__name__}")
    ok &= c.add("loader has no fallback search order", "_find_npz" not in open(os.path.join(HERE, "data.py")).read(),
                "one explicit path")
    return ok


# ---------------------------------------------------------------------------------------------
# 3. INGEST. The recording rebuilds from committed code, bit for bit.
# ---------------------------------------------------------------------------------------------
def check_ingest(c):
    import data as D
    import ingest
    try:
        ok, rep = ingest.verify(D.DEFAULT_NPZ)
        return c.add("recording rebuilds bit-exactly from the microscope derivatives", ok,
                     f"pos bit_exact={rep['pos']['bit_exact']}, dt equal={rep['dt']['equal']}")
    except FileNotFoundError as e:
        return c.add("recording rebuilds bit-exactly from the microscope derivatives", False, str(e))


# ---------------------------------------------------------------------------------------------
# 4. DETERMINISM. The seed must control the draw, and be the ONLY thing that does.
# ---------------------------------------------------------------------------------------------
def check_determinism(c):
    import determinism as DET
    import torch
    ok = c.add("seed controls the draw, and only the seed does", DET.selftest("cpu", verbose=False),
               "same seed identical, other seed differs")
    if torch.cuda.is_available():
        ok &= c.add("same, on the GPU", DET.selftest("cuda:0", verbose=False), "")
    # CODE only, comments stripped. The retired-ledger scan does the opposite -- it must read
    # comments, because that is where beliefs hide. Here a comment describing the removal of a
    # setter is not a setter, and a check that cannot tell those apart is not measuring the code.
    code = "\n".join(l.split("#", 1)[0] for l in open(os.path.join(HERE, "train.py")))
    setters = [t for t in ("use_deterministic_algorithms", "allow_tf32", "cudnn.benchmark")
               if t in code]
    ok &= c.add("no module sets the arithmetic flags behind our back", not setters,
                "determinism.enforce is the only place they are set"
                if not setters else f"train.py sets {setters} directly")
    return ok


# ---------------------------------------------------------------------------------------------
# 5. PROVENANCE. A run must carry its own source.
# ---------------------------------------------------------------------------------------------
def check_provenance(c):
    import data as D
    import provenance as PROV
    d = tempfile.mkdtemp(prefix="cert_prov_")
    PROV.write_manifest(d, inputs=[("recording", D.DEFAULT_NPZ, "check")], extra={"seed": 0})
    ok, why = PROV.check_manifest(d)
    n = len(json.load(open(os.path.join(d, "run_manifest.json")))["sources"])
    return c.add("a run archives its own source and hashes its inputs", ok,
                 f"{n} source files archived" if ok else "; ".join(why))


# ---------------------------------------------------------------------------------------------
# 6. THE INHERITED LEDGER. No retired conclusion may remain compiled into the source.
# ---------------------------------------------------------------------------------------------
def check_no_inherited_beliefs(c):
    hits = []
    for f in SOURCE_FILES:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        for i, line in enumerate(open(p, errors="replace"), 1):
            for pat in RETIRED_LEDGER:
                if re.search(pat, line):
                    hits.append(f"{f}:{i} /{pat}/")
    return c.add("no retired conclusion is compiled into the source", not hits,
                 "clean" if not hits else f"{len(hits)} hits: {hits[:3]}")


# ---------------------------------------------------------------------------------------------
# 7. THE REGISTERS. We must be able to say exactly what we believe, and it must be nothing.
# ---------------------------------------------------------------------------------------------
def check_registers(c):
    ok = True
    hp = os.path.join(HERE, "HYPOTHESES.md")
    bp = os.path.join(HERE, "BELIEFS.md")
    if not os.path.exists(hp):
        return c.add("the two registers exist", False, "HYPOTHESES.md missing")
    if not os.path.exists(bp):
        return c.add("the two registers exist", False, "BELIEFS.md missing")
    htxt, btxt = open(hp).read(), open(bp).read()
    rows = re.findall(r"^\|\s*(H\d+)\s*\|.*$", htxt, re.M)
    STATUSES = ("untested", "re-tested: confirmed", "re-tested: refuted", "unscoreable")
    unstatused = [r for r in re.findall(r"^\|\s*H\d+\s*\|.*$", htxt, re.M)
                  if not any(st in r for st in STATUSES)]
    ok &= c.add("every inherited claim carries an open status", rows and not unstatused,
                f"{len(rows)} hypotheses, all statused" if not unstatused
                else f"{len(unstatused)} without a recognised status")
    confirmed = [r for r in re.findall(r"^\|\s*H\d+\s*\|.*$", htxt, re.M)
                 if "re-tested: confirmed" in r]        # table ROWS only, not the vocabulary note
    ok &= c.add("no inherited claim is recorded as established", not confirmed,
                "nothing confirmed yet" if not confirmed else f"{len(confirmed)} confirmed")
    entries = [l for l in btxt.splitlines() if re.match(r"^\|\s*B\d+\s*\|", l)]
    ok &= c.add("the belief register is EMPTY", len(entries) == 0,
                "0 entries" if not entries else f"{len(entries)} entries -- nothing is earned yet")
    return ok


# ---------------------------------------------------------------------------------------------
# 8. THE FIT. Optional because it is the only slow check.
# ---------------------------------------------------------------------------------------------
def check_fit(c, device="cpu"):
    env = dict(os.environ, PYTHONPATH=SRC + ":" + os.environ.get("PYTHONPATH", ""))
    outs = []
    for i in (1, 2):
        d = tempfile.mkdtemp(prefix=f"cert_fit{i}_")
        r = subprocess.run([PY, os.path.join(HERE, "train.py"), SPEC, "--smoke", "1",
                            "--device", device, "--outdir", d, "--seed", "11"],
                           capture_output=True, text=True, env=env, timeout=3600)
        if r.returncode != 0:
            return c.add("a short fit runs end to end", False, r.stderr.strip().splitlines()[-1][:110])
        outs.append(d)
    c.add("a short fit runs end to end", True, f"2 runs on {device}")
    import torch
    def flat(d):
        p = sorted(glob.glob(os.path.join(d, "checkpoints", "model_*.pt")))[-1]
        sd = torch.load(p, map_location="cpu", weights_only=False)
        ts = []
        def walk(o):
            if torch.is_tensor(o): ts.append(o.detach().flatten().float())
            elif isinstance(o, dict):
                for k in sorted(o): walk(o[k])
            elif isinstance(o, (list, tuple)):
                for x in o: walk(x)
        walk(sd); return torch.cat(ts)
    a, b = flat(outs[0]), flat(outs[1])
    same = bool(torch.equal(a, b))
    return c.add(f"two fits at one seed are bit-identical ({device})", same,
                 f"{a.numel()} params, max|d|={float((a - b).abs().max()):.3e}")


# ---------------------------------------------------------------------------------------------
# 9. THE GPU DEBT. Determinism is achievable on the CPU and NOT on the GPU, because
#    `plexus.models.base.Field.sample` uses grid_sample and `grid_sampler_2d_backward_cuda` has no
#    deterministic implementation -- `active_stress` calls it every frame. That is a change to the
#    shared library, so it is not Phase 0 work.
#
#    The rule from the note applies: what cannot be certified is RECORDED, never quietly relaxed.
#    So instead of asserting bit-identity we MEASURE the same-seed spread and write it down. Phase
#    2 needs that number anyway -- it is one of the two noise floors, and the previous campaign
#    had neither.
# ---------------------------------------------------------------------------------------------
def check_gpu_repeat(c, n=3, device="cuda:0"):
    import torch
    if not torch.cuda.is_available():
        return c.add("GPU same-seed spread measured", True, "no GPU present; skipped")
    env = dict(os.environ, PYTHONPATH=SRC + ":" + os.environ.get("PYTHONPATH", ""))
    outs = []
    for i in range(n):
        d = tempfile.mkdtemp(prefix=f"cert_gpu{i}_")
        r = subprocess.run([PY, os.path.join(HERE, "train.py"), SPEC, "--smoke", "1",
                            "--device", device, "--outdir", d, "--seed", "11",
                            "--allow_nondeterministic_ops", "1"],
                           capture_output=True, text=True, env=env, timeout=3600)
        if r.returncode != 0:
            return c.add("GPU same-seed spread measured", False,
                         r.stderr.strip().splitlines()[-1][:110])
        outs.append(d)

    def flat(d):
        pth = sorted(glob.glob(os.path.join(d, "checkpoints", "model_*.pt")))[-1]
        sd = torch.load(pth, map_location="cpu", weights_only=False)
        ts = []
        def walk(o):
            if torch.is_tensor(o): ts.append(o.detach().flatten().float())
            elif isinstance(o, dict):
                for k in sorted(o): walk(o[k])
            elif isinstance(o, (list, tuple)):
                for x in o: walk(x)
        walk(sd); return torch.cat(ts)

    v = [flat(d) for d in outs]
    scale = float(v[0].abs().max())
    worst, ident = 0.0, 0
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            worst = max(worst, float((v[i] - v[j]).abs().max()))
            ident += int(bool(torch.equal(v[i], v[j])))
    rec = {"device": device, "runs": n, "params": int(v[0].numel()),
           "max_abs_pairwise": worst, "relative": worst / scale if scale else 0.0,
           "identical_pairs": ident, "iterations": 2,
           "blocked_by": "grid_sampler_2d_backward_cuda (plexus.models.base.Field.sample, "
                         "called by active_stress)",
           "note": "measured at 2 iterations; it will grow with optimisation depth, and Phase 2 "
                   "must re-measure it at the depth the loop actually uses"}
    json.dump(rec, open(os.path.join(HERE, "_metrology", "gpu_repeat.json"), "w"), indent=1)
    # The CHECK is that the number exists and is recorded, not that it is zero -- it is not zero,
    # and pretending otherwise is what the previous campaign did for sixty rounds.
    return c.add("GPU same-seed spread measured and recorded", True,
                 f"max|d|={worst:.2e} (rel {worst / scale:.1e}) over {n} runs -- NOT bit-identical")


# ---------------------------------------------------------------------------------------------
# THE CANARIES. Break it on purpose; every break must be caught.
# ---------------------------------------------------------------------------------------------
def canary():
    import data as D
    import provenance as PROV
    results = []

    def note(name, caught, detail=""):
        results.append({"canary": name, "caught": bool(caught), "detail": detail})

    # 1. the recording is not where it is declared to be
    try:
        D.open_npz("/tmp/definitely_not_here.npz"); note("missing recording", False, "no refusal")
    except D.DataRefusal:
        note("missing recording", True, "refused")

    # 2. the recording is a DIFFERENT recording under the right filename
    import numpy as np
    tmp = tempfile.mktemp(suffix=".npz")
    z = D.open_npz()
    np.savez(tmp, pos=z["pos"][::-1].copy(), vel=z["vel"], ids=z["ids"], dt=z["dt"], source=z["source"])
    try:
        D.open_npz(tmp, D.HEALTHY_POS_SHA256); note("substituted recording", False, "accepted a different array")
    except D.DataRefusal:
        note("substituted recording", True, "content hash refused it")

    # 3. an operator the spec names has been renamed in the library
    c2 = Check()
    try:
        from plexus.models.registry import get_operator
        import plexus.models.registry as REG
        real = REG.get_operator
        REG.get_operator = lambda n: (_ for _ in ()).throw(KeyError(n)) if n == "active_stress" else real(n)
        ok = check_operators(c2)
        note("renamed operator", not ok, "gate refused" if not ok else "gate passed a missing operator")
    finally:
        REG.get_operator = real

    # 4. a run that does not archive its own source
    d = tempfile.mkdtemp(prefix="canary_prov_")
    PROV.write_manifest(d, inputs=[("recording", D.DEFAULT_NPZ, "x")], extra={"seed": 0})
    for f in glob.glob(os.path.join(d, "_src", "*.py")):
        os.remove(f)
    ok, _ = PROV.check_manifest(d)
    note("run source deleted after the fact", not ok, "manifest check refused" if not ok else "accepted")

    # 5. a run with no seed recorded
    d2 = tempfile.mkdtemp(prefix="canary_seed_")
    PROV.write_manifest(d2, inputs=[("recording", D.DEFAULT_NPZ, "x")])       # no seed in extra
    ok2, why2 = PROV.check_manifest(d2)
    note("no seed recorded", not ok2, "refused" if not ok2 else "accepted a run with no seed")

    # 6. a retired conclusion re-enters the source
    c3 = Check()
    victim = os.path.join(HERE, "data.py")
    orig = open(victim).read()
    try:
        open(victim, "w").write(orig + "\n# net-harmful Falsified#8 -- reinstated by the canary\n")
        ok3 = check_no_inherited_beliefs(c3)
        note("retired conclusion re-enters the source", not ok3,
             "gate refused" if not ok3 else "gate did not notice")
    finally:
        open(victim, "w").write(orig)

    print(f"\n{'=' * 92}\n  CANARIES -- the apparatus broken on purpose\n{'=' * 92}")
    for r in results:
        print(f"  [{'  ok  ' if r['caught'] else ' MISS '}] {r['canary']:<44s} {r['detail']}")
    n = sum(r["caught"] for r in results)
    print(f"\n  CANARY: {n}/{len(results)} caught -- {'PASS' if n == len(results) else 'FAIL'}")
    print("=" * 92)
    json.dump(results, open(os.path.join(HERE, "_metrology", "canary.json"), "w"), indent=1)
    return n == len(results)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--fit", action="store_true", help="also run two short fits (slow)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)

    if a.canary:
        return 0 if canary() else 1

    c = Check()
    check_operators(c)
    check_data(c)
    check_ingest(c)
    check_determinism(c)
    check_provenance(c)
    check_no_inherited_beliefs(c)
    check_registers(c)
    if a.fit:
        check_fit(c, a.device)
        check_gpu_repeat(c)
    ok = c.report("PHASE 0 GATE -- the apparatus runs, repeats, and says what it is")
    json.dump({"checks": c.rows, "pass": ok},
              open(os.path.join(HERE, "_metrology", "apparatus.json"), "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
