#!/usr/bin/env python
"""Run the gates in `config/gates/`, measure them, assert their declared thresholds, write the table.

WHAT A GATE IS HERE, and it is not what `discovery_okuda/ops/test_0*.py` was. Those scripts built
their spec imperatively, loaded a cached tissue instead of running one, and REPORTED numbers. There
was nothing to read before the run and nothing that could fail. A gate is now two things in one file:

    config/gates/<name>.yaml     the model, as an ordinary Plexus spec the engine runs
      + its `_gate:` block       the questions, the tiers, and the thresholds

and the outputs go to a dedicated tree, `log/gates/<name>/`, leaving `log/okuda_ECM/` untouched as
the okuda side so the two can always be diffed.

THE THRESHOLD IS DECLARED BEFORE THE RUN, AND THAT IS AUDITABLE. `--freeze-reference` records a
sha1 of the canonicalised `_gate:` block beside the reference run. Every later grading re-hashes and
refuses to grade if the block has moved, naming the rows that changed. The paper's rule --
*a threshold chosen after seeing the number is not a threshold* -- is otherwise an honour system.

TIERS ARE THE PAPER'S AND `basis:` IS THE HONEST PART. `bookkeeping` asks whether the code does what
the operator says; `closed_form` whether the implementation reproduces the physics it was GIVEN;
`measurement` whether the model agrees with something OBSERVED IN CELLS. A threshold copied off a
previous run of the same code is none of those three -- it is a regression pin -- so every row also
declares `basis: identity | analytic | reference | literature`, and the roll-up reports the split.
Without that column a suite of regression pins reads as validation.

UNITS. A row whose measure is physical converts through the spec's `units:` block and RAISES if none
is declared. That is the mechanism that stops a dimensionless run quoting a micrometre -- the failure
that once made a membrane 24x too thick and a modulus a pressure.

    python tools/run_gates.py --list
    python tools/run_gates.py --gate 00_spheroid              run it, measure it, grade it
    python tools/run_gates.py --gate 00_spheroid --measure-only     grade what is already on disk
    python tools/run_gates.py --gate 00_spheroid --freeze-reference
    python tools/run_gates.py --all --cluster                 submit each gate to gpu_l4

EXIT CODES.  0 all rows pass  |  1 a row failed  |  2 a row is blocked and none failed
             3 preflight or infrastructure  |  4 a known-red row turned green
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "config", "gates")
LOG = os.path.join(ROOT, "log", "gates")
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import gate_measures as GM                                            # noqa: E402

# A UNIT STRING THAT NAMES THE MESH IS REFUSED -- ON A MEASUREMENT ROW. The paper's rule is that a
# threshold belongs in the unit of the phenomenon, and the mechanical form of that rule is this list:
# "0.82 grid cells" sounds small and is 15 um, nearly two cell diameters.
#
# IT DOES NOT APPLY TO A BOOKKEEPING ROW, and the first version of this check got that wrong: it
# refused gate 01's `myosin_array_aligned_with_half_edges`, whose unit is "entries of disagreement
# between the myosin array and the half-edge table". That row IS about the array -- it asserts that
# `junction_sync` re-keyed a per-half-edge store after the topology moved under it -- and stating it
# in cells or micrometres would be the dishonest version. The same goes for `reservoir_headroom`,
# which is a fraction of a buffer because the question is about the buffer. The rule is that a claim
# about the WORLD may not be denominated in the mesh; a claim about the CODE must be.
_MESH_UNITS = ("grid cell", " dx", "lattice", " slot")


def _sha1(o):
    return hashlib.sha1(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:16]


def gates():
    out = {}
    for p in sorted(glob.glob(os.path.join(CFG, "*.yaml"))):
        c = yaml.safe_load(open(p))
        if isinstance(c, dict) and "_gate" in c:
            out[c["_gate"]["id"]] = (p, c)
    return out


# ---------------------------------------------------------------------------------- preflight
def preflight(path, cfg):
    """Everything that can be checked WITHOUT the GPU, checked before any is spent."""
    bad = []
    g = cfg["_gate"]
    try:
        import plexus.operators                                       # noqa: F401
        import plexus.schema as S
        S.load(path)
    except Exception as e:
        bad.append(f"the spec does not load: {type(e).__name__}: {str(e)[:120]}")
    known_specs = {yaml.safe_load(open(q))["general"]["name"]
                   for q in glob.glob(os.path.join(CFG, "*.yaml"))}
    for label, spec_name in (g.get("arms") or {}).items():
        if spec_name not in known_specs:
            bad.append(f"arm {label!r} names spec {spec_name!r}, which is not in config/gates/")
    seen = set()
    for m in g.get("measures", []):
        n = m.get("name", "?")
        if n in seen:
            bad.append(f"{n}: duplicate row name")
        seen.add(n)
        if m.get("fn") not in GM.MEASURES:
            bad.append(f"{n}: fn {m.get('fn')!r} is not in gate_measures.MEASURES")
        if m.get("tier") not in ("bookkeeping", "closed_form", "measurement"):
            bad.append(f"{n}: tier {m.get('tier')!r} is not one of the paper's three")
        if m.get("basis") not in ("identity", "analytic", "reference", "literature"):
            bad.append(f"{n}: basis {m.get('basis')!r} is not identity/analytic/reference/literature")
        for k in ("unit", "assert", "why"):
            if not m.get(k):
                bad.append(f"{n}: no {k}:")
        u = str(m.get("unit", "")).lower()
        if m.get("tier") == "measurement" and any(t in u for t in _MESH_UNITS):
            bad.append(f"{n}: a MEASUREMENT row may not be denominated in the mesh "
                       f"({m['unit']!r}); state it in the unit of the phenomenon")
        for w in (m.get("arms") or ([m["arm"]] if m.get("arm") else [])):
            if w != "self" and w not in (g.get("arms") or {}):
                bad.append(f"{n}: arm {w!r} is not declared in _gate.arms")
        a = m.get("assert") or {}
        if len(a) != 1 or next(iter(a)) not in GM.ASSERTS:
            bad.append(f"{n}: assert {a!r} must be exactly one of {sorted(GM.ASSERTS)}")
        # A PHYSICAL ROW WITHOUT A `units:` BLOCK IS THE ERROR THIS EXISTS TO CATCH.
        if m.get("fn") in GM.PHYSICAL and not (cfg.get("general", {}) or {}).get("units"):
            bad.append(f"{n}: {m['fn']} is physical ({GM.PHYSICAL[m['fn']][1]}) but the spec "
                       f"declares no `general.units:` -- this row would quote a unit it has not earned")
    return bad


# ---------------------------------------------------------------------------------- running
def data_dir_of(name):
    from plexus.paths import graphs_data_path
    return graphs_data_path("gates", name)


def submit_cluster(path, cfg, force=False, pre_folder="gates"):
    """Submit the gate to gpu_l4 and return the job name. Same node type, same environment and the
    same `PLEXUS_STRICT_DETERMINISM=1` as the twin-run harness, so a gate and a twin row are
    comparable rather than merely both green."""
    sys.path.insert(0, os.path.join(ROOT, "discovery_okuda"))
    import cluster as C
    name = cfg["general"]["name"]
    # `pre_folder` IS EXPLICIT because `plexus.paths` gives `gates` and `atlas` no trigger
    # substrings on purpose: a spec is routed by the folder its caller names, never inferred from
    # a name that happens to contain the word. Inferring "gates" from a name containing "gate" is
    # how a spec ends up in a folder nobody chose.
    out = os.path.join(ROOT, "log", pre_folder, cfg["_gate"]["id"])
    os.makedirs(out, exist_ok=True)
    sh = os.path.join(out, "run.sh")
    with open(sh, "w") as f:
        f.write("\n".join([
            "#!/bin/bash -l",
            f"cd {C.cpath(ROOT)}",
            f"export PYTHONPATH={C.cpath(os.path.join(ROOT, 'src'))}",
            "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
            "export MPLBACKEND=Agg PLEXUS_STRICT_DETERMINISM=1",
            # NO `--no-describe`. That flag does more than skip the captioner: `Plexus_Main`
            # gates the whole PLOT block on it, so the gate would land with a trajectory and no
            # `3d.png` and no movie -- and a gate whose only output is a table sends you back to
            # the cluster to find out what a red row looks like.
            f"conda run -n {C.ENV} python Plexus_Main.py -o generate {pre_folder}/{name} "
            f"--device cuda:0" + (" --force" if force else ""),
        ]) + "\n")
    os.chmod(sh, 0o755)
    o = C.cpath(os.path.join(out, "run.out"))
    gpu = "-gpu num=1 " if C.GPU != "0" else ""
    excl = "".join(f'-R "hname!={h}" ' for h in C.EXCLUDE_HOSTS if h)
    cmd = (f"bsub -n {C.NCPUS} {gpu}{excl}-q {C.QUEUE} -W {C.WALL} -J {pre_folder}_{name} "
           f"-o {o} -e {o[:-4]}.err bash -l {C.cpath(sh)}")
    C._ssh(cmd, timeout=60)
    return f"{pre_folder}_{name}"


def run_one(path, cfg, device="cuda:0", force=False):
    """`Plexus_Main.py -o generate gates/<name>`. A FRESH PROCESS, and that is not tidiness.

    `ecm_ops.STRESS_HISTORY`, `contact_ops.CONTACT_HISTORY` and their siblings are MODULE GLOBALS
    that the old rigs cleared by hand between runs. Two gates graded in one interpreter would
    concatenate them, and every last-frame stress row would read off the wrong offset -- a wrong
    number, not an error.
    """
    name = cfg["general"]["name"]
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"),
               PLEXUS_STRICT_DETERMINISM="1", MPLBACKEND="Agg")
    cmd = [sys.executable, "Plexus_Main.py", "-o", "generate", f"gates/{name}",
           "--device", device]        # see submit_cluster on why --no-describe is not passed
    if force:
        cmd.append("--force")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=6 * 3600)
    out = os.path.join(LOG, cfg["_gate"]["id"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run.log"), "w") as f:
        f.write(r.stdout[-400_000:] + "\n=== stderr ===\n" + r.stderr[-200_000:])
    return r.returncode, time.perf_counter() - t0


# ---------------------------------------------------------------------------------- grading
def _convert(val, fn, cfg):
    """Simulation units -> the row's physical unit, through the spec's `units:` block."""
    if fn not in GM.PHYSICAL:
        return val
    kind, _u = GM.PHYSICAL[fn]
    u = (cfg.get("general", {}) or {}).get("units") or {}
    if kind == "length":
        k = float(u["length_um"])
    elif kind == "time":
        k = float(u["time_s"]) / 3600.0                      # frames -> hours
    elif kind == "stress":
        # DERIVED, never declared: a stress is force/length^2, so it comes from the two base scales
        # and cannot be quoted without BOTH. `plexus.units` is the one place that arithmetic lives.
        from plexus.units import parse as _parse_units
        U = _parse_units(u)
        if U.stress_Pa is None:
            raise KeyError("a stress row needs `force_nN` as well as `length_um`; the spec declares "
                           "only one of them, so the number has no Pa to be quoted in")
        k = float(U.stress_Pa)
    else:
        raise KeyError(kind)
    return [x * k for x in val] if isinstance(val, list) else val * k


def open_arms(cfg):
    """{label: Traj} for a gate's arms, plus `self` for its own spec.

    A CONTRAST NEEDS TWO RUNS, and gate 01's whole question is a contrast: does a tissue with a
    myosin belt intercalate LESS than the same tissue without one, and is `junction_sync` -- which
    only re-keys a bookkeeping array -- trajectory-neutral? Neither is answerable from one
    trajectory, and a gate that answered them from one would be comparing a number with a memory.

    An arm names another spec in `config/gates/`, so an arm is itself a declared, runnable file and
    not a variant hidden inside the grader.
    """
    out = {"self": GM.open_traj(data_dir_of(cfg["general"]["name"]))}
    for label, spec_name in (cfg["_gate"].get("arms") or {}).items():
        d = data_dir_of(spec_name)
        if not os.path.isdir(d):
            out[label] = None                     # a missing arm blocks its rows; it does not pass
            continue
        out[label] = GM.open_traj(d)
    return out


def grade(path, cfg, traj_dir):
    g = cfg["_gate"]
    arms = open_arms(cfg)
    T = arms["self"]
    rows, n_fail, n_block, n_turned = [], 0, 0, 0
    for m in g["measures"]:
        row = dict(name=m["name"], tier=m["tier"], basis=m["basis"], unit=m["unit"],
                   fn=m["fn"], args=m.get("args", {}), reduce=m.get("reduce", "all"),
                   assertion=m["assert"], why=" ".join(str(m["why"]).split()),
                   status=m.get("status", "ok"))
        if row["status"] == "blocked":
            row.update(outcome="BLOCKED", value=None, blocked_by=m.get("blocked_by", "unstated"))
            n_block += 1
            rows.append(row)
            continue
        try:
            # A row may name ONE arm (`arm:`) or TWO (`arms: [a, b]`); the default is the gate's own
            # trajectory. A row whose arm did not run is BLOCKED, never passed.
            want = m.get("arms") or ([m["arm"]] if m.get("arm") else ["self"])
            sel = [arms.get(w) for w in want]
            if any(x is None for x in sel):
                missing = [w for w, x in zip(want, sel) if x is None]
                row.update(outcome="BLOCKED", value=None,
                           blocked_by=f"arm(s) {missing} have not been run")
                n_block += 1
                rows.append(row)
                continue
            series = GM.MEASURES[m["fn"]](*sel, **m.get("args", {}))
            val = GM.REDUCERS[row["reduce"]](series)
            val = _convert(val, m["fn"], cfg)
            kind, arg = next(iter(m["assert"].items()))
            ok = GM.ASSERTS[kind](val, arg)
        except Exception as e:
            row.update(outcome="INFRA_FAIL", value=None,
                       error=f"{type(e).__name__}: {str(e)[:160]}")
            n_fail += 1
            rows.append(row)
            continue
        row["value"] = (val if not isinstance(val, list) else
                        (val[0] if len(set(map(str, val))) == 1 else
                         {"min": min(val), "max": max(val), "last": val[-1], "n": len(val)}))
        if row["status"] == "known_red":
            row["outcome"] = "TURNED_GREEN" if ok else "KNOWN_RED"
            n_turned += int(ok)
        else:
            row["outcome"] = "PASS" if ok else "FAIL"
            n_fail += int(not ok)
        rows.append(row)
    return rows, n_fail, n_block, n_turned


def tier_split(rows):
    """The number the promotion note has to carry, and it is two numbers, not one.

    The paper's tiers say what KIND of question a row asks; `basis` says where its number came from.
    A `measurement`-tier row whose threshold is a previous run of the same code is a REGRESSION PIN,
    and counting it as validation is how a suite of self-comparisons comes to read as biology.
    """
    t = {k: sum(1 for r in rows if r["tier"] == k)
         for k in ("bookkeeping", "closed_form", "measurement")}
    b = {k: sum(1 for r in rows if r["basis"] == k)
         for k in ("identity", "analytic", "reference", "literature")}
    n = max(len(rows), 1)
    return dict(rows=len(rows), by_tier=t, by_basis=b,
                verification=t["bookkeeping"] + t["closed_form"],
                verification_frac=round((t["bookkeeping"] + t["closed_form"]) / n, 3),
                regression_pins=b["reference"],
                observed_in_cells=b["literature"],
                observed_frac=round(b["literature"] / n, 3))


def write_table(gid, cfg, rows, meta):
    out = os.path.join(LOG, gid)
    os.makedirs(out, exist_ok=True)
    doc = dict(gate=gid, question=" ".join(str(cfg["_gate"].get("question", "")).split()),
               spec=os.path.relpath(meta["spec"], ROOT), header=meta,
               split=tier_split(rows), measures=rows)
    json.dump(doc, open(os.path.join(out, "gate_table.json"), "w"), indent=1, default=str)

    MARK = {"PASS": "pass", "FAIL": "**FAIL**", "BLOCKED": "blocked",
            "KNOWN_RED": "known-red", "TURNED_GREEN": "**turned green**",
            "INFRA_FAIL": "**infra**"}
    L = [f"# gate {gid}", "", doc["question"], "",
         f"spec `{doc['spec']}`  ·  {meta.get('wall_s', 0):.0f} s  ·  "
         f"commit `{meta.get('commit','?')}`  ·  {meta.get('device','?')}", "",
         "| row | tier | basis | unit | declared | measured | |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        k, a = next(iter(r["assertion"].items()))
        v = r.get("value")
        v = "--" if v is None else (json.dumps(v, default=str) if isinstance(v, dict)
                                    else (f"{v:.6g}" if isinstance(v, float) else str(v)))
        L.append(f"| `{r['name']}` | {r['tier']} | {r['basis']} | {r['unit']} | "
                 f"{k} {a} | {v} | {MARK.get(r['outcome'], r['outcome'])} |")
    s = doc["split"]
    L += ["", f"**{s['verification']} of {s['rows']} rows are verification** "
              f"(bookkeeping or closed form); {s['regression_pins']} pin a regression against a "
              f"previous run of this same model; **{s['observed_in_cells']} compare with something "
              f"observed in cells** ({100 * s['observed_frac']:.0f}%)."]
    open(os.path.join(out, "gate_table.md"), "w").write("\n".join(L) + "\n")
    return doc


def rollup():
    """`log/gates/gate_table.json` -- what the promotion note reads."""
    all_docs = []
    for p in sorted(glob.glob(os.path.join(LOG, "*", "gate_table.json"))):
        all_docs.append(json.load(open(p)))
    every = [r for d in all_docs for r in d["measures"]]
    doc = dict(gates=[d["gate"] for d in all_docs], split=tier_split(every) if every else {},
               outcomes={o: sum(1 for r in every if r["outcome"] == o)
                         for o in sorted({r["outcome"] for r in every})} if every else {},
               tables=all_docs)
    json.dump(doc, open(os.path.join(LOG, "gate_table.json"), "w"), indent=1, default=str)
    return doc


# ---------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="append", default=None, help="gate id; repeatable")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--measure-only", action="store_true", help="grade what is already on disk")
    ap.add_argument("--freeze-reference", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cluster", action="store_true",
                    help="submit to gpu_l4 instead of running here, then exit; grade later with "
                         "--measure-only")
    a = ap.parse_args()

    G = gates()
    if a.list or not (a.gate or a.all):
        print(f"  {len(G)} gate(s) in config/gates/")
        for gid, (p, c) in sorted(G.items()):
            m = c["_gate"]["measures"]
            s = tier_split([dict(tier=x["tier"], basis=x["basis"]) for x in m])
            print(f"    {gid:18s} {c['general']['n_frames']:>5} frames  {len(m):>2} rows  "
                  f"({s['by_tier']['bookkeeping']}b/{s['by_tier']['closed_form']}c/"
                  f"{s['by_tier']['measurement']}m, {s['observed_in_cells']} vs cells)  "
                  f"{os.path.relpath(p, ROOT)}")
        return 0

    chosen = sorted(G) if a.all else [g for g in a.gate if g in G]
    missing = [] if a.all else [g for g in a.gate if g not in G]
    if missing:
        print(f"  unknown gate(s): {missing}; have {sorted(G)}")
        return 3

    worst = 0
    for gid in chosen:
        path, cfg = G[gid]
        print(f"\n=== gate {gid}  ({os.path.relpath(path, ROOT)})", flush=True)
        bad = preflight(path, cfg)
        if bad:
            print("  PREFLIGHT FAILED -- no GPU spent:")
            for b in bad:
                print("   -", b)
            worst = max(worst, 3)
            continue
        print(f"  preflight ok: {len(cfg['_gate']['measures'])} rows", flush=True)

        if a.cluster:
            jn = submit_cluster(path, cfg, force=a.force)
            print(f"  submitted {jn} to the cluster; grade it with --measure-only when it lands")
            continue

        wall = 0.0
        if not a.measure_only:
            rc, wall = run_one(path, cfg, device=a.device, force=a.force)
            if rc:
                print(f"  the run exited {rc} -- see log/gates/{gid}/run.log")
                worst = max(worst, 3)
                continue
            print(f"  ran in {wall:.0f} s", flush=True)

        d = data_dir_of(cfg["general"]["name"])
        if not os.path.isdir(d):
            print(f"  no trajectory at {d}")
            worst = max(worst, 3)
            continue
        meta = dict(spec=path, spec_sha1=_sha1(cfg), gate_sha1=_sha1(cfg["_gate"]),
                    commit=subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
                    device=a.device, strict_determinism=True, wall_s=round(wall, 1),
                    trajectory=os.path.relpath(d, ROOT))
        rows, n_fail, n_block, n_turned = grade(path, cfg, d)
        doc = write_table(gid, cfg, rows, meta)
        for r in rows:
            k, arg = next(iter(r["assertion"].items()))
            v = r.get("value")
            vs = "--" if v is None else (json.dumps(v, default=str)[:44] if isinstance(v, dict)
                                         else (f"{v:.6g}" if isinstance(v, float) else str(v)))
            print(f"    {r['outcome']:12s} {r['name']:26s} {r['tier'][:4]}/{r['basis'][:4]:9s} "
                  f"{k} {arg}  ->  {vs}")
        s = doc["split"]
        print(f"  {len(rows)} rows: {s['verification']} verification, "
              f"{s['regression_pins']} regression pins, {s['observed_in_cells']} against cells "
              f"| {n_fail} failed, {n_block} blocked")
        if a.freeze_reference:
            rd = os.path.join(LOG, gid, "reference")
            os.makedirs(rd, exist_ok=True)
            json.dump(dict(gate_sha1=meta["gate_sha1"], commit=meta["commit"],
                           device=a.device, rows=rows), open(os.path.join(rd, "reference.json"), "w"),
                      indent=1, default=str)
            print(f"  froze the reference: gate_sha1 {meta['gate_sha1']}")
        worst = max(worst, 4 if n_turned else (1 if n_fail else (2 if n_block else 0)))

    r = rollup()
    if r.get("split"):
        s = r["split"]
        print(f"\n  ROLL-UP over {len(r['gates'])} gate(s), {s['rows']} rows: "
              f"{s['verification']} verification ({100 * s['verification_frac']:.0f}%), "
              f"{s['regression_pins']} regression pins, "
              f"{s['observed_in_cells']} against an observation ({100 * s['observed_frac']:.0f}%)")
        print(f"  outcomes: {r['outcomes']}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
