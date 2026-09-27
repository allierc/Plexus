#!/usr/bin/env python
"""One markdown per experiment; this launches it, polls it and reports it.

    PYTHONPATH=src python tools/exp.py launch  <n> [--arm A] [--where AXIS=V,V] [--dry-run]
    PYTHONPATH=src python tools/exp.py poll    [n...]
    PYTHONPATH=src python tools/exp.py report  [n...]

Read `experiments/INSTRUCTION.md` first. The design is connectome-gnn's
`tools/exp.py`, decision for decision, because it is already right about the things
that are easy to get wrong: the experiment NUMBER is the only handle, run names are
DERIVED from a pattern and a grid point so the prose and the disk cannot drift, the
STATUS block belongs to the tool, and nothing in a report is typed.

WHAT IS PLEXUS-SPECIFIC, and why this is not a copy:

  TWO VERBS. An arm is `kind: generate` -- a forward spec through
  `Plexus_Main.py -o generate`, landing in `graphs_data/<group>/<name>/` -- or
  `kind: fit`, a run spec through `plexus.tasks.spec_trainer`/`.trainer`, landing
  in `log/task/<run>/`. They are launched differently and polled from different
  trees, so `kind` is read before anything else.

  LOCAL IS A REAL OPTION. The devcontainer has two A6000s, so "submit it" is a
  question. `where: local` runs detached here and `where: l4` submits to the
  queue; an experiment can put both on one axis and MEASURE the difference rather
  than assume it. That axis is the reason this tool exists rather than two
  wrappers around the two `submit_*.py`.

  THE SPEC IS GENERATED, NOT STAGED. connectome-gnn copies a spec into the shared
  config directory under its own name. Here a grid point needs its OWN output
  directory, and for both verbs the output directory is named by the spec, so the
  tool writes one spec per grid point into `experiments/specs/expNN/` with the
  derived run name inside it. `differs_by` is applied there, as dotted keys, and
  the diff is asserted rather than eyeballed.
"""
from __future__ import annotations

import argparse
import copy
import glob
import itertools
import json
import os
import re
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "discovery_okuda"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import cluster as C                                                   # noqa: E402

EXP = os.path.join(ROOT, "experiments")
PY = "/workspace/.conda_envs/neural-graph-linux/bin/python"
STATUS_OPEN, STATUS_CLOSE = "<!-- STATUS -->", "<!-- /STATUS -->"


# --------------------------------------------------------------------------- #
#  the markdown
# --------------------------------------------------------------------------- #
def exp_path(number):
    hits = sorted(glob.glob(os.path.join(EXP, f"exp{int(number):02d}_*.md")))
    if not hits:
        raise SystemExit(f"no experiment {number} in {EXP}")
    return hits[0]


def all_experiments():
    return sorted(glob.glob(os.path.join(EXP, "exp[0-9][0-9]_*.md")))


def load(path):
    """(front matter, body). The front matter is the tool's; the body is the reader's."""
    text = open(path).read()
    if not text.startswith("---"):
        raise SystemExit(f"{path}: no YAML front matter")
    end = text.index("\n---", 3)
    return yaml.safe_load(text[3:end]), text[end + 4:]


_BY_OP = re.compile(r"^(\w+)\[([\w:@#]+)\]$")


def _by_op(lst, name, key):
    """The one element of a list of operator dicts whose `op` is `name` -- or, as `op@set`, whose `op`
    is `op` and `at` is `set`, for a spec that runs one operator on two sets (exp 15 rig 4's
    `cell_chem_react@cell` and `cell_chem_react@host`) -- or, as `op@set#k`, the k-th (from 0) such
    operator on that set, for a spec that runs one operator twice on one set (the metabolite medium's
    two `project`s)."""
    name, _, k = name.partition("#")
    name, _, at = name.partition("@")
    hit = [o for o in lst if isinstance(o, dict) and o.get("op") == name and (not at or o.get("at") == at)]
    if k:
        hit = hit[int(k):int(k) + 1]
    if len(hit) != 1:
        raise SystemExit(f"differs_by: {key!r} -- {len(hit)} operators named {name!r}, need exactly 1")
    return hit[0]


def _set_dotted(d, key, value):
    """`training.epochs: 20` -> d["training"]["epochs"] = 20, creating nothing.

    `operators[cell_cycle].model: adder` addresses the ONE operator of that name in a list, so an
    arm can patch the rule it tests without an index that breaks when the schedule changes. The
    last part may create a key on an operator (a parameter the base left at its default); every
    section on the way must already exist.
    """
    parts = key.split(".")
    for p in parts[:-1]:
        m = _BY_OP.match(p)
        if m:
            if m.group(1) not in d:
                raise SystemExit(f"differs_by: {key!r} -- the spec has no section {m.group(1)!r}")
            d = _by_op(d[m.group(1)], m.group(2), key)
            continue
        if p not in d:
            raise SystemExit(f"differs_by: {key!r} -- the spec has no section {p!r}")
        d = d[p]
    d[parts[-1]] = value


def grid(fm, where=None):
    """Every point of `axes:` as a dict, filtered by `--where AXIS=V1,V2`."""
    ax = {k: list(v) for k, v in (fm.get("axes") or {}).items()}
    for k, vals in (where or {}).items():
        if k not in ax:
            raise SystemExit(f"--where {k}=... : no such axis (have {sorted(ax)})")
        ax[k] = [v for v in ax[k] if str(v) in {str(x) for x in vals}]   # `--where seed=3,4` on int seeds
    keys = list(ax)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(ax[k] for k in keys))]


def runs(fm, where=None, only_arm=None):
    """[(arm, point, run_name), ...] -- the derivation the whole tool rests on."""
    out = []
    for arm in fm["arms"]:
        if only_arm and arm["id"] != only_arm:
            continue
        pts = grid(fm, where)
        for k, vals in (arm.get("axes_only") or {}).items():
            pts = [p for p in pts if p.get(k) in vals]
        for p in pts:
            out.append((arm, p, arm["run_pattern"].format(**p)))
    return out


# --------------------------------------------------------------------------- #
#  writing one grid point's spec
# --------------------------------------------------------------------------- #
def write_spec(fm, arm, point, run_name, number):
    """One spec per grid point, with the derived name inside it and `differs_by` applied.

    THE DIFF IS ASSERTED, NOT EYEBALLED. A spec that quietly picked up a second
    change is an experiment measuring two things and attributing them to one.
    """
    d = os.path.join(EXP, "specs", f"exp{int(number):02d}")
    os.makedirs(d, exist_ok=True)
    kind = arm["kind"]
    src = (os.path.join(ROOT, "config", "run", f"{arm['spec']}.yaml") if kind == "fit"
           else os.path.join(ROOT, "config", f"{arm['spec']}.yaml"))
    if not os.path.isfile(src):
        raise SystemExit(f"arm {arm['id']}: {src} does not exist")
    base = yaml.safe_load(open(src))
    spec = copy.deepcopy(base)
    if kind == "fit":
        spec["name"] = run_name
    else:
        spec["general"]["name"] = run_name
    for k, v in (arm.get("differs_by") or {}).items():
        # A VALUE MAY NAME THE GRID POINT: `general.seed: "{seed}"` takes the point's own seed, so a
        # repeat axis can differ in its random seed without one arm per repeat. An all-digit result
        # is written as an integer.
        if isinstance(v, str) and "{" in v:
            v = v.format(**point)
            v = int(v) if v.lstrip("-").isdigit() else v
        _set_dotted(spec, k, v)

    changed = _diff_keys(base, spec)
    want = set(arm.get("differs_by") or {}) | {"name" if kind == "fit" else "general.name"}
    if changed != want:
        raise SystemExit(f"arm {arm['id']} {run_name}: the spec differs in {sorted(changed)} "
                         f"but the arm declares {sorted(want)}")
    out = os.path.join(d, f"{run_name}.yaml")
    yaml.safe_dump(spec, open(out, "w"), sort_keys=False, default_flow_style=None, width=100)
    return out


def _op_labels(lst):
    """Each operator's `differs_by` address: its `op`; `op@at` when the list holds that `op` twice;
    `op@at#k` when it holds it twice on the same set (k counts from 0 in list order)."""
    n, na, seen, out = {}, {}, {}, []
    for o in lst:
        n[o["op"]] = n.get(o["op"], 0) + 1
        na[(o["op"], o.get("at"))] = na.get((o["op"], o.get("at")), 0) + 1
    for o in lst:
        key = (o["op"], o.get("at"))
        if n[o["op"]] == 1:
            out.append(o["op"])
        elif na[key] == 1:
            out.append(f"{o['op']}@{o.get('at')}")
        else:
            out.append(f"{o['op']}@{o.get('at')}#{seen.get(key, 0)}")
            seen[key] = seen.get(key, 0) + 1
    return out


def _diff_keys(a, b, prefix=""):
    """Dotted keys whose value differs between two nested dicts.

    A list of operator dicts with unique `op` names (or unique `op@at`) is compared per operator,
    reported as `operators[<op>].<key>` (`operators[<op>@<set>].<key>`) -- the same address `differs_by` uses -- so a one-parameter patch on
    one operator reads as exactly that and not as "the whole list changed".
    """
    out = set()
    for k in set(a) | set(b):
        p = f"{prefix}{k}"
        va, vb = a.get(k, object()), b.get(k, object())
        if isinstance(va, dict) and isinstance(vb, dict):
            out |= _diff_keys(va, vb, p + ".")
        elif (isinstance(va, list) and isinstance(vb, list) and len(va) == len(vb)
              and all(isinstance(o, dict) and "op" in o for o in va + vb)
              and [o["op"] for o in va] == [o["op"] for o in vb]
              and len(set(_op_labels(va))) == len(va) and _op_labels(va) == _op_labels(vb)):
            for oa, ob, lab in zip(va, vb, _op_labels(va)):
                out |= _diff_keys(oa, ob, f"{p}[{lab}].")
        elif va != vb:
            out.add(p)
    return out


# --------------------------------------------------------------------------- #
#  where a run's output lives, per verb
# --------------------------------------------------------------------------- #
def out_dir(arm, run_name):
    from plexus.paths import graphs_data_path
    if arm["kind"] == "fit":
        return os.path.join(graphs_data_path().replace("graphs_data", "log"), "task", run_name)
    group = arm["spec"].split("/")[0]
    return graphs_data_path(group, run_name)


def job_dir(number, run_name):
    return os.path.join(ROOT, "log", "experiments", f"exp{int(number):02d}", run_name)


# --------------------------------------------------------------------------- #
#  launch
# --------------------------------------------------------------------------- #
def launch(number, dry_run=False, only_arm=None, where=None):
    path = exp_path(number)
    fm, body = load(path)
    if fm.get("mode") == "steps":
        raise SystemExit(f"experiment {number} runs in steps: tools/exp_step.py run {number} <step.yaml>")
    rs = runs(fm, where, only_arm)
    print(f"[launch] experiment {number}: {len(rs)} run(s)")
    ids = dict(fm.get("job_ids") or {})
    for arm, point, run_name in rs:
        spec = write_spec(fm, arm, point, run_name, number)
        jd = job_dir(number, run_name)
        os.makedirs(jd, exist_ok=True)
        cmd = _command(fm, arm, point, run_name, spec, jd)
        if dry_run:
            print(f"  [dry] {run_name:34s} {cmd[:150]}")
            continue
        jid = _submit(point.get("where", "l4"), cmd, jd, run_name)
        if jid:
            ids[run_name] = jid
        print(f"  {run_name:34s} {jid or 'FAILED'}")
    if not dry_run:
        fm["job_ids"] = ids
        _rewrite(path, fm, body)
    return rs


def _command(fm, arm, point, run_name, spec, jd):
    """The one line the job runs. RELATIVE after a `cd`, for the cluster's sake."""
    rel = os.path.relpath(spec, ROOT)
    if arm["kind"] == "generate":
        group = arm["spec"].split("/")[0]
        # `Plexus_Main` resolves a spec by <group>/<name> under config/, so the generated
        # spec is copied there under its derived name -- the group keeps `add_pre_folder`
        # able to classify it, which a bare name would not be.
        dst = os.path.join(ROOT, "config", group, f"{run_name}.yaml")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.abspath(dst) != os.path.abspath(spec):
            open(dst, "w").write(open(spec).read())
        return f"python -u Plexus_Main.py -o generate {group}/{run_name} --device cuda:0 --force"
    mod = ("plexus.tasks.spec_trainer" if "spec" in yaml.safe_load(open(spec))
           else "plexus.tasks.trainer")
    ph = "train test analyse" if mod.endswith("spec_trainer") else "train test plot"
    return f"python -u -m {mod} --device cuda:0 -o {ph} {rel}"


def _submit(where, cmd, jd, run_name, wall=240, job_prefix="exp_"):
    sh = os.path.join(jd, "run.sh")
    with open(sh, "w") as f:
        f.write("\n".join([
            "#!/bin/bash -l",
            f"cd {C.cpath(ROOT) if where == 'l4' else ROOT}",
            f"export PYTHONPATH={C.cpath(os.path.join(ROOT,'src')) if where=='l4' else os.path.join(ROOT,'src')}",
            "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
            "export MPLBACKEND=Agg",
            (f"conda run -n {C.ENV} {cmd}" if where == "l4" else f"{PY} {cmd.split(' ',1)[1]}"),
        ]) + "\n")
    os.chmod(sh, 0o755)
    o = os.path.join(jd, "cluster.out")
    if where == "local":
        # DETACHED, so a local arm does not hold the session for its whole run. The wall
        # clock is read back from the output tree exactly as a cluster arm's is.
        with open(o, "w") as f:
            subprocess.Popen(["bash", "-l", sh], stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True, cwd=ROOT)
        return f"local:{run_name}"
    gpu = "-gpu num=1 " if C.GPU != "0" else ""
    excl = "".join(f'-R "hname!={h}" ' for h in C.EXCLUDE_HOSTS if h)
    bs = (f"bsub -n {C.NCPUS} {gpu}{excl}-q {C.QUEUE} -W {int(wall)} -J {job_prefix}{run_name} "
          f"-o {C.cpath(o)} -e {C.cpath(o[:-4])}.err bash -l {C.cpath(sh)}")
    r = C._ssh(bs, timeout=60)
    if r is None or r.returncode != 0:
        print(f"    SUBMIT FAILED: {'ssh timeout' if r is None else (r.stderr or r.stdout)[:120]}")
        return None
    m = re.search(r"Job <(\d+)>", r.stdout or "")
    return m.group(1) if m else (r.stdout or "").strip()[:40]


# --------------------------------------------------------------------------- #
#  poll -- every number below is READ, never typed
# --------------------------------------------------------------------------- #
def measure(arm, run_name, number):
    """What this run cost and what it produced. `{}` when it has not landed."""
    import numpy as np
    d, jd = out_dir(arm, run_name), job_dir(number, run_name)
    m = {}
    co = os.path.join(jd, "cluster.out")
    if os.path.isfile(co):
        txt = open(co, errors="ignore").read()
        w = re.search(r"Run time :\s+(\d+) sec", txt)          # LSF's own, for a queued job
        if w:
            m["wall_s"] = int(w.group(1))
        if "TERM_" in txt:
            m["died"] = re.search(r"(TERM_\w+)", txt).group(1)
    if arm["kind"] == "generate":
        tj = os.path.join(d, "trajectory.npz")
        if os.path.isfile(tj):
            z = np.load(tj)
            if "frame_ms" in z.files:
                fm_ = np.asarray(z["frame_ms"], float)
                m["ms_per_frame"] = round(float(np.median(fm_)), 3)
                m["frames"] = int(fm_.size)
                m["wall_s"] = m.get("wall_s", round(float(fm_.sum()) / 1000.0, 1))
            for k in z.files:
                if k.endswith("__pos"):
                    m["fingerprint"] = round(float(np.asarray(z[k], float).std()), 9)
                    break
            m["landed"] = True
    else:
        rp = os.path.join(d, "results", "report.json")
        if os.path.isfile(rp):
            r = json.load(open(rp))
            m["epochs"] = r.get("epochs")
            m["wall_s"] = m.get("wall_s", r.get("seconds"))
            m["best_val"] = r.get("best_val_mse")
            if m.get("epochs"):
                m["s_per_epoch"] = round(float(r.get("seconds", 0)) / max(r["epochs"], 1), 2)
            m["landed"] = True
        tj = glob.glob(os.path.join(d, "results", "*_test.json"))
        if tj:
            m["test"] = json.load(open(tj[0])).get("normalised_mse_settled")
    if not m.get("landed") and os.path.isdir(d):
        m["state"] = "running"
    return m


def _mean_sd(vals, nd=4):
    v = [float(x) for x in vals if x is not None]
    if not v:
        return ""
    if len(v) == 1:
        return f"{v[0]:.{nd}g}"
    import statistics as st
    return f"{st.mean(v):.{nd}g} ± {st.pstdev(v):.2g}"


def status_block(fm, number):
    """The tool-owned block: LANDED summary first, then per run.

    THE SUMMARY IS GROUPED OVER THE LAST AXIS, which is the repeat axis by convention (`repeat`,
    `fold`, `seed`) -- the one a mean is taken across. A flat per-run list answers "did it run";
    only the summary answers the question the experiment was asked, which is a comparison between
    groups at a stated spread. This is connectome-gnn's layout: landed summary, then per run.
    """
    axes = list(fm.get("axes") or {})
    rep = axes[-1] if axes else None
    groups, rows, landed = {}, [], 0
    for arm, point, run_name in runs(fm):
        m = measure(arm, run_name, number)
        landed += bool(m.get("landed"))
        key = (arm["id"],) + tuple(point[a] for a in axes if a != rep)
        groups.setdefault(key, []).append(m)
        cost = (f"{m['ms_per_frame']} ms/frame" if "ms_per_frame" in m else
                f"{m['s_per_epoch']} s/epoch" if "s_per_epoch" in m else "")
        res = (f"{m['fingerprint']:.10g}" if "fingerprint" in m else
               f"{m['best_val']:.10g}" if m.get("best_val") is not None else "")
        state = m.get("died") or ("landed" if m.get("landed") else m.get("state", "queued"))
        rows.append([f"`{run_name}`", state, str(m.get("wall_s", "")), cost, res])

    gh = ["arm"] + [a for a in axes if a != rep] + [
        "n", "cost", "result", "result spread across repeats"]
    g = ["| " + " | ".join(gh) + " |", "|" + "---|" * len(gh)]
    for key, ms in groups.items():
        ms = [m for m in ms if m.get("landed")]
        if not ms:
            continue
        cost = (_mean_sd([m.get("ms_per_frame") for m in ms]) + " ms/frame"
                if "ms_per_frame" in ms[0] else
                _mean_sd([m.get("s_per_epoch") for m in ms]) + " s/epoch")
        res = [m.get("fingerprint", m.get("best_val")) for m in ms]
        res = [r for r in res if r is not None]
        spread = (f"{(max(res) - min(res)) / max(abs(sum(res) / len(res)), 1e-30):.2e} relative"
                  if len(res) > 1 else "")
        g.append("| " + " | ".join(list(key) + [str(len(ms)), cost,
                                                  f"{sum(res)/len(res):.10g}" if res else "",
                                                  spread]) + " |")
    ph = ["run", "state", "wall s", "cost", "result"]
    per = ["| " + " | ".join(ph) + " |", "|" + "---|" * len(ph)]
    per += ["| " + " | ".join(r) + " |" for r in rows]
    block = (f"{STATUS_OPEN}\n\n## Status\n\n**{landed}/{len(rows)} landed** -- polled "
             f"{time.strftime('%Y-%m-%d %H:%M')}\n\n"
             f"### Landed -- grouped over `{rep}`\n\n" + "\n".join(g) +
             "\n\n### Per run\n\n" + "\n".join(per) + f"\n\n{STATUS_CLOSE}")
    return block, landed, len(rows)


def poll(number):
    path = exp_path(number)
    fm, body = load(path)
    block, landed, total = status_block(fm, number)
    if STATUS_OPEN in body:
        body = re.sub(re.escape(STATUS_OPEN) + r".*?" + re.escape(STATUS_CLOSE),
                      lambda _: block, body, flags=re.S)
    else:
        body = body.rstrip() + "\n\n## Status\n\n" + block + "\n"
    _rewrite(path, fm, body)
    print(f"[poll] experiment {number}: {landed}/{total} landed")
    return landed, total


def _rewrite(path, fm, body):
    with open(path, "w") as f:
        f.write("---\n" + yaml.safe_dump(fm, sort_keys=False, default_flow_style=None,
                                         width=100) + "---" + body)


def write_index():
    lines = ["# Experiments", "",
             "| # | name | purpose | progress | file |", "|---|---|---|---|---|"]
    for p in all_experiments():
        fm, body = load(p)
        if fm.get("branch") == "simulation":
            # A SEARCH HAS NO GRID TO COUNT. Its progress is the number of iterations in its
            # results table and the verdict of the last one, read from the table itself.
            rows = [l for l in body.splitlines()
                    if re.match(r"\|\s*[A-Z]?\d+[a-z]?\s*\|", l)]   # `7a`, `7b`: parallel variants; `H1`: an element test
            # THE VERDICT COLUMN IS FOUND BY ITS HEADER, not by position: experiments add ruler
            # columns (exp 2 carries five), and a fixed index read a jitter value as the verdict.
            head = next((l for l in body.splitlines()
                         if l.startswith("|") and "verdict" in l.lower()), "")
            cols = [c.strip().lower() for c in head.split("|")]
            vi = cols.index("verdict") if "verdict" in cols else 5
            last = rows[-1].split("|")[vi].strip() if rows else "not started"
            prog = f"{len(rows)} iteration(s), last: {last}"
        else:
            _, landed, total = status_block(fm, fm["number"])
            prog = f"{landed}/{total}"
        lines.append(f"| {fm['number']} | {fm['name']} | {fm.get('purpose','')} | "
                     f"{prog} | [`{os.path.basename(p)}`]({os.path.basename(p)}) |")
    open(os.path.join(EXP, "INDEX.md"), "w").write("\n".join(lines) + "\n")
    print(f"[index] {os.path.join(EXP, 'INDEX.md')}")


def report(numbers=None):
    """`experiments/report.md` -- one section per experiment, every cell derived."""
    out = ["# Experiment report", "",
           f"*Generated {time.strftime('%Y-%m-%d %H:%M')} by `tools/exp.py report`. "
           f"Nothing here is typed.*", ""]
    for p in all_experiments():
        fm, _ = load(p)
        if numbers and fm["number"] not in numbers:
            continue
        block, landed, total = status_block(fm, fm["number"])
        out += [f"## {fm['number']}. {fm.get('title', fm['name'])}", "",
                f"**{fm.get('purpose','')}**", "", f"{landed}/{total} landed.", "",
                block.replace(STATUS_OPEN, "").replace(STATUS_CLOSE, "").strip(), ""]
    open(os.path.join(EXP, "report.md"), "w").write("\n".join(out) + "\n")
    print(f"[report] {os.path.join(EXP, 'report.md')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    l = sub.add_parser("launch"); l.add_argument("number", type=int)
    l.add_argument("--arm"); l.add_argument("--where"); l.add_argument("--dry-run", action="store_true")
    pl = sub.add_parser("poll"); pl.add_argument("numbers", nargs="*", type=int)
    rp = sub.add_parser("report"); rp.add_argument("numbers", nargs="*", type=int)
    a = ap.parse_args()
    if a.verb == "launch":
        w = None
        if a.where:
            k, v = a.where.split("=", 1)
            w = {k: v.split(",")}
        launch(a.number, a.dry_run, a.arm, w)
    elif a.verb == "poll":
        ns = a.numbers or [load(p)[0]["number"] for p in all_experiments()
                           if load(p)[0].get("branch") != "simulation"]
        for n in ns:
            poll(n)
        write_index()
    else:
        report(a.numbers or None)


if __name__ == "__main__":
    main()
