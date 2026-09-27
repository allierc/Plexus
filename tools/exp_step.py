#!/usr/bin/env python
"""A STEP: one change at the coarse level, tested as >= 4 competing candidates of a small YAML sweep.

    PYTHONPATH=src:tools python tools/exp_step.py check NN step.yaml      validate + list the jobs, submit nothing
    PYTHONPATH=src:tools python tools/exp_step.py run   NN step.yaml      submit, wait, score, record (ONE background call)
    PYTHONPATH=src:tools python tools/exp_step.py status NN               the steps so far, the champion

WHY (2026-09-27). Measured on exp06-15: 79 % of landings left the card unchanged and ~60 % of runs changed one
number of an earlier run by x2-x4; the human-built rotor (builder/exp_02_bacterium) moved ~7 keys per run, x7,
a quarter of them structural. A step makes the big move structural: the tool, not the session, decides what
counts.

TWO LEVELS. COARSE = the spec with every NUMBER removed (sets, operators and their implementation/model, the
keys present, every string and boolean). FINE = the numbers. Two specs are coarse-equal iff they differ only
in numbers. A step is accepted only if:
  - it has >= 4 candidates (the product of the `sweep:` lists; the swept values must be numbers);
  - its coarse form differs from the previous step's winner (a STRUCTURAL step) -- or, if coarse-equal, one
    swept value is >= 10x away from the previous winner's (a REGIME step, which then COUNTS only if the
    winner's outcome differs from the previous winner's: audit band, or a count / extent ratio by > 25 %);
  - a `build:` block, if any, names a registered operator variant and its tests pass.
Each candidate runs every arm (the experiment's controls, `differs_by` from the candidate) and gets the FULL
gate card (tools/exp_gate_score.py's score on those runs). The best card wins (a tie: the central sweep
value); the winner is the parent of the next step. The champion is the best card over all steps: it never
goes down. `kind: confirm` re-runs one candidate on >= 3 seeds; it needs no coarse change.

THE STEP FILE:
    intent: hollow bud -> solid cluster with leader cells        # the coarse change, one sentence
    base: tissue/exp17_solid                                     # a spec under config/; default: the last winner
    sweep: {"operators[protrusion].force": [0.3, 1, 3, 10]}      # 1-2 numeric keys, >= 4 candidates
    arms: {...}                                                  # optional: override the md's arms from here on
    seeds: [1]                                                   # optional; kind: confirm needs >= 3
    build: {operator: "cell_mechanics[apicobasal_contact]", law: "...", source: "...", tests: [tests/x.py]}
    answer: "..."           # required when the md declares `question:` -- how the phenomenon emerges in the last
                            # winner and how that differs from the paper's mechanism; the intent follows from it

The experiment's md front matter carries: mode: steps, gates (a path), start (the first parent, a
graphs_data run whose spec.yaml is the start), main_arm, step_arms ({id: {differs_by: {...}} or {same_as: id}}).
In `differs_by`, the value "__delete__" removes the key. Every leaf key named `seed` takes the seed.
"""
from __future__ import annotations

import argparse
import copy
import glob
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp  # noqa: E402

MIN_CANDIDATES = 4
REGIME_JUMP = 10.0            # a coarse-equal step must move a swept value >= 10x from the last winner's
OUTCOME_CHANGE = 0.25         # ... and then change a count or extent ratio by > 25 % (or the audit band)
NEXT_STEP_MIN = 10
DELETE = "__delete__"


# ------------------------------------------------------------------------------------------ coarse level
def _flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{p}{k}."))
    elif isinstance(d, list) and d and all(isinstance(o, dict) and "op" in o for o in d):
        for lab, o in zip(exp._op_labels(d), d):
            out.update(_flat(o, f"{p[:-1]}[{lab}]."))         # `operators[cell_grow].gain`, the differs_by address
    else:
        out[p[:-1]] = d
    return out


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _coarse_value(v):
    if _is_num(v):
        return "#"
    if isinstance(v, list) and v and all(_is_num(x) or (isinstance(x, list) and all(_is_num(y) for y in x)) for x in v):
        return "#list"
    return v


def coarse(spec: dict) -> dict:
    """The spec with every number removed: keys kept, strings / booleans / structure kept. Names, seeds and
    the movie's look are not part of the model."""
    return {k: _coarse_value(v) for k, v in _flat(spec).items()
            if not (k == "general.name" or k.endswith(".seed") or k == "general.seed"
                    or k.startswith("plotting.") or k.startswith("describe"))}


def coarse_diff(a: dict, b: dict, limit=14) -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in a:
            out.append(f"+ {k}")
        elif k not in b:
            out.append(f"- {k}")
        elif a[k] != b[k]:
            out.append(f"~ {k}: {a[k]} -> {b[k]}")
    return out[:limit] + ([f"... {len(out) - limit} more"] if len(out) > limit else [])


# ------------------------------------------------------------------------------------------ specs
def _set(spec, key, value):
    if value == DELETE:
        parts = key.split(".")
        d = spec
        for p in parts[:-1]:
            m = exp._BY_OP.match(p)
            d = exp._by_op(d[m.group(1)], m.group(2), key) if m else d[p]
        d.pop(parts[-1], None)
    else:
        exp._set_dotted(spec, key, value)


def _seed(spec, s):
    def walk(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "seed" and _is_num(v):
                    d[k] = int(s)
                else:
                    walk(v)
        elif isinstance(d, list):
            for x in d:
                walk(x)
    walk(spec)
    return spec


def exp_md(n):
    p = exp.exp_path(n)
    fm, _ = exp.load(p)
    if fm.get("mode") != "steps":
        raise SystemExit(f"{p}: not a steps experiment (front matter `mode: steps`)")
    return p, fm, p[:-3]


def _gd():
    from plexus.paths import graphs_data_path
    return graphs_data_path()


def load_steps(d):
    f = os.path.join(d, "steps.jsonl")
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def _spec_of(ref):
    """A spec from `group/name` under config/, or a graphs_data run's spec.yaml ('run:group/name')."""
    if ref.startswith("run:"):
        return yaml.safe_load(open(os.path.join(_gd(), ref[4:], "spec.yaml")))
    return yaml.safe_load(open(os.path.join(ROOT, "config", f"{ref}.yaml")))


def plan(n, step: dict) -> dict:
    """Validate a step and list its jobs. Raises SystemExit with the reason when the tool refuses it."""
    md, fm, d = exp_md(n)
    steps = [s for s in load_steps(d) if s.get("state") == "landed"]
    k = len(load_steps(d)) + 1
    prev = steps[-1] if steps else None
    parent_ref = (prev or {}).get("winner_spec") or fm["start"]
    base_ref = step.get("base") or parent_ref
    base = _spec_of(base_ref)
    group = base_ref[4:].split("/")[0] if base_ref.startswith("run:") else base_ref.split("/")[0]
    kind = step.get("kind", "step")
    sweep = step.get("sweep") or {}
    seeds = [int(s) for s in (step.get("seeds") or [1])]
    arms = step.get("arms") or (prev or {}).get("arms") or fm["step_arms"]
    main = fm["main_arm"]
    for key, vals in sweep.items():
        if not isinstance(vals, list) or not all(_is_num(v) for v in vals):
            raise SystemExit(f"sweep {key!r}: a list of numbers (a coarse change goes in the base spec, not the sweep)")
        for arm, a in arms.items():
            if key in ((a or {}).get("differs_by") or {}):
                raise SystemExit(f"sweep {key!r} is also set by arm {arm!r}")
    combos = [dict(zip(sweep, vals)) for vals in itertools.product(*sweep.values())] if sweep else [{}]
    if kind == "confirm":
        if len(seeds) < 3:
            raise SystemExit("kind: confirm needs >= 3 seeds")
    elif len(combos) < MIN_CANDIDATES:
        raise SystemExit(f"{len(combos)} candidate(s): a step is >= {MIN_CANDIDATES} competing candidates (sweep)")
    cands = []
    for ci, combo in enumerate(combos, 1):
        s = copy.deepcopy(base)
        for key, v in combo.items():
            _set(s, key, v)
        cands.append(dict(c=ci, values=combo, spec=s))
    parent = _spec_of(parent_ref)
    cf, pf = coarse(cands[0]["spec"]), coarse(parent)
    structural = cf != pf
    jump = 0.0
    if not structural and kind != "confirm":
        pflat = _flat(parent)
        for key, vals in sweep.items():
            p0 = pflat.get(key)
            if _is_num(p0) and p0 != 0:
                jump = max([jump] + [abs(math.log10(abs(v / p0))) for v in vals if v != 0])
        if prev is not None and jump < math.log10(REGIME_JUMP):
            raise SystemExit(f"not a step: the coarse form equals the last winner's and no swept value is "
                             f">= {REGIME_JUMP:g}x from its value (largest {10 ** jump:.2g}x). Change the structure "
                             f"(an operator, a variant, a set, a key) in the base spec, or jump a regime.")
    if step.get("build"):
        _check_build(step["build"])
    answer = _check_answer(fm, step)
    jobs = []
    for c in cands:
        for arm, a in arms.items():
            if (a or {}).get("same_as"):
                continue
            for sd in seeds:
                s = copy.deepcopy(c["spec"])
                for key, v in ((a or {}).get("differs_by") or {}).items():
                    _set(s, key, v)
                _seed(s, sd)
                run = f"exp{n:02d}_s{k:02d}_c{c['c']}_{arm}_s{sd}"
                s["general"]["name"] = run
                jobs.append(dict(c=c["c"], arm=arm, seed=sd, run=run, group=group, spec=s))
    return dict(k=k, kind=kind, intent=step.get("intent", ""), base=base_ref, parent=parent_ref, arms=arms,
                main=main, seeds=seeds, sweep=sweep, cands=cands, jobs=jobs, structural=structural,
                jump=10 ** jump, coarse_diff=coarse_diff(pf, cf), group=group, build=step.get("build"), answer=answer)


MIN_ANSWER_WORDS = 25


def _check_answer(fm, step):
    """THE ONE QUESTION (2026-09-27). Asked "how does the bud emerge here, and how does that differ from Wang's
    mechanism?", exp11's session wrote a five-change structural plan it had never proposed on its own. An
    experiment declares ONE question in its front matter (`question:`, about its phenomenon and the paper's
    mechanism); every step must answer it, in words, before it runs. The answer is the coarse description of
    the model's mechanism; the step's `intent` should be the change it points to. Recorded with the step,
    shown in the report and the watcher."""
    q = fm.get("question")
    if not q:
        return None
    a = str(step.get("answer") or "").strip()
    if len(a.split()) < MIN_ANSWER_WORDS:
        raise SystemExit(f"answer the experiment's question before the step (`answer:`, >= {MIN_ANSWER_WORDS} words):\n  {q}")
    return a


def _check_build(b):
    import plexus.operators  # noqa: F401  registers everything
    from plexus.models.registry import _OP_CONTRACTS
    m = re.match(r"^(\w+)(?:\[([\w:]+)\])?$", str(b.get("operator", "")))
    if not m:
        raise SystemExit(f"build.operator {b.get('operator')!r}: write name or name[variant]")
    name, var = m.group(1), m.group(2)
    if name not in _OP_CONTRACTS:
        raise SystemExit(f"build: operator {name!r} is not registered")
    if var and var not in _OP_CONTRACTS[name].implementations:
        raise SystemExit(f"build: {name} has no variant {var!r} (has {sorted(_OP_CONTRACTS[name].implementations)})")
    for k in ("law", "source"):
        if not b.get(k):
            raise SystemExit(f"build: `{k}:` is required (the equation, and the paper it comes from)")
    tests = b.get("tests") or []
    if not tests:
        raise SystemExit("build: `tests:` is required (at least: the default is the old law, bit for bit)")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=f"{ROOT}/src:{ROOT}/tools"), timeout=900)
    if r.returncode != 0:
        raise SystemExit("build: the tests fail:\n" + (r.stdout + r.stderr)[-1500:])


# ------------------------------------------------------------------------------------------ run
def _submit(n, job):
    dst = os.path.join(ROOT, "config", job["group"], f"{job['run']}.yaml")
    yaml.safe_dump(job["spec"], open(dst, "w"), sort_keys=False, default_flow_style=None, width=100)
    jd = exp.job_dir(n, job["run"])
    os.makedirs(jd, exist_ok=True)
    for f in ("cluster.out", "cluster.err"):
        if os.path.exists(os.path.join(jd, f)):
            os.remove(os.path.join(jd, f))
    cmd = f"python -u Plexus_Main.py -o generate {job['group']}/{job['run']} --device cuda:0 --force"
    return exp._submit("l4", cmd, jd, job["run"], wall=240)


def _state(n, job, t0):
    import exp_wait
    dead = exp_wait._lsf_died(n, job["run"])
    if dead:
        return "died", dead
    d = os.path.join(_gd(), job["group"], job["run"])
    tj, mv = os.path.join(d, "trajectory.npz"), os.path.join(d, "movie.mp4")
    fresh = lambda f: os.path.exists(f) and os.path.getmtime(f) > t0                 # noqa: E731
    if fresh(tj) and fresh(mv):
        return "landed", ""
    co = os.path.join(exp.job_dir(n, job["run"]), "cluster.out")
    if os.path.exists(co) and "Successfully completed" in open(co, errors="ignore").read():
        return ("landed", "no movie") if fresh(tj) else ("died", "completed without a trajectory")
    return "open", ""


def _caption(group, run):
    f = os.path.join(_gd(), "video_descriptions.txt")
    if not os.path.exists(f):
        return ""
    txt = open(f, errors="ignore").read()
    i = txt.rfind(f"video name: {group}/{run}/movie")
    if i < 0:
        return ""
    m = re.search(r"- description:\s*(.*?)(?:\n- |\n\n|\Z)", txt[i:], flags=re.S)
    return (m.group(1).strip() if m else "")[:400]


def _outcome(h):
    """The coarse outcome of a run: audit band, and per set count and extent ratios last/first."""
    o = {"band": (h.get("audit") or {}).get("band")}
    for s, r in (h.get("sets") or {}).items():
        c, e = r.get("count") or [None, None], r.get("extent") or [None, None]
        if c[0]:
            o[f"{s}.count"] = round(c[1] / c[0], 3) if c[1] is not None else None
        if e and e[0]:
            o[f"{s}.extent"] = round(e[1] / e[0], 3) if e[1] is not None else None
    return o


def _outcome_differs(a, b):
    if not a or not b:
        return True
    if a.get("band") != b.get("band"):
        return True
    for k in set(a) & set(b):
        if k != "band" and a[k] and b[k] and abs(a[k] / b[k] - 1) > OUTCOME_CHANGE:
            return True
    return False


def score_candidates(n, P, G):
    """{c: card dict} -- the full gate card of each candidate, its runs standing for the arms."""
    import exp_gate_score as S
    import exp_measures
    measured = {}
    for j in P["jobs"]:
        vals = {}
        for m in G.get("measures") or []:
            if m.get("arms", "all") != "all" and j["arm"] not in m["arms"]:
                continue
            kw = dict(m.get("kw") or {})
            kw.update((m.get("kw_by_arm") or {}).get(j["arm"]) or {})
            try:
                v = exp_measures.run_measure(m["measure"], f"{j['group']}/{j['run']}", **kw)
            except Exception as e:                                                   # noqa: BLE001
                v = {"error": f"{type(e).__name__}: {e}"}
            vals.update({f"{m['measure']}.{k}": x for k, x in (v or {}).items()})
        measured[j["run"]] = vals
    G2 = dict(G, seeds=P["seeds"])
    cards = {}
    for c in P["cands"]:
        M = {}
        for arm, a in P["arms"].items():
            src = (a or {}).get("same_as") or arm
            for sd in P["seeds"]:
                run = f"exp{n:02d}_s{P['k']:02d}_c{c['c']}_{src}_s{sd}"
                if run in measured:
                    M[(arm, sd)] = measured[run]
        cards[c["c"]] = S.score(G2, M)
    return cards


def run_step(n, step, every=60, timeout=4 * 3600):
    import exp_land
    md, fm, d = exp_md(n)
    P = plan(n, step)
    G = yaml.safe_load(open(os.path.join(ROOT, fm["gates"])))
    t0 = time.time()
    rec = dict(step=P["k"], state="running", at=time.strftime("%Y-%m-%d %H:%M"), kind=P["kind"], intent=P["intent"],
               base=P["base"], parent=P["parent"], structural=P["structural"], jump=P["jump"],
               coarse_diff=P["coarse_diff"], sweep=P["sweep"], seeds=P["seeds"], arms=P["arms"], build=P["build"],
               question=exp_md(n)[1].get("question"), answer=P["answer"],
               jobs=[j["run"] for j in P["jobs"]])
    steps = load_steps(d)
    last_land = max((time.mktime(time.strptime(s["landed_at"], "%Y-%m-%d %H:%M")) for s in steps if s.get("landed_at")), default=None)
    if last_land:
        rec["gap_min"] = round((t0 - last_land) / 60, 1)
    _append(d, rec)
    print(f"[step] exp{n:02d} step {P['k']} ({'structural' if P['structural'] else 'regime' if P['kind'] != 'confirm' else 'confirm'}): "
          f"{len(P['cands'])} candidate(s) x {len([a for a in P['arms'].values() if not (a or {}).get('same_as')])} arm(s) "
          f"x {len(P['seeds'])} seed(s) = {len(P['jobs'])} job(s)", flush=True)
    for j in P["jobs"]:
        jid = _submit(n, j)
        print(f"  {j['run']:40s} {jid or 'SUBMIT FAILED'}", flush=True)
    while time.time() - t0 < timeout:
        st = {j["run"]: _state(n, j, t0) for j in P["jobs"]}
        if all(s != "open" for s, _ in st.values()):
            break
        time.sleep(every)
    st = {j["run"]: _state(n, j, t0) for j in P["jobs"]}
    health = {}
    for j in P["jobs"]:
        if st[j["run"]][0] == "landed":
            try:
                health[j["run"]] = exp_land.health(f"{j['group']}/{j['run']}")
            except Exception as e:                                                   # noqa: BLE001
                health[j["run"]] = {"non_finite": [f"health failed: {e}"]}
    cards = score_candidates(n, P, G)
    mid = (len(P["cands"]) + 1) / 2
    best = max(P["cands"], key=lambda c: (round(cards[c["c"]]["total"], 6), -abs(c["c"] - mid)))
    main_run = f"exp{n:02d}_s{P['k']:02d}_c{best['c']}_{P['main']}_s{P['seeds'][0]}"
    out_w = _outcome(health.get(main_run) or {})
    prev = next((s for s in reversed(steps) if s.get("state") == "landed" and s.get("valid")), None)
    counts = (P["kind"] == "confirm" or P["structural"] or prev is None
              or _outcome_differs(out_w, prev.get("winner_outcome")))
    win_ref = f"{P['group']}/exp{n:02d}_s{P['k']:02d}_best"
    yaml.safe_dump(dict(best["spec"], general=dict(best["spec"]["general"], name=f"exp{n:02d}_s{P['k']:02d}_best")),
                   open(os.path.join(ROOT, "config", f"{win_ref}.yaml"), "w"), sort_keys=False, default_flow_style=None, width=100)
    champion = max([s.get("winner_card", -1) for s in steps if s.get("valid")] + [cards[best["c"]]["total"] if counts else -1])
    rec.update(state="landed", landed_at=time.strftime("%Y-%m-%d %H:%M"), valid=bool(counts),
               candidates=[dict(c=c["c"], values=c["values"], card=cards[c["c"]]["total"],
                                passed=cards[c["c"]]["passed"], invalid=cards[c["c"]]["invalid"],
                                rows={r["id"]: r.get("points") for r in cards[c["c"]]["rows"]},
                                caps=cards[c["c"]]["caps"]) for c in P["cands"]],
               states={r: s for r, (s, _) in st.items() if s != "landed"} or None,
               winner=best["c"], winner_card=cards[best["c"]]["total"], winner_spec=win_ref,
               winner_outcome=out_w, champion=champion,
               captions={j["run"]: _caption(j["group"], j["run"]) for j in P["jobs"] if j["arm"] == P["main"]})
    _replace_last(d, rec)
    report = _report(n, rec, health)
    open(os.path.join(d, f"step_{P['k']:02d}.md"), "w").write(report + "\n")
    print(report, flush=True)
    return rec


def _append(d, rec):
    with open(os.path.join(d, "steps.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _replace_last(d, rec):
    f = os.path.join(d, "steps.jsonl")
    lines = open(f).read().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if json.loads(lines[i]).get("step") == rec["step"]:
            lines[i] = json.dumps(rec, default=str)
            break
    open(f, "w").write("\n".join(lines) + "\n")


def _report(n, rec, health):
    import exp_land
    L = [f"**exp{n:02d} step {rec['step']}** -- {rec['intent']}",
         f"{'STRUCTURAL' if rec['structural'] else 'CONFIRM' if rec['kind'] == 'confirm' else 'REGIME (x%.3g)' % rec['jump']}"
         f" step, {'COUNTS' if rec['valid'] else 'VOID: the winner did the same as the last winner (next step must be structural)'}"
         + (f"; {rec['gap_min']} min since the last step landed" if rec.get("gap_min") is not None else ""), ""]
    if rec.get("answer"):
        L += [f"*{rec.get('question')}*", f"{rec['answer']}", ""]
    if rec["coarse_diff"]:
        L += ["Coarse change from the parent:"] + [f"    {x}" for x in rec["coarse_diff"]] + [""]
    L += ["| cand | values | card | passed | gate points |", "|---|---|---|---|---|"]
    for c in rec["candidates"]:
        mark = " **winner**" if c["c"] == rec["winner"] else ""
        L.append(f"| c{c['c']}{mark} | {c['values']} | {c['card']:.2f} | {c['passed']} | "
                 + ", ".join(f"{k} {v}" for k, v in c["rows"].items()) + (f" CAPS {c['caps']}" if c["caps"] else "") + " |")
    L += ["", f"Winner card {rec['winner_card']:.2f}; CHAMPION {rec['champion']:.2f}; next parent `{rec['winner_spec']}`.", "",
          "Health (flags only):"]
    for r, h in health.items():
        fl = [l for l in exp_land.health_lines(h) if l.strip().startswith("- **")]
        if fl:
            L += [f"- `{r}`"] + [f"  {x.strip()}" for x in fl]
    if rec.get("states"):
        L.append(f"NOT LANDED: {rec['states']}")
    L += ["", "Captions of the main arm:"] + [f"- `{r}`: {c or '(no caption yet)'}" for r, c in (rec.get("captions") or {}).items()]
    due = time.strftime("%H:%M", time.localtime(time.time() + NEXT_STEP_MIN * 60))
    L += ["", f"**NEXT STEP DUE by {due}**: a coarse change from `{rec['winner_spec']}` (structure, or a >= 10x regime jump), >= 4 candidates."]
    return "\n".join(L)


def status(n):
    md, fm, d = exp_md(n)
    for s in load_steps(d):
        print(f"step {s['step']:>2} {s.get('state'):8s} {'struct' if s.get('structural') else s.get('kind'):8s} "
              f"{'valid' if s.get('valid') else '-----'} winner c{s.get('winner')} {s.get('winner_card', float('nan')):.2f} "
              f"champion {s.get('champion', float('nan')):.2f}  {s.get('intent', '')[:70]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=["check", "run", "status"])
    ap.add_argument("n", type=int)
    ap.add_argument("step", nargs="?")
    a = ap.parse_args()
    if a.verb == "status":
        status(a.n)
        return 0
    step = yaml.safe_load(open(a.step))
    if a.verb == "check":
        P = plan(a.n, step)
        print(f"step {P['k']}: {'structural' if P['structural'] else 'confirm' if P['kind'] == 'confirm' else 'regime x%.3g' % P['jump']}, "
              f"{len(P['cands'])} candidate(s), {len(P['jobs'])} job(s)")
        for x in P["coarse_diff"]:
            print("   ", x)
        for j in P["jobs"]:
            print(f"  {j['group']}/{j['run']}")
        return 0
    run_step(a.n, step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
