#!/usr/bin/env python
"""Drive experiment 4 (experiments/exp04_membrane_channels.md) one ROUND at a time: three siblings
of one channel, at most three jobs on ${CLUSTER_QUEUE_PREFIX}l4 at once.

    PYTHONPATH=src python tools/channel_round.py launch 1a 1b 1c      # write the specs, submit (<= 3 in flight)
    PYTHONPATH=src python tools/channel_round.py status [1a ...]      # LSF state + landed/not, per run
    PYTHONPATH=src python tools/channel_round.py judge 1a 1b 1c       # caption locally, then ruler + VLM
    PYTHONPATH=src python tools/channel_round.py row 1a --agent "7.25" \
        --verdict "..." --changed "..." --said "..."                  # append the results row (exp 1's layout)

WHAT IS DERIVED AND WHAT IS WRITTEN. The spec name, the channel and every ruler number in a row
are read from the run and from `experiments/specs/exp04/judge.jsonl` (`judge.py --out`); only the
agent's score (the channel judge's report), the verdict and the two prose columns are given on the
command line -- the experiments' one rule, "nothing in the report is typed", kept for every number
the run produced. The row has exp 1's columns plus `channel`.

THE CAP IS THE USER'S (2026-09-25): at most three jobs in flight. `launch` counts this experiment's
jobs that LSF still has PEND or RUN and refuses to go past three -- the queue is asked, not a local
list, because a submit that looked like it failed may have queued (discovery_okuda/cluster.py's
rule: the reported outcome is a hint, the world's state is the fact).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "discovery_okuda"))
MD = os.path.join(ROOT, "experiments", "exp04_membrane_channels.md")
JUDGE = os.path.join(ROOT, "experiments", "specs", "exp04", "judge.jsonl")
PY = "/workspace/.conda_envs/neural-graph-linux/bin/python"
MAX_IN_FLIGHT = 20                                   # the human, 2026-09-26: up to 20 gpu_l4 jobs, 2 per channel per batch
TARGET = None


def _target():
    from exp import load
    fm, _ = load(MD)
    return " ".join(str(fm["target"]).split())


def in_flight():
    """This experiment's jobs LSF holds as PEND or RUN: {job name: state}."""
    import cluster as C
    r = C._ssh("bjobs -w -noheader 2>/dev/null", timeout=60)
    out = {}
    for line in ((r.stdout if r is not None else "") or "").splitlines():
        f = line.split()                                     # a PEND job has no EXEC_HOST: find the name, not a column
        name = next((x for x in f if x.startswith("exp_exp04_")), None)
        if name and len(f) >= 3 and f[2] in ("PEND", "RUN", "PSUSP", "USUSP", "SSUSP"):
            out[name] = f[2]
    return out


def launch(labels):
    import channel_spec as S
    from exp import _submit, job_dir, load, _rewrite
    busy = in_flight()
    if len(busy) + len(labels) > MAX_IN_FLIGHT:
        raise SystemExit(f"{len(busy)} exp04 job(s) in flight ({', '.join(busy) or '-'}); {len(labels)} more "
                         f"would pass the cap of {MAX_IN_FLIGHT}")
    fm, body = load(MD)
    ids = dict(fm.get("job_ids") or {})
    rounds = dict(fm.get("rounds") or {})
    for lab in labels:
        if lab in S.RAN:
            raise SystemExit(f"{lab} has run (channel_spec.RAN); a new round needs new labels")
        ch = S.VERSIONS[lab]["channel"]
        P = S.params_of(lab) if S.VERSIONS[lab].get("builder") != "flow" else None
        if P is not None and S.VERSIONS[lab].get("builder") == "mechano":
            S.build_mechano(lab, P, ch)
            ch = "__built__"
        if S.VERSIONS[lab].get("builder") == "flow":
            S.build_flow(lab, S.params_of_flow(lab))
        elif ch == "__built__":
            ch = S.VERSIONS[lab]["channel"]
        elif ch in ("mscs", "mscl", "traak"):
            S.build_mechano(lab, P, ch)
        elif ch == "kv12":
            S.build_ions(lab, S.params_of_ions(lab), ch)
        elif ch == "ahl":
            Pa = S.copy.deepcopy(S.ASSEMBLY)
            chain_, v_ = [], lab
            while v_ is not None:
                chain_.append(v_); v_ = S.VERSIONS[v_]["parent"]
            for v_ in reversed(chain_):
                Pa.update(S.VERSIONS[v_]["change"])
            S.build_assembly(lab, Pa, ch)
        elif ch == "ahl2":
            Pa = S.copy.deepcopy(S.ASSEMBLY2)
            chain_, v_ = [], lab
            while v_ is not None:
                chain_.append(v_); v_ = S.VERSIONS[v_]["parent"]
            for v_ in reversed(chain_):
                Pa.update(S.VERSIONS[v_]["change"])
            S.build_assembly2(lab, Pa, ch)
        elif ch == "hole":
            Ph = S.copy.deepcopy(S.HOLE)
            chain_, v_ = [], lab
            while v_ is not None:
                chain_.append(v_); v_ = S.VERSIONS[v_]["parent"]
            for v_ in reversed(chain_):
                Ph.update(S.VERSIONS[v_]["change"])
            S.build_hole(lab, Ph)
        else:
            raise SystemExit(f"{ch}: no generator")
        run = f"exp04_v{lab}"
        jd = job_dir(4, run)
        os.makedirs(jd, exist_ok=True)
        cmd = f"python -u Plexus_Main.py -o generate channel/{run} --device cuda:0 --force"
        jid = _submit_streamed(cmd, jd, run)
        ids[run] = jid
        if lab[0].isdigit():                                     # `H4`: an element test, not a round
            K_ = int(re.match(r"(\d+)", lab).group(1))
            rounds[K_] = ch if rounds.get(K_) in (None, ch) else "batch (every channel)"
        print(f"  {run:16s} {ch:5s} job {jid}")
    fm["job_ids"] = ids
    fm["rounds"] = rounds
    _rewrite(MD, fm, body)


def _submit_streamed(cmd, jd, run_name):
    """exp.py's `_submit` for `where: l4`, with ONE difference: `conda run --no-capture-output`.
    Without it conda holds the job's stdout until the process exits, so a job that is killed -- as
    round 1 was, at its blow-up -- leaves no operator log at all, and `status` cannot follow a run."""
    import cluster as C
    sh = os.path.join(jd, "run.sh")
    with open(sh, "w") as f:
        f.write("\n".join([
            "#!/bin/bash -l",
            f"cd {C.cpath(ROOT)}",
            f"export PYTHONPATH={C.cpath(os.path.join(ROOT, 'src'))}",
            "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
            "export MPLBACKEND=Agg PYTHONUNBUFFERED=1",
            f"conda run --no-capture-output -n {C.ENV} {cmd}",
        ]) + "\n")
    os.chmod(sh, 0o755)
    o = os.path.join(jd, "cluster.out")
    gpu = "-gpu num=1 " if C.GPU != "0" else ""
    excl = "".join(f'-R "hname!={h}" ' for h in C.EXCLUDE_HOSTS if h)
    bs = (f"bsub -n {C.NCPUS} {gpu}{excl}-q {C.QUEUE} -W 480 -J exp_{run_name} "
          f"-o {C.cpath(o)} -e {C.cpath(o[:-4])}.err bash -l {C.cpath(sh)}")
    r = C._ssh(bs, timeout=60)
    if r is None or r.returncode != 0:
        print(f"    SUBMIT FAILED: {'ssh timeout' if r is None else (r.stderr or r.stdout)[:120]}")
        return None
    m = re.search(r"Job <(\d+)>", r.stdout or "")
    return m.group(1) if m else (r.stdout or "").strip()[:40]


def status(labels=None):
    from exp import load
    from plexus.paths import graphs_data_path
    fm, _ = load(MD)
    busy = in_flight()
    runs = [f"exp04_v{l}" for l in labels] if labels else sorted(fm.get("job_ids") or {})
    for run in runs:
        d = graphs_data_path("channel", run)
        landed = os.path.isfile(os.path.join(d, "trajectory.npz"))
        movie = os.path.isfile(os.path.join(d, "movie.mp4"))
        co = os.path.join(ROOT, "log", "experiments", "exp04", run, "cluster.out")
        tail = ""
        if os.path.isfile(co):
            txt = open(co, errors="ignore").read()
            m = re.findall(r"(\d+)/(\d+) \[", txt)
            if m:
                tail = f"frame {m[-1][0]}/{m[-1][1]}"
            if "Traceback" in txt:
                tail = "TRACEBACK: " + txt[txt.rfind("Traceback"):].strip().splitlines()[-1][:120]
            w = re.search(r"Run time :\s+(\d+) sec", txt)
            if w:
                tail += f"  ran {w.group(1)} s"
        print(f"  {run:16s} job {str((fm.get('job_ids') or {}).get(run)):10s} "
              f"{busy.get('exp_' + run, 'done' if landed else '-'):5s} landed {landed!s:5s} movie {movie!s:5s} {tail}")


def judge(labels):
    subprocess.run([PY, os.path.join(ROOT, "tools", "exp_record.py"), "4", "--caption-missing"],
                   env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    os.makedirs(os.path.dirname(JUDGE), exist_ok=True)
    for lab in labels:
        subprocess.run([PY, os.path.join(ROOT, "tools", "judge.py"), f"channel/exp04_v{lab}", "--measure", "channel",
                        "--target", _target(), "--out", JUDGE],
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})


def _last_judge(run):
    rec = None
    if os.path.isfile(JUDGE):
        for line in open(JUDGE):
            try:
                d = json.loads(line)
            except Exception:                                    # noqa: BLE001
                continue
            if d.get("spec") == f"channel/{run}":
                rec = d
    return rec


def _lsf_reason(run):
    """The LSF termination reason of a run's job, read from its cluster.out (TERM_OWNER, ...)."""
    co = os.path.join(ROOT, "log", "experiments", "exp04", run, "cluster.out")
    if os.path.isfile(co):
        m = re.search(r"(TERM_\w+)", open(co, errors="ignore").read())
        if m:
            return m.group(1)
    return None


def _ruler_cell(m, run=None):
    """The HEADLINE, as exp 1's single ruler number: pore diameter closed -> widest, tension at
    half-opening, conductance closed -> largest. Everything else the ruler read is in judge.jsonl."""
    if not m:
        why = _lsf_reason(run) if run else None
        return f"no trajectory ({why})" if why else "not landed"
    if m.get("error"):
        return "CRASHED: " + m["error"].strip().splitlines()[-1][:100]
    parts = []
    if "pore_diameter_closed_nm" in m:
        parts.append(f"D {m['pore_diameter_closed_nm']} -> {m.get('pore_diameter_max_nm', '?')} nm")
    if "half_open_tension_mN_m" in m:
        parts.append(f"tau1/2 {m['half_open_tension_mN_m']} mN/m")
    elif "tension_max_mN_m" in m:
        parts.append(f"no opening up to {m['tension_max_mN_m']} mN/m")
    if "conductance_closed_nS" in m:
        parts.append(f"G {m['conductance_closed_nS']} -> {m.get('conductance_max_nS', '?')} nS")
    if "charge_through_core_e" in m and m.get("state"):
        tc = m["through_core_net_in"]
        cat = "Na" if "Na" in tc else "K"
        parts.append(f"{m['state']}: {m['charge_through_core_e']:+.0f} e through the core ({cat} {tc.get(cat, 0):+d}, "
                     f"Cl {tc.get('Cl', 0):+d}) in {m.get('sim_time_ns', '?')} ns = {m.get('conductance_through_core_pS', '?')} pS "
                     f"at {m.get('psi_mV', '?')} mV; path R_min {m.get('narrowest_path_radius_nm')} nm, dry gate {m.get('dry_gate_on_path')}")
    elif "charge_moved_in_e" in m:
        parts.append(f"Q {m['charge_moved_in_e']:+.0f} e into the cell (K {m.get('K_net_in', 0):+d}, "
                     f"Cl {m.get('Cl_net_in', 0):+d}) in {m.get('sim_time_ns', '?')} ns: counted "
                     f"{m.get('conductance_counted_nS', '?')} nS vs continuum {m.get('conductance_continuum_nS', '?')} nS")
    if m.get("nonfinite"):
        parts.append(f"NONFINITE {m['nonfinite']}")
    mv = m.get("movie") or {}
    if mv and abs(float(mv.get("duration_s", 10.0)) - 10.0) > 0.3:
        parts.append(f"movie {mv.get('duration_s')} s")
    return "; ".join(parts) or "no channel readings"


def row(label, agent, verdict, changed, said):
    import channel_spec as S
    from exp import load, _rewrite
    run = f"exp04_v{label}"
    rec = _last_judge(run) or {}
    ch = S.VERSIONS[label]["channel"]
    cells = [label, f"`channel/{run}`", S.CHANNEL_NAME.get(ch, ch), agent,
             _ruler_cell(rec.get("measured"), run), verdict, changed, said]
    cells = [str(c).replace("|", "/").replace("\n", " ") for c in cells]
    fm, body = load(MD)
    # AFTER THE TABLE'S LAST ROW, not at the file's end: `## History` sits below the table (INSTRUCTION.md's
    # template, 2026-09-27), and a row appended at the end would land under it
    lines = body.rstrip("\n").split("\n")
    last = max((i for i, l_ in enumerate(lines) if re.match(r"^\| [A-Za-z]?\d+[a-z]? \| `channel/exp04_v", l_)),
               default=len(lines) - 1)
    lines.insert(last + 1, "| " + " | ".join(cells) + " |")
    body = "\n".join(lines) + "\n"
    _rewrite(MD, fm, body)
    subprocess.run([PY, os.path.join(ROOT, "tools", "exp_record.py"), "4"],
                   env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    print(f"[row] {label} appended (INDEX.md: `tools/exp.py poll` once per round -- it re-reads every grid run)")


def reruler(labels):
    """Rewrite the ruler cell of rows already in the table from their LAST judge.jsonl line -- after
    the ruler itself was fixed (H1/H3 read 'CRASHED' from a caption-step traceback, 2026-09-26)."""
    from exp import load, _rewrite
    fm, body = load(MD)
    lines = body.splitlines()
    for lab in labels:
        run = f"exp04_v{lab}"
        rec = _last_judge(run) or {}
        cell = _ruler_cell(rec.get("measured"), run).replace("|", "/").replace("\n", " ")
        for i, l in enumerate(lines):
            c = l.split(" | ")
            if l.startswith(f"| {lab} | `channel/{run}`") and len(c) > 4:
                c[4] = cell
                lines[i] = " | ".join(c)
                print(f"[reruler] {lab}: {cell}")
    _rewrite(MD, fm, "\n".join(lines) + "\n")
    subprocess.run([PY, os.path.join(ROOT, "tools", "exp_record.py"), "4"],
                   env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})


# ---- THE NOMINAL VIEW, applied to a spec's PLOTTING only (the human, 2026-09-26) --------------------
# Every exp04 movie is re-rendered with record step 0007's view: camera elev 40 / azim 30, zoom 1.0,
# every protein chain and the whole slab drawn, no cut-away, ions as small dots, the panels' x axis in ns.
# The physics blocks (sets, seeds, operators, schedule, n_frames, dt) are not touched; only
# `plotting` and the recording cap (a view of the same run) change.
NOMINAL_VIEW = {"camera": {"elev": 40.0, "azim": 30.0}, "zoom": 1.0}
ION_DOTS = {"render": "dots", "point_size": 4.0}
ION_COLORS = {"K": [0.70, 0.53, 1.00], "Cl": [0.35, 0.85, 0.45]}


def patch_view(path):
    import yaml
    sp = yaml.safe_load(open(path))
    pl = sp.get("plotting") or {}
    sets = list((sp.get("sets") or {}).keys())
    prot = [s_ for s_ in sets if "_" in s_ and s_.split("_")[-1].isdigit()]
    ions = [s_ for s_ in ("K", "Cl") if s_ in sets]
    pl.update(NOMINAL_VIEW)
    for k in ("near_side",):
        pl.pop(k, None)
    surf = pl.setdefault("surface", {})
    if "lipid" in surf:
        surf["lipid"].pop("near_side", None)
    comp = [s_ for s_ in pl.get("compartment_sets", []) if s_ not in prot + ions]
    pl["compartment_sets"] = prot + comp + ions
    pl["hide_sets"] = [s_ for s_ in pl.get("hide_sets", []) if s_ not in prot + ions]
    op = pl.setdefault("opacity", {})
    for s_ in prot:
        op[s_] = 1.0
    if ions:
        pl.pop("spheres", None)
        cols = pl.setdefault("colors", {})
        for s_ in ions:
            surf[s_] = dict(ION_DOTS); op[s_] = 1.0; cols[s_] = ION_COLORS[s_]
    g = sp.get("general") or {}
    u = g.get("units") or {}
    if u.get("time_s") and g.get("dt"):
        pl["curve_time"] = {"per_frame_s": float(g["dt"]) * float(u["time_s"]), "unit": "ns"}
    for cv in pl.get("curve", []) or []:
        if cv.get("ymax") == 0.0 and (cv.get("ymin") or 0.0) < 0.0:
            cv["ymax"] = round(0.05 * abs(float(cv["ymin"])), 3)       # headroom: a 1e-5 blip re-ranged the axis
    g.pop("save_data", None)                                            # every frame of a 1M-frame run was 20 GB
    g["record_cap"] = 3001
    sp["plotting"] = pl; sp["general"] = g
    with open(path, "w") as f:
        yaml.safe_dump(sp, f, sort_keys=False, default_flow_style=None, width=110)
    return len(prot), ions


def rerender(labels):
    """Re-run frozen versions with the nominal view -- ALL AT ONCE on ${CLUSTER_QUEUE_PREFIX}l4 (the human's ask for this
    batch, above the usual cap of 3): the same physics, the movie the watcher shows re-made."""
    from exp import job_dir, load, _rewrite
    fm, body = load(MD)
    ids = dict(fm.get("rerender_job_ids") or {})
    for lab in labels:
        run = f"exp04_v{lab}"
        cfg = os.path.join(ROOT, "config", "channel", f"{run}.yaml")
        if not os.path.isfile(cfg):
            print(f"  {run}: no spec"); continue
        n_prot, ions = patch_view(cfg)
        jd = job_dir(4, run)
        os.makedirs(jd, exist_ok=True)
        jid = _submit_streamed(f"python -u Plexus_Main.py -o generate channel/{run} --device cuda:0 --force", jd, run)
        ids[run] = jid
        print(f"  {run:14s} view patched ({n_prot} chains drawn{', ions ' + '+'.join(ions) + ' as dots' if ions else ''})  job {jid}")
    fm["rerender_job_ids"] = ids
    _rewrite(MD, fm, body)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    l = sub.add_parser("launch"); l.add_argument("labels", nargs="+")
    s = sub.add_parser("status"); s.add_argument("labels", nargs="*")
    j = sub.add_parser("judge"); j.add_argument("labels", nargs="+")
    rr = sub.add_parser("rerender"); rr.add_argument("labels", nargs="+")
    ru = sub.add_parser("reruler"); ru.add_argument("labels", nargs="+")
    r = sub.add_parser("row"); r.add_argument("label")
    r.add_argument("--agent", required=True); r.add_argument("--verdict", required=True)
    r.add_argument("--changed", required=True); r.add_argument("--said", required=True)
    a = ap.parse_args()
    if a.verb == "launch":
        launch(a.labels)
    elif a.verb == "status":
        status(a.labels or None)
    elif a.verb == "judge":
        judge(a.labels)
    elif a.verb == "rerender":
        rerender(a.labels)
    elif a.verb == "reruler":
        reruler(a.labels)
    else:
        row(a.label, a.agent, a.verdict, a.changed, a.said)


if __name__ == "__main__":
    main()
