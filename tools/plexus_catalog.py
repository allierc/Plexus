#!/usr/bin/env python
"""Catalog every spec Plexus has run or recorded, and pool the ones relevant to an experiment.

    python tools/plexus_catalog.py build                 index everything -> experiments/specs/catalog/plexus_catalog.jsonl
    python tools/plexus_catalog.py ops                   operator frequency over the catalog (to write rules)
    python tools/plexus_catalog.py pool 11 [12 ...]      rebuild experiments/expNN_<name>/relevant_in_plexus/
    python tools/plexus_catalog.py pool --all

WHY. An experiment that starts from nothing rebuilds what Plexus already has: 2,865 config specs,
773 specs under log/, 710 builder steps, 226 experiment specs (2026-09-26). The catalog reads each
spec of the CODEBASE once -- config/, log/, builder/, experiments/ -- with the operators it uses
(`op` and `op[model]`), its sets, the output folder it produced if one exists (graphs_data/<group>/
<name>, or its log folder), and the words written about it (the builder's `why` and caption, the
spec's own header comments). graphs_data is never walked, only each spec's own folder is looked up.
`pool` then selects, per experiment, what its `relevant.yaml` asks for.

WHAT A POOL IS. `experiments/expNN_<name>/relevant_in_plexus/` holds SYMLINKS ONLY, relative, through
the repo's `graphs_data` link, so nothing is copied and the folder reads the same on any mount:

    INDEX.md            one table, most relevant first: kind, name, why it matched, outputs, links
    index.json          the same, for an agent
    runs/<group>__<run>             -> graphs_data/<group>/<run>/         (the output of a spec below)
    log/<path__joined>              -> log/<path>/                        (spec.yaml, metrics, movie)
    builder/<exp>__<NNNN>.<ext>     -> builder/<exp>/{spec,png,mp4,why,caption}/NNNN.<ext>
    experiments/<exp>__<NNNN>.<ext> -> experiments/<exp>/{spec,png,mp4,why}/NNNN.<ext>
    specs/<group>__<stem>.yaml      -> config/<group>/.../<stem>.yaml     (declared specs, with or without a run)

The folder is DERIVED: `pool` deletes it and rebuilds it from the catalog and the rules, so a
hand-made link in it does not survive. What a pool entry is FOR is decided by the experiment's
judge and its Stage 0 audit, not by this tool; the tool only finds the candidates.

THE RULES (`experiments/expNN_<name>/relevant.yaml`):

    ops_any: [cell_chem_diffuse, "cell_chem_react[rock_paper_scissor]"]   an op, or op[model]
    keywords: ["spiral", "\\brps\\b"]           regex, case-insensitive, over name + path + text
    key_ops: [cell_chem_react[rock_paper_scissor]]   the defining operators: +6 each
    exclude: ["_worktrees/"]                    regex over the path; worktree copies are excluded by default
    score_min: 3                                 +6 per key op, +3 per other op, +1 per keyword, +1 with an output
    max_per_family: 3                            entries sharing kind, group and the name's first two tokens
    reserve_per_key_op: 3                        each key op's best entries are kept first, over the caps
    max_per_kind: {run: 40, log: 20, builder: 30, experiment: 20, config: 20}
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import shutil
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GD = os.path.join(ROOT, "graphs_data")                     # the repo's link to GraphData/graphs_data
EXP = os.path.join(ROOT, "experiments")
CAT = os.path.join(EXP, "specs", "catalog", "plexus_catalog.jsonl")
try:
    _Loader = yaml.CSafeLoader
except AttributeError:                                                       # pragma: no cover
    _Loader = yaml.SafeLoader


# ============================================================================ reading one spec
def _header(path, n=12):
    """The spec's leading comment block -- where Plexus specs say what they are for."""
    out = []
    try:
        for line in open(path, errors="ignore"):
            s = line.strip()
            if s.startswith("#"):
                out.append(s.lstrip("# ").rstrip())
                if len(out) >= n:
                    break
            elif s and out:
                break
            elif s and not out:
                break
    except OSError:
        pass
    return " ".join(x for x in out if x)


def parse_spec(path):
    """(name, ops, sets, n_frames, dim) of a spec file; empty values when it is not a Plexus spec."""
    try:
        s = yaml.load(open(path, errors="ignore"), Loader=_Loader)
    except Exception:                                                        # noqa: BLE001
        return None
    if not isinstance(s, dict):
        return None
    g = s.get("general") or {}
    ops = set()
    for sec in ("operators", "seed"):
        for o in s.get(sec) or []:
            if isinstance(o, dict) and o.get("op"):
                ops.add(str(o["op"]))
                if o.get("model"):
                    ops.add(f"{o['op']}[{o['model']}]")
    sets = sorted((s.get("sets") or {}).keys()) if isinstance(s.get("sets"), dict) else []
    return dict(name=str(g.get("name") or ""), ops=sorted(ops), sets=sets,
                n_frames=g.get("n_frames"), dim=g.get("dim"))


def _outputs(d):
    f = set(os.listdir(d)) if os.path.isdir(d) else set()
    return dict(trajectory="trajectory.npz" in f or "traj.npz" in f,
                movie=any(x.endswith(".mp4") for x in f),
                still=any(x.endswith(".png") for x in f),
                metrics="metrics.json" in f)


# ============================================================================ build
# THE SPECS COME FROM THE CODEBASE ONLY -- config/, log/, builder/, experiments/. graphs_data is never
# walked: for each spec the tool only asks whether ITS OWN output folder exists
# (graphs_data/<group>/<name>), so the catalog is a map of what the repo declared and what it produced,
# not an inventory of a data tree other sessions also write to.
GROUPS = sorted(d for d in os.listdir(os.path.join(ROOT, "config")) if os.path.isdir(os.path.join(ROOT, "config", d)))


def _run_folder(group, *names):
    """graphs_data/<group>/<name> for the first name whose folder exists, else ''."""
    for n in names:
        if n and group and os.path.isdir(os.path.join(GD, group, n)):
            return f"graphs_data/{group}/{n}"
    return ""


def build():
    rows = []
    # 1. declared config specs, each with its run folder if the run exists
    name_group = {}
    for sp in sorted(glob.glob(os.path.join(ROOT, "config", "**", "*.yaml"), recursive=True)):
        rel = os.path.relpath(sp, os.path.join(ROOT, "config"))
        grp = rel.split(os.sep)[0]
        stem = os.path.splitext(os.path.basename(sp))[0]
        p = parse_spec(sp)
        if not p:
            continue
        run = _run_folder(grp, p["name"], stem)
        if p["name"]:
            name_group.setdefault(p["name"], grp)
        name_group.setdefault(stem, grp)
        rows.append(dict(kind="config", id=f"config:{rel}", path=os.path.relpath(sp, ROOT), group=grp, **p,
                         run=run, outputs=_outputs(os.path.join(ROOT, run)) if run else {}, text=_header(sp)))
    # 2. log/ folders with a spec (the folder IS the output)
    for sp in glob.glob(os.path.join(ROOT, "log", "**", "spec.yaml"), recursive=True):
        d = os.path.dirname(sp)
        rel = os.path.relpath(d, ROOT)
        p = parse_spec(sp)
        if not p:
            continue
        txt = ""
        mj = os.path.join(d, "metrics.json")
        if os.path.exists(mj):
            txt = open(mj, errors="ignore").read(400)
        rows.append(dict(kind="log", id=f"log:{rel}", path=rel, group=rel.split(os.sep)[1] if rel.count(os.sep) else "",
                         **p, run=rel, outputs=_outputs(d), text=(_header(sp) + " " + txt).strip()))
    # 3. builder and 4. experiment records: spec/why/caption/png/mp4 NNNN; the run by the spec's name
    for kind, base in (("builder", "builder"), ("experiment", "experiments")):
        for sp in sorted(glob.glob(os.path.join(ROOT, base, "*", "spec", "*.yaml"))):
            rec = os.path.dirname(os.path.dirname(sp))
            step = os.path.splitext(os.path.basename(sp))[0]
            p = parse_spec(sp)
            if not p:
                continue
            run = ""
            if os.path.islink(sp):                                           # a record step links the run's spec
                tgt = os.path.realpath(sp)
                gdr = os.path.realpath(GD)
                if tgt.startswith(gdr):
                    run = "graphs_data/" + os.path.relpath(os.path.dirname(tgt), gdr)
            if not run and p["name"]:
                run = _run_folder(name_group.get(p["name"], ""), p["name"])
            why = ""
            for sub in ("why", "caption"):
                w = os.path.join(rec, sub, f"{step}.txt")
                if os.path.exists(w):
                    why += open(w, errors="ignore").read(700) + " "
            rows.append(dict(kind=kind, id=f"{kind}:{os.path.basename(rec)}/{step}",
                             path=os.path.relpath(rec, ROOT), step=step, group=name_group.get(p["name"], ""), **p,
                             run=run, outputs=dict(movie=os.path.exists(os.path.join(rec, "mp4", f"{step}.mp4")),
                                                   still=os.path.exists(os.path.join(rec, "png", f"{step}.png")),
                                                   trajectory=bool(run) and os.path.exists(os.path.join(ROOT, run, "trajectory.npz"))),
                             text=why.strip()))
    # 5. experiment grid specs (tools/exp.py writes them into experiments/specs/expNN/)
    for sp in sorted(glob.glob(os.path.join(EXP, "specs", "exp*", "*.yaml"))):
        p = parse_spec(sp)
        if p:
            run = next((_run_folder(g, p["name"]) for g in GROUPS if _run_folder(g, p["name"])), "")
            rows.append(dict(kind="experiment", id=f"experiment:{os.path.relpath(sp, EXP)}",
                             path=os.path.relpath(sp, ROOT), group="", **p, run=run,
                             outputs=_outputs(os.path.join(ROOT, run)) if run else {}, text=_header(sp)))
    os.makedirs(os.path.dirname(CAT), exist_ok=True)
    with open(CAT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    c = collections.Counter(r["kind"] for r in rows)
    w = collections.Counter(r["kind"] for r in rows if r.get("run"))
    print(f"[catalog] {len(rows)} specs -> {os.path.relpath(CAT, ROOT)}  {dict(c)}; with an output: {dict(w)}")
    return rows


def load_catalog():
    if not os.path.exists(CAT):
        return build()
    return [json.loads(x) for x in open(CAT)]


# ============================================================================ pool
def _exp_folder(n):
    h = sorted(glob.glob(os.path.join(EXP, f"exp{int(n):02d}_*/")))
    if not h:
        raise SystemExit(f"no experiment folder for {n}")
    return h[0].rstrip("/")


def _family(r):
    """What makes two entries the same thing at different settings: kind, group and the first two
    underscore tokens of the name (`vicsek_slime_4t_glide03` and `_glide12` are one family)."""
    nm = (r.get("name") or os.path.basename(r["path"])).split("_")
    return f"{r['kind']}:{r.get('group', '')}:{'_'.join(nm[:2])}"


def select(rows, R):
    """Score every catalog entry against the experiment's rules and keep the best, diversely.

    score = 6 per `key_ops` matched (the experiment's defining operators) + 3 per other `ops_any`
    matched + 1 per keyword + 1 when an output exists. At most `max_per_family` entries of one family
    (default 3), and one entry per (record, spec name) for builder/experiment steps, so twenty steps
    of one spec cannot crowd out the one spec that defines the experiment."""
    key_ops = set(R.get("key_ops") or [])
    ops_any = set(R.get("ops_any") or []) | key_ops
    kws = [re.compile(k, re.I) for k in R.get("keywords") or []]
    excl = [re.compile(k, re.I) for k in (R.get("exclude") or []) + ([] if R.get("include_worktrees") else ["_worktrees/"])]
    smin = float(R.get("score_min", 3))
    scored = []
    for r in rows:
        if any(e.search(r["path"]) for e in excl):
            continue
        mo = sorted(ops_any & set(r.get("ops") or []))
        hay = " ".join([r.get("name") or "", r["path"], r.get("text") or ""])
        mk = sorted({k.pattern for k in kws if k.search(hay)})
        out = any((r.get("outputs") or {}).values())
        s = 6 * len(key_ops & set(mo)) + 3 * len(set(mo) - key_ops) + len(mk) + (1 if out else 0)
        if s >= smin and (mo or mk):
            scored.append(dict(r, score=s, matched_ops=mo, matched_keywords=mk))
    cap = {"run": 40, "log": 20, "builder": 30, "experiment": 20, "config": 20}
    cap.update(R.get("max_per_kind") or {})
    fam_max = int(R.get("max_per_family", 3))
    reserve = int(R.get("reserve_per_key_op", 3))
    by, fam, seen, taken = collections.defaultdict(list), collections.Counter(), set(), set()

    def take(r, force=False):
        if r["id"] in taken:
            return
        if r["kind"] in ("builder", "experiment") and r.get("step"):
            k = (r["path"], r.get("name"))
            if k in seen:
                return
        f = _family(r)
        if fam[f] >= fam_max or (not force and len(by[r["kind"]]) >= int(cap.get(r["kind"], 20))):
            return
        if r["kind"] in ("builder", "experiment") and r.get("step"):
            seen.add((r["path"], r.get("name")))
        fam[f] += 1
        taken.add(r["id"])
        by[r["kind"]].append(r)

    ranked = sorted(scored, key=lambda r: (-r["score"], r["id"]))
    # EVERY DEFINING OPERATOR IS REPRESENTED: its best `reserve` entries come first, whatever else
    # outscores them -- twenty-five variants of a richer spec must not hide the one spec that runs it.
    for op in sorted(key_ops):
        n = 0
        for r in ranked:
            if n >= reserve:
                break
            if op in r["matched_ops"] and r["id"] not in taken:
                before = len(taken)
                take(r, force=True)
                n += len(taken) > before
    for r in ranked:
        take(r)
    out = [r for k in ("run", "builder", "log", "experiment", "config") for r in by[k]]
    return sorted(out, key=lambda r: (("run", "builder", "log", "experiment", "config").index(r["kind"]), -r["score"]))


def _link(src_abs, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(src_abs, os.path.dirname(dst)), dst)


def pool(n, rows):
    ef = _exp_folder(n)
    rf = os.path.join(ef, "relevant.yaml")
    if not os.path.exists(rf):
        print(f"[pool] exp{int(n):02d}: no relevant.yaml, skipped")
        return
    R = yaml.safe_load(open(rf)) or {}
    sel = select(rows, R)
    out = os.path.join(ef, "relevant_in_plexus")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    table = []
    for r in sel:
        links = []
        if r["kind"] in ("builder", "experiment") and r.get("step"):
            rec = os.path.join(ROOT, r["path"])
            for sub, ext in (("spec", "yaml"), ("png", "png"), ("mp4", "mp4"), ("why", "txt"), ("caption", "txt")):
                src = os.path.join(rec, sub, f"{r['step']}.{ext}")
                if os.path.exists(src):
                    suffix = f".{sub}.txt" if ext == "txt" else f".{ext}"
                    dst = os.path.join(out, "builder" if r["kind"] == "builder" else "experiments",
                                       f"{os.path.basename(rec)}__{r['step']}{suffix}")
                    _link(src, dst); links.append(os.path.relpath(dst, out))
        elif r["kind"] == "log":
            dst = os.path.join(out, "log", r["path"].split("/", 1)[1].replace("/", "__"))
            _link(os.path.join(ROOT, r["path"]), dst); links.append(os.path.relpath(dst, out))
        else:                                                                # config / experiment grid spec
            base = os.path.join(ROOT, "config") if r["kind"] == "config" else EXP
            dst = os.path.join(out, "specs", os.path.relpath(os.path.join(ROOT, r["path"]), base).replace("/", "__"))
            _link(os.path.join(ROOT, r["path"]), dst); links.append(os.path.relpath(dst, out))
        if r.get("run") and r["kind"] != "log":
            rr = r["run"].split("/", 1)[1] if r["run"].startswith("graphs_data/") else r["run"]
            dst = os.path.join(out, "runs", rr.replace("/", "__"))
            if not os.path.lexists(dst):
                _link(os.path.join(ROOT, r["run"]), dst)
            links.append(os.path.relpath(dst, out))
        o = r.get("outputs") or {}
        table.append(dict(score=r["score"], kind=r["kind"], id=r["id"], name=r.get("name"), path=r["path"],
                          matched_ops=r["matched_ops"], matched_keywords=r["matched_keywords"],
                          outputs=[k for k, v in o.items() if v], n_frames=r.get("n_frames"),
                          text=(r.get("text") or "")[:300], links=links))
    json.dump(table, open(os.path.join(out, "index.json"), "w"), indent=1)
    L = [f"# Relevant in Plexus -- exp{int(n):02d}", "",
         f"DERIVED by `tools/plexus_catalog.py pool {int(n)}` from `{os.path.relpath(rf, ROOT)}` over the catalog "
         f"`{os.path.relpath(CAT, ROOT)}`; do not edit, rebuild. Symlinks only. Score = 6 per defining operator (`key_ops`), "
         "3 per other operator, 1 per keyword, 1 when an output exists; at most 3 per family of variants. A candidate, "
         "not a verdict: the Stage 0 audit says what each is FOR.", "",
         f"{len(table)} entries: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(t['kind'] for t in table).items()), "",
         "| score | kind | name / path | operators matched | keywords | outputs | what it is (caption, why, header) |",
         "|---|---|---|---|---|---|---|"]
    for t in table:
        first = " / ".join(f"[{os.path.basename(l)}]({l})" for l in t["links"][:3])
        txt = (t["text"] or "").replace("|", "/").replace("\n", " ")[:160]
        L.append(f"| {t['score']} | {t['kind']} | {t['name'] or ''} {first} | {', '.join(t['matched_ops'])} | "
                 f"{', '.join(t['matched_keywords'])} | {', '.join(t['outputs'])} | {txt} |")
    open(os.path.join(out, "INDEX.md"), "w").write("\n".join(L) + "\n")
    print(f"[pool] exp{int(n):02d}: {len(table)} entries -> {os.path.relpath(out, ROOT)}  "
          f"{dict(collections.Counter(t['kind'] for t in table))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "pool", "ops"])
    ap.add_argument("n", nargs="*", type=int)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build(); return
    rows = load_catalog()
    if a.cmd == "ops":
        c = collections.Counter(o for r in rows for o in r.get("ops") or [])
        for o, k in c.most_common():
            print(f"{k:6d}  {o}")
        return
    ns = a.n or ([int(os.path.basename(p.rstrip("/"))[3:5]) for p in sorted(glob.glob(os.path.join(EXP, "exp[0-9][0-9]_*/")))
                  if os.path.exists(os.path.join(p, "relevant.yaml"))] if a.all else [])
    for n in ns:
        pool(n, rows)


if __name__ == "__main__":
    main()
