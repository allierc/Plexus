#!/usr/bin/env python
"""Build an experiment's RECORD FOLDER -- the builder's layout -- from its markdown.

    PYTHONPATH=src python tools/exp_record.py 2            # experiments/exp02_<name>/
    PYTHONPATH=src python tools/exp_record.py 1 2 --caption-missing

WHY THE BUILDER'S LAYOUT. The session watcher (`/watch`, `src/plexus/gui/watch.py`) walks a folder
of `png/NNNN.png` with `mp4/`, `spec/`, `why/`, `caption/` beside it and a `journal.txt`, and it is
pointed at a folder by `PLEXUS_BUILDER`. Giving an experiment that same folder lets the SAME watcher
show it, unchanged, instead of a second viewer that would drift from the first.

WHAT IS DERIVED FROM WHERE -- nothing is typed:

    step NNNN        the Nth iteration row of the markdown's results table, in table order
    mp4/NNNN.mp4     symlink -> graphs_data/<group>/<run>/movie.mp4
    png/NNNN.png     symlink -> graphs_data/<group>/<run>/3d.png          (the run's own still)
    spec/NNNN.yaml   symlink -> graphs_data/<group>/<run>/spec.yaml       (the spec AS RUN), else
                     config/<group>/<run>.yaml
    why/NNNN.txt     that row of the table, one `column: value` line per column
    caption/NNNN.txt the run's VLM description from graphs_data/video_descriptions.txt (the file
                     the generate job's describe step appends to), when there is one
    journal.txt      one line per step: iteration, run, verdict, what changed

The links are RELATIVE (`../../../graphs_data/...`), through the repo's `graphs_data` link, so the
folder reads the same wherever the repo is mounted. The folder is REBUILT from the markdown on every
call -- step files not backed by a row are removed -- so it cannot drift from the table. A caption
written by `--caption-missing` (the local VLM, one run at a time) is kept across rebuilds.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
GD = os.path.join(ROOT, "graphs_data")
KINDS = {"png": ".png", "mp4": ".mp4", "spec": ".yaml", "why": ".txt", "caption": ".txt"}


def md_of(number: int) -> str:
    hits = glob.glob(os.path.join(EXP, f"exp{number:02d}_*.md"))
    if not hits:
        raise SystemExit(f"no experiments/exp{number:02d}_*.md")
    return hits[0]


def table(body: str) -> list[dict]:
    """The results table's iteration rows as dicts keyed by header. A row is one whose first cell
    is an iteration number (`7`, `7a` for a parallel variant, `H1` for an element test) and whose
    second is a `group/run`."""
    lines = body.splitlines()
    head = None
    rows = []
    for l in lines:
        if not l.startswith("|"):
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if head is None and "spec" in [c.lower() for c in cells] and cells and cells[0].lower() == "v":
            head = cells
            continue
        if head and re.fullmatch(r"[A-Z]?\d+[a-z]?", cells[0] or "") and len(cells) >= 2:
            m = re.search(r"`([^`/]+)/([^`]+)`", cells[1])
            if m:
                rows.append({"_v": cells[0], "_group": m.group(1), "_run": m.group(2),
                             **{h: (cells[i] if i < len(cells) else "") for i, h in enumerate(head)}})
    return rows


def descriptions() -> dict:
    """`<group>/<run>` -> its LAST description block in graphs_data/video_descriptions.txt."""
    f = os.path.join(GD, "video_descriptions.txt")
    out, key, buf = {}, None, []
    if not os.path.exists(f):
        return out
    for line in open(f, errors="replace"):
        m = re.match(r"- video name: (.+?)/movie\s*$", line)
        if m:
            if key:
                out[key] = "".join(buf).strip()
            key, buf = m.group(1), [line]
        elif key is not None:
            buf.append(line)
    if key:
        out[key] = "".join(buf).strip()
    return out


def _link(target: str, link: str) -> bool:
    if os.path.lexists(link):
        os.remove(link)
    if not os.path.exists(target):
        return False
    os.symlink(os.path.relpath(target, os.path.dirname(link)), link)
    return True


def grid_rows(number: int, fm: dict, body: str) -> list[dict]:
    """A GRID experiment's steps: every (arm, grid point) in the order `tools/exp.py` launches
    them -- the same derivation, so the watcher walks exactly the runs the markdown declares.

    `why` gets the arm's label, what it patches, and its row of the markdown's prediction table
    (the table whose first column names arms in backticks). The CAPTION is the run's trajectory
    readout -- `experiments/specs/expNN/audit.jsonl` (the growth auditor) and `signatures.jsonl`
    -- when those have been computed: for these runs the numbers are the description.
    """
    from exp import runs
    import json
    pred = {}
    head = None
    for l in body.splitlines():
        if not l.startswith("|"):
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if cells and cells[0].lower() == "arm" and head is None:
            head = cells
            continue
        m = re.fullmatch(r"`([\w]+)`", cells[0] or "") if head else None
        if m and m.group(1) not in pred:
            pred[m.group(1)] = dict(zip(head, cells))
    reads = {}
    for f in ("audit", "signatures", "elongation"):
        jf = os.path.join(EXP, "specs", f"exp{number:02d}", f"{f}.jsonl")
        if os.path.exists(jf):
            for line in open(jf):
                try:
                    d = json.loads(line)
                except Exception:                                # noqa: BLE001
                    continue
                reads.setdefault(d.get("spec"), {})[f] = d
    rows = []
    for arm, point, run in runs(fm):
        group = arm["spec"].split("/")[0]
        rd = reads.get(f"{group}/{run}", {})
        rows.append({"_v": f"{arm['id']} {' '.join(f'{k}={v}' for k, v in point.items())}",
                     "_group": group, "_run": run, "_pred": pred.get(arm["id"], {}),
                     "_read": rd, "arm": arm["id"], "label": arm.get("label", ""),
                     "differs by": ", ".join(f"{k}: {v}" for k, v in (arm.get("differs_by") or {}).items()
                                             if "seed" not in k),
                     "verdict": (f"agent {rd['audit']['score']:.1f} ({rd['audit']['band']})"
                                 if "audit" in rd else "not scored yet"),
                     "what changed": arm.get("label", "")})
    return rows


def _readout(r: dict) -> str:
    """The caption of a grid step: the trajectory's numbers, no picture read."""
    nl = "\n"
    out = []
    a = r["_read"].get("audit")
    if a:
        out.append(nl.join([f"growth audit  {a['score']:.2f}  {a['band']}", f"  {a['reason']}",
                            f"  growth x{a['growth']:.2f} ({a['cells'][0]} -> {a['cells'][1]} cells), "
                            f"p90 wobble {a['jitter_p90']:.4f}", f"  uniformity {a['uniformity']}"]))
    e_ = r["_read"].get("elongation")
    if e_:
        out.append(nl.join([f"elongation  AR {e_['AR_start']:.3f} -> {e_['AR_end']:.3f} "
                            f"(rise {e_['AR_rise']:+.3f})",
                            f"  elongation axis {e_['angle_end_deg']} deg from the declared axis {e_['axis']}",
                            f"  cells {e_['cells'][0]} -> {e_['cells'][1]}"]))
    s_ = r["_read"].get("signatures")
    if s_:
        keys = ["S1_slope", "S2_rho", "S3_g1_slope", "S4_drift", "S4_cv_rise", "S5_L_min",
                "S5_tail_rate", "S5_tail_r2", "S5_sibling_dL", "S6_beta", "cycles"]
        out.append(nl.join(["signatures"] + [
            f"  {k:<14} {s_[k]:.3f}" if isinstance(s_.get(k), float) else f"  {k:<14} {s_.get(k)}"
            for k in keys if k in s_]))
    if r["_pred"]:
        out.append(nl.join(["predicted"] + [f"  {k}: {v}" for k, v in r["_pred"].items()]))
    return (nl + nl).join(out)


def build(number: int) -> str:
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from exp import load                                         # the tool's own md reader
    md = md_of(number)
    fm, body = load(md)
    folder = os.path.splitext(md)[0]
    for k in KINDS:
        os.makedirs(os.path.join(folder, k), exist_ok=True)
    grid = fm.get("branch") == "grid" or "arms" in fm
    rows = grid_rows(number, fm, body) if grid else table(body)
    desc = descriptions()
    keep = set()
    journal = [f"# exp {number} {fm.get('name', '')} -- rebuilt from {os.path.basename(md)} "
               f"by tools/exp_record.py; one step per results-table row"]
    for i, r in enumerate(rows, 1):
        n = f"{i:04d}"
        keep.add(n)
        run = os.path.join(GD, r["_group"], r["_run"])
        p = lambda k: os.path.join(folder, k, n + KINDS[k])      # noqa: E731
        got = {"mp4": _link(os.path.join(run, "movie.mp4"), p("mp4")),
               "png": _link(os.path.join(run, "3d.png"), p("png"))}
        if not _link(os.path.join(run, "spec.yaml"), p("spec")):
            _link(os.path.join(ROOT, "config", r["_group"], r["_run"] + ".yaml"), p("spec"))
        with open(p("why"), "w") as fh:
            fh.write(f"step   {n}  =  {'run ' + r['_v'] if grid else 'iteration v' + r['_v']} of experiment {number}\n")
            fh.write(f"spec   {r['_group']}/{r['_run']}\n")
            fh.write(f"run    graphs_data/{r['_group']}/{r['_run']}/\n\n")
            for h, v in r.items():
                if not h.startswith("_") and h.lower() not in ("v", "spec"):
                    fh.write(f"{h}: {v}\n")
            if r.get("_pred"):
                fh.write("\npredicted (the markdown's hypothesis table):\n")
                for h, v in r["_pred"].items():
                    fh.write(f"  {h}: {v}\n")
        cap = _readout(r) if grid else desc.get(f"{r['_group']}/{r['_run']}")
        if cap:
            with open(p("caption"), "w") as fh:
                fh.write(cap + "\n")
        verdict = next((v for h, v in r.items() if h.lower() == "verdict"), "")
        changed = next((v for h, v in r.items() if h.lower() == "what changed"), "")
        tag = r["_v"] if grid else f"v{r['_v']}"
        journal.append(f"{n}  {tag:<5} {r['_group']}/{r['_run']:<14} {verdict}  -- {changed}"
                       + ("" if got["mp4"] else "   [no movie yet]"))
    for k, ext in KINDS.items():                                 # steps no row backs any more
        for f in glob.glob(os.path.join(folder, k, "[0-9][0-9][0-9][0-9]" + ext)):
            if os.path.basename(f)[:4] not in keep:
                os.remove(f)
    with open(os.path.join(folder, "journal.txt"), "w") as fh:
        fh.write("\n".join(journal) + "\n")
    print(f"[record] {folder}: {len(rows)} step(s)")
    return folder


def caption_missing(folder: str):
    """Caption, with the local VLM, every step that has a movie and no caption -- one at a time."""
    os.environ["PLEXUS_BUILDER"] = folder
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import gui_drive as G
    for mp4 in sorted(glob.glob(os.path.join(folder, "mp4", "[0-9][0-9][0-9][0-9].mp4"))):
        n = int(os.path.basename(mp4)[:4])
        cap = os.path.join(folder, "caption", f"{n:04d}.txt")
        if os.path.exists(cap) and os.path.getsize(cap) > 0:
            continue
        r = G.caption(os.path.realpath(mp4), index=n)
        print(f"[record] caption {n:04d}: {'ok' if r.get('caption') else r.get('error')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("numbers", nargs="+", type=int)
    ap.add_argument("--caption-missing", action="store_true",
                    help="caption steps with no caption using the local VLM, sequentially")
    a = ap.parse_args()
    for num in a.numbers:
        fo = build(num)
        if a.caption_missing:
            caption_missing(fo)
