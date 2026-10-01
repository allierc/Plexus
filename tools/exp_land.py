#!/usr/bin/env python
"""Land a batch: every chore a session did by hand after its waiter exited, done in one call.

    PYTHONPATH=src:tools python tools/exp_land.py 11                                  runs landed since the last landing
    PYTHONPATH=src:tools python tools/exp_land.py 11 --runs tissue/exp11_p2_hole_s1 tissue/exp11_p2_hole_s2
    PYTHONPATH=src:tools python tools/exp_land.py --health tissue/exp11_p2_hole_s1     one run's health, nothing written

`tools/exp_wait.py` calls this itself when its batch is decided (unless `--no-land`), so a session is
woken ONCE, with the batch already checked, scored and recorded.

WHAT IT DOES, in order:
  1. HEALTH, per run, read from the trajectory and the movie alone (no ruler, no gate):
       - every set's count first -> last (its `occ` block), and its extent (bounding-box diagonal)
         first -> last -- "what the tissue did" (exp12: a budding runaway was visible in the count);
       - NON-FINITE values in EVERY float block of EVERY set, with the first frame -- not only the
         movie's subject set (a NaN in the membrane under a finite tissue went unseen);
       - the growth auditor (`tools/growth_audit.py`, jump and finiteness on every frame) on mesh runs;
       - the movie: present, and whether its pixels move while the state does (exp06: ~14 movies drawn
         in the wrong colour showed nothing while the numbers moved).
  2. SCORE: `tools/exp_gate_score.py N` (measures through its cache, the card appended), then
     `--check` (unknown keys, arms, rulers, staleness).
  3. RECORD: `tools/exp_record.py N` rebuilds the watcher's record.
  4. WRITE: `experiments/expNN_<name>/landings/<date>_<time>.md`, the same text between
     `<!-- LANDING -->` and `<!-- /LANDING -->` in the experiment's markdown (tool-owned, like
     STATUS), and stdout -- ending with the time the next batch is due (10 minutes after landing).

Nothing is submitted, nothing runs on the cluster: `/groups` is mounted and read locally.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]
PY = sys.executable
EXP = os.path.join(ROOT, "experiments")
LAND_OPEN, LAND_CLOSE = "<!-- LANDING -->", "<!-- /LANDING -->"
NEXT_BATCH_MIN = 10          # INSTRUCTION.md: at most 10 minutes from a landing to the next batch submitted
STATIC_PIX = 0.002           # mean |pixel change| between sampled movie frames, a fraction of 255, below which
                             # the movie reads as still (a clean 8 s tissue movie moves 0.01-0.05)
MOVED = 0.02                 # the state moved: extent or count changed by more than 2 % of its first value


def _gd():
    from plexus.paths import graphs_data_path
    return graphs_data_path()


def _rows(z, s):
    """Recorded rows of set s: the leading axis of its occ or pos block (0 if it has neither)."""
    for k in (f"{s}__occ", f"{s}__pos"):
        if k in z.files:
            a = z[k]
            return int(a.shape[0]) if a.ndim >= 2 else 0
    return 0


def _count(z, s, t, T):
    k = f"{s}__occ"
    if k in z.files:
        a = np.asarray(z[k][t])
        return int((a > 0).sum())
    k = f"{s}__pos"
    return int(np.asarray(z[k]).shape[1]) if k in z.files and np.asarray(z[k]).ndim >= 2 else None


def _extent(z, s, t):
    k = f"{s}__pos"
    if k not in z.files:
        return None
    P = np.asarray(z[k][t], float)
    if P.ndim != 2:
        return None
    if f"{s}__occ" in z.files:
        occ = np.asarray(z[f"{s}__occ"][t]).reshape(-1)[:len(P)] > 0
        P = P[:len(occ)][occ]
    P = P[np.isfinite(P).all(1)]
    return float(np.linalg.norm(P.max(0) - P.min(0))) if len(P) else None


def _movie_motion(mp4, n=12):
    """Median mean-|pixel change| between n evenly spaced frames, as a fraction of 255; None if unreadable."""
    try:
        import imageio
        r = imageio.get_reader(mp4, "ffmpeg")
        nfr = r.count_frames()
        if nfr < 2:
            return 0.0, nfr
        idx = np.unique(np.linspace(0, nfr - 1, n).astype(int))
        F = [np.asarray(r.get_data(int(i)), np.float32)[::4, ::4] for i in idx]
        r.close()
    except Exception:                                                                # noqa: BLE001
        return None, 0
    d = [float(np.abs(b - a).mean() / 255.0) for a, b in zip(F[:-1], F[1:])]
    return float(np.median(d)), nfr


def health(run: str, audit=True) -> dict:
    """What the run did, from its own folder. `run` is `<group>/<name>`."""
    d = os.path.join(_gd(), run)
    tj, mv = os.path.join(d, "trajectory.npz"), os.path.join(d, "movie.mp4")
    h = {"run": run, "trajectory": os.path.exists(tj), "movie": os.path.exists(mv)}
    if not h["trajectory"]:
        return h
    z = np.load(tj)
    # SIMULATED FRAMES ARE NOT RECORDED ROWS: `frame_ms` has one entry per simulated frame, the state blocks
    # one row per RECORDED frame (`record_cap`): exp08's 2,000 frames are 183 rows, and indexing rows by
    # the frame count raised IndexError (2026-09-27). Rows are read per set, from its own blocks.
    first_key = next((k for k in z.files if k.endswith("__pos")), None)
    R = int(np.asarray(z[first_key]).shape[0]) if first_key else 0
    h["frames"] = int(np.asarray(z["frame_ms"]).size) if "frame_ms" in z.files else R
    h["rows"] = R
    if "frame_ms" in z.files:
        h["ms_per_frame"] = round(float(np.median(np.asarray(z["frame_ms"], float))), 1)
    sets = sorted({k.split("__")[0] for k in z.files if "__" in k})
    h["sets"], moved = {}, False
    bad = []
    for s in sets:
        row = {}
        T = _rows(z, s)
        if T:
            c0, c1 = _count(z, s, 0, T), _count(z, s, T - 1, T)
            e0, e1 = _extent(z, s, 0), _extent(z, s, T - 1)
            row.update(count=[c0, c1], extent=[None if e0 is None else round(e0, 3), None if e1 is None else round(e1, 3)])
            if c0 and c1 is not None and abs(c1 - c0) > MOVED * c0:
                moved = True
            if e0 and e1 is not None and abs(e1 - e0) > MOVED * e0:
                moved = True
        for k in z.files:
            if not k.startswith(s + "__") or "mesh_" in k:
                continue
            a = z[k]
            if not np.issubdtype(a.dtype, np.floating):
def training_health(run: str) -> dict:
    """A TRAINER'S run (`training/<model>/<name>`, exp16) has no trajectory: its health is its training
    history (`results/history.jsonl`: every loss finite, the last validation against persistence), its
    held-out test (`results/<name>_test.json`, the free rollout finite) and its movie."""
    import json
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from exp_measures.common import run_dir
    d = run_dir(run)
    name = os.path.basename(d.rstrip("/"))
    hist = os.path.join(d, "results", "history.jsonl")
    rows = [json.loads(l) for l in open(hist)] if os.path.exists(hist) else []
    tj = os.path.join(d, "results", f"{name}_test.json")
    h = {"run": run, "training": True, "trajectory": True, "movie": os.path.exists(os.path.join(d, "results", "movie.mp4")),
         "iterations": max([r.get("it", 0) for r in rows] or [0]), "tested": os.path.exists(tj), "non_finite": []}
    bad = [r["it"] for r in rows if "loss" in r and not np.isfinite(r["loss"])]
    if bad:
        h["non_finite"].append(f"loss not finite at iteration(s) {bad[:5]}")
    val = [r for r in rows if any(k.startswith("val_rmse_h") for k in r)]
    if val:
        h["val_last"] = {k: round(v, 5) for k, v in val[-1].items() if k.startswith(("val_rmse", "val_persistence", "val_blur"))}
    if h["tested"]:
        t = json.load(open(tj))
        # THREE TEST FORMATS: exp16's held-out forecast (`horizons`, `free.rmse`), its full rollout (mode "full":
        # top-level `rmse`, `r2`) and exp17's trace test (mode "trace": skills over the mean baseline, `free`).
        if t.get("mode") == "trace":
            if not t["free"].get("finite", 1.0):
                h["non_finite"].append("the free rollout left the finite numbers")
            h["test"] = {"skill_short": round(t["skill_short"], 4), "skill_long": round(t["skill_long"], 4),
                         "free_r2_denoised": round(t["free"]["r2_denoised"], 4)}
        elif t.get("mode") == "full":
            if not np.isfinite(np.asarray(t.get("rmse", np.nan), float)).all():
                h["non_finite"].append("the full rollout's RMSE is not finite")
            h["test"] = {k: t[k] for k in ("r2", "r2_spatial") if k in t}
        else:
            if not np.isfinite(t["free"]["rmse"]).all():
                h["non_finite"].append("the free rollout's RMSE is not finite")
            h["test"] = {k: (round(v["rmse"], 5), round(v["persistence_rmse"], 5)) for k, v in t["horizons"].items()}
    ev = [r for r in rows if "eval_skill_short" in r]
    if ev:
        h["eval_last"] = {k: round(ev[-1][k], 4) for k in ("eval_skill_short", "eval_skill_long")}
    return h


                continue
            nf = ~np.isfinite(a)
    if run.startswith("training/"):
        return training_health(run)
            if nf.any():
                fr = int(np.argmax(nf.reshape(nf.shape[0], -1).any(1))) if a.ndim >= 2 and a.shape[0] == T else None
                bad.append(f"{k}: {int(nf.sum())} non-finite" + (f" from frame {fr}" if fr is not None else ""))
            del a, nf
        h["sets"][s] = row
    h["non_finite"] = bad
    if audit and "vertex__mesh_nF" in z.files:
        try:
            import warnings
            import growth_audit as GA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = GA.audit(run)
            h["audit"] = {"score": r["score"], "band": r["band"], "reason": r["reason"],
                          "jump_max": r.get("jump_max"), "jump_at": r.get("jump_at")}
        except Exception as e:                                                       # noqa: BLE001
            h["audit"] = {"error": f"{type(e).__name__}: {e}"}
    if h["movie"]:
        m, nfr = _movie_motion(mv)
        h["movie_motion"], h["movie_frames"] = (None if m is None else round(m, 4)), nfr
        h["movie_static"] = bool(m is not None and m < STATIC_PIX and moved)
    h["state_moved"] = moved
    return h


def health_lines(h: dict) -> list[str]:
    """One compact block per run, the flags first."""
    flags = []
    if not h.get("trajectory"):
        return [f"- `{h['run']}`: NO trajectory"]
    if h.get("non_finite"):
        flags.append("NON-FINITE: " + "; ".join(h["non_finite"][:4]) + (" ..." if len(h["non_finite"]) > 4 else ""))
    a = h.get("audit") or {}
    if a.get("score") is not None and a["score"] < 2:
        flags.append(f"WRECKED (audit {a['score']}): {a['reason']}")
    if not h.get("movie"):
        flags.append("NO movie.mp4")
    elif h.get("movie_static"):
        flags.append(f"STATIC MOVIE: pixels move {h['movie_motion']} of 255 per step while the state moved -- "
                     f"look at it before believing it")
    sets = ", ".join(f"{s} {r['count'][0]}->{r['count'][1]}"
                     + (f" (extent {r['extent'][0]}->{r['extent'][1]})" if r.get("extent") and r["extent"][0] is not None else "")
                     for s, r in h.get("sets", {}).items() if r.get("count") and r["count"][0] is not None)
    head = (f"- `{h['run']}`: {h.get('frames')} frames"
            + (f" ({h['rows']} recorded)" if h.get("rows") and h.get("rows") != h.get("frames") else "")
            + (f", {h['ms_per_frame']} ms/frame" if h.get("ms_per_frame") else "")
            + (f", audit {a['score']} ({a['band']})" if a.get("score") is not None else "")
            + (f", movie motion {h['movie_motion']}" if h.get("movie_motion") is not None else ""))
    out = [head, f"  - sets: {sets}"]
    out += [f"  - **{f}**" for f in flags]
    return out


def _exp_dir(n):
    from exp import exp_path
    p = exp_path(n)
    return p, p[:-3]


def _since_last(n) -> list[str]:
    """Runs of experiment n whose trajectory is newer than the last landing report."""
    _, d = _exp_dir(n)
    reps = sorted(glob.glob(os.path.join(d, "landings", "*.md")))
    t_last = os.path.getmtime(reps[-1]) if reps else 0.0
    out = []
    for tj in glob.glob(os.path.join(_gd(), "*", f"exp{int(n):02d}*", "trajectory.npz")):
        if os.path.getmtime(tj) > t_last:
    if h.get("training"):
        f = h["non_finite"] + ([] if h["tested"] else ["NOT TESTED"]) + ([] if h["movie"] else ["no movie"])
        head = f"- **`{h['run']}`**: " + "; ".join(f) if f else f"- `{h['run']}`: healthy"
        return [head, f"  - {h['iterations']} iterations; last validation {h.get('val_last', {})}",
                f"  - test (model, persistence) per horizon: {h.get('test', {})}"]
            out.append(os.path.relpath(os.path.dirname(tj), _gd()))
    return sorted(out)


def _gap(n, runs, d):
    """The time this batch waited on the SESSION: from the previous landing report to this batch's first
    submission (its job's run.sh, written by tools/exp.py at submit). None when either is unknown."""
    from exp import job_dir
    subs = [os.path.getmtime(p) for r in runs
            if os.path.exists(p := os.path.join(job_dir(n, r.split("/", 1)[1]), "run.sh"))]
    reps = [p for p in glob.glob(os.path.join(d, "landings", "*.md")) if subs and os.path.getmtime(p) < min(subs)]
    if not subs or not reps:
        return None
    prev = max(os.path.getmtime(p) for p in reps)
    g = (min(subs) - prev) / 60.0
    return (f"Gap between batches: previous landing {time.strftime('%H:%M', time.localtime(prev))} -> this batch "
            f"submitted {time.strftime('%H:%M', time.localtime(min(subs)))} = {g:.0f} min"
            + (f" -- OVER the {NEXT_BATCH_MIN}-minute limit" if g > NEXT_BATCH_MIN else f" (limit {NEXT_BATCH_MIN})"))


_NOISE = re.compile(r"^(Warp \d|\s+CUDA Toolkit|\s+Devices:|\s+\"c(pu|uda)|\s+Kernel cache|\s+/home/.*/warp/|"
                    r"\s+CUDA peer|\s+Supported fully|.*RuntimeWarning|\s+jit = np\.nanmax)")


def _question(md):
    try:
        import exp
        return (exp.load(md)[0] or {}).get("question")
    except Exception:                                                                # noqa: BLE001
        return None


def _sh(args, timeout=3600):
    env = dict(os.environ, PYTHONPATH=f"{os.path.join(ROOT, 'src')}:{os.path.join(ROOT, 'tools')}")
    try:
        r = subprocess.run([PY] + args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        return r.returncode, "\n".join(l for l in (r.stdout + r.stderr).splitlines() if not _NOISE.search(l)).strip()
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {timeout} s"


def _write_block(md_path, block):
    text = open(md_path).read()
    blk = f"{LAND_OPEN}\n\n## Last landing\n\n{block}\n\n{LAND_CLOSE}"
    if LAND_OPEN in text and LAND_CLOSE in text:
        text = re.sub(re.escape(LAND_OPEN) + r".*?" + re.escape(LAND_CLOSE), lambda _: blk, text, flags=re.S)
    elif "<!-- STATUS -->" in text:
        text = text.replace("<!-- STATUS -->", blk + "\n\n<!-- STATUS -->", 1)
    else:
        text = text.rstrip() + "\n\n" + blk + "\n"
    open(md_path, "w").write(text)


def land(n: int, runs: list[str] | None = None, waited: str = "") -> str:
    t0 = time.time()
    md, d = _exp_dir(n)
    runs = runs if runs is not None else _since_last(n)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    lines = [f"**exp{n:02d} landed {stamp}** -- {len(runs)} run(s)" + (f"; {waited}" if waited else ""), ""]
    lines.append("### Health (trajectory and movie only)")
    flagged = 0
    for r in runs:
        try:
            h = health(r)
        except Exception as e:                                                       # noqa: BLE001
            h = {"run": r, "trajectory": True, "movie": os.path.exists(os.path.join(_gd(), r, "movie.mp4")),
                 "non_finite": [f"health failed: {type(e).__name__}: {e}"]}
        hl = health_lines(h)
        flagged += any(l.strip().startswith("- **") for l in hl)
        lines += hl
    if not runs:
        lines.append("- (no run newer than the last landing)")
    gates = os.path.join(d, "gates.yaml")
    if os.path.exists(gates):
        rc, out = _sh(["tools/exp_gate_score.py", str(n)])
        lines += ["", "### Card (`tools/exp_gate_score.py`)", "```", out[-3500:], "```"]
        rc, out = _sh(["tools/exp_gate_score.py", str(n), "--check"])
        lines += ["", "### Check (`--check`)", "```", out[-2000:], "```"]
    else:
        lines += ["", f"(no {os.path.relpath(gates, ROOT)} -- nothing scored)"]
    rc, out = _sh(["tools/exp_record.py", str(n)], timeout=1800)
    lines += ["", f"Record rebuilt: {out.splitlines()[-1] if out else 'rc ' + str(rc)}"]
    gap = _gap(n, runs, d)
    if gap:
        lines += ["", gap]
    due = time.strftime("%H:%M", time.localtime(time.time() + NEXT_BATCH_MIN * 60))
    lines += ["", f"{flagged} run(s) flagged above. Landing took {int(time.time() - t0)} s.",
              f"**NEXT BATCH DUE by {due}** ({NEXT_BATCH_MIN} min, INSTRUCTION.md \"Ten minutes between batches\"): "
              f"write the rows and findings from THIS report, look at the flagged runs' movies, submit."]
    q = _question(md)
    if q:
        lines += ["", f"**THE QUESTION** -- answer it under `## The question` before the next batch "
                      f"(`- YYYY-MM-DD HH:MM -- ...`, >= 25 words; tools/exp.py launch checks): {q}"]
    text = "\n".join(lines)
    os.makedirs(os.path.join(d, "landings"), exist_ok=True)
    open(os.path.join(d, "landings", time.strftime("%Y-%m-%d_%H%M") + ".md"), "w").write(text + "\n")
    try:
        _write_block(md, text)
    except Exception as e:                                                           # noqa: BLE001
        text += f"\n(could not write the LANDING block into {md}: {e})"
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", type=int, nargs="?")
    ap.add_argument("--runs", nargs="*", help="<group>/<name> runs (default: every run newer than the last landing)")
    ap.add_argument("--health", nargs="*", help="print these runs' health only; nothing is scored or written")
    a = ap.parse_args()
    if a.health:
        for r in a.health:
            print("\n".join(health_lines(health(r))))
        return 0
    if a.n is None:
        ap.error("an experiment number, or --health RUN")
    print(land(a.n, a.runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
