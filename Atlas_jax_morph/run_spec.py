"""run_spec -- execute ONE Plexus spec and record it as atlas evidence.

The atlas's claim is that a mechanism read out of someone else's code became a Plexus operator
that WORKS. A record entry cannot carry that claim; only a run can. So every claim of that kind
ends here, in an evidence folder shaped exactly like the discovery track's
(`log/okuda/coral_fixed_ball/`), so the two campaigns can be read side by side:

    log/atlas/<name>/
        spec_run.yaml   the spec exactly as it ran
        diag.json       config, summary, the ACTED LEDGER, wall clock, run id
        metrics.json    summary + the per-frame series
        metrics.npz     the arrays, so a new observable can re-score without re-simulating
        strip.png       panels on ONE common spatial scale
        movie.mp4

THE ACTED LEDGER IS THE POINT. `run_one.py` earned this the hard way: *a run in which a scheduled
operator never acted is not evidence.* An operator can be in the schedule, be instantiated, be
called every frame, and return a zero delta forever -- because its gate never opened, its
parameter is zero, or it reads a field nobody writes. The run completes, the movie looks fine, and
the conclusion is about a model that was never running. Here every operator is wrapped, every call
is counted, and an operator that never moved anything is named in `diag.json` under
`inert_operators` and printed in red at the end.

ONE FRAME SIZE FOR THE WHOLE RUN. The strip is drawn on a single spatial scale taken from the last
frame. The discovery campaign spent a day reading growth as shrinkage because each panel was
auto-scaled to its own contents.

    python run_spec.py <name>              # config/atlas/<name>.yaml
    python run_spec.py <path.yaml> --device cuda:0 --no-movie
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))

CONFIG_DIR = os.path.join(PLEXUS, "config", "atlas")
LOG_DIR = os.path.join(PLEXUS, "log", "atlas")


# ------------------------------------------------------------------------------------------- #
#  the acted ledger
# ------------------------------------------------------------------------------------------- #
def install_acted_ledger():
    """Wrap every operator the engine instantiates so we can tell doing from being scheduled.

    Returns the ledger dict, live. Implemented by patching the engine's `get_operator` rather
    than the engine loop: the wrapper is a subclass, so every capability check the engine makes
    against the class (EMIT, INTEGRAND, MAY_MUTATE_INTEGRATED_STATE) still sees the real thing.
    """
    import torch

    from plexus import engine

    ledger: dict[str, dict] = {}
    real = engine.get_operator

    def watched(name, impl=None):
        cls = real(name, impl)

        class Watched(cls):                       # noqa: N801 -- a decorator would re-register
            def forward(self, H, mask=None):
                rec = ledger.setdefault(name, {"calls": 0, "acted": 0, "moved": 0.0,
                                               "structural": bool(
                                                   getattr(cls, "MAY_MUTATE_INTEGRATED_STATE",
                                                           False))})
                before = {n: int(l.active.sum()) for n, l in H.levels.items()}
                out = super().forward(H, mask)
                rec["calls"] += 1
                moved = 0.0
                for d in (out or {}).values():
                    if isinstance(d, torch.Tensor) and d.numel():
                        moved = max(moved, float(d.abs().max()))
                after = {n: int(l.active.sum()) for n, l in H.levels.items()}
                if moved > 0 or before != after:
                    rec["acted"] += 1
                    rec["moved"] = max(rec["moved"], moved)
                return out

        Watched.__name__ = f"Watched{cls.__name__}"
        return Watched

    engine.get_operator = watched
    return ledger


def inert(ledger):
    """Operators that ran and never moved anything.

    A `structural`/readout operator is excluded: it legitimately writes state directly and
    returns no delta, so silence there is not evidence of a no-op. It is reported separately
    rather than quietly counted as fine -- an unverifiable operator is its own category.
    """
    return sorted(n for n, r in ledger.items()
                  if r["calls"] > 0 and r["acted"] == 0 and not r["structural"])


def unverifiable(ledger):
    return sorted(n for n, r in ledger.items() if r["structural"] and r["acted"] == 0)


# ------------------------------------------------------------------------------------------- #
#  metrics -- generic, and the same for every atlas run
# ------------------------------------------------------------------------------------------- #
def series_for(pos, occ):
    """Per-frame observables for one set. `pos` [T,N,D], `occ` [T,N] (0/1)."""
    T = pos.shape[0]
    n, gyr, cen, ext, nnd = [], [], [], [], []
    for t in range(T):
        live = occ[t].astype(bool)
        p = pos[t][live]
        n.append(int(live.sum()))
        if len(p) < 2:
            gyr.append(0.0), cen.append(0.0), ext.append(0.0), nnd.append(0.0)
            continue
        c = p.mean(0)
        gyr.append(float(np.sqrt(((p - c) ** 2).sum(1).mean())))
        cen.append(float(np.linalg.norm(c)))
        ext.append(float((p.max(0) - p.min(0)).max()))
        q = p if len(p) <= 400 else p[np.random.default_rng(0).choice(len(p), 400, replace=False)]
        d = np.linalg.norm(q[:, None] - q[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        nnd.append(float(d.min(1).mean()))
    return {"n_active": n, "gyration": gyr, "centroid_norm": cen, "extent": ext,
            "nn_distance": nnd}


def summarize(all_series):
    s = {}
    for sname, ser in all_series.items():
        for k, v in ser.items():
            if not v:
                continue
            s[f"{sname}_{k}_first"] = round(float(v[0]), 6)
            s[f"{sname}_{k}_final"] = round(float(v[-1]), 6)
            s[f"{sname}_{k}_peak"] = round(float(np.max(v)), 6)
    return s


# ------------------------------------------------------------------------------------------- #
#  the strip -- one spatial scale for the whole run
# ------------------------------------------------------------------------------------------- #
def strip(pos, occ, out_png, n_panels=6, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = pos.shape[0]
    picks = np.linspace(0, T - 1, n_panels).astype(int)
    live = occ[-1].astype(bool)
    p_last = pos[-1][live]
    if len(p_last) == 0:
        return None
    c, r = p_last.mean(0), max(1e-6, np.abs(p_last - p_last.mean(0)).max() * 1.15)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.5 * n_panels, 2.8), facecolor="black")
    for ax, t in zip(np.atleast_1d(axes), picks):
        m = occ[t].astype(bool)
        q = pos[t][m]
        ax.set_facecolor("black")
        if len(q):
            ax.scatter(q[:, 0], q[:, 1], s=6, c="#4FA3FF", linewidths=0)
        ax.set_xlim(c[0] - r, c[0] + r)          # ONE scale, from the last frame, for every panel
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#444444")
        ax.text(0.03, 0.97, f"t={t}   n={int(m.sum())}", transform=ax.transAxes, color="white",
                fontsize=9, va="top", ha="left")
    if title:
        fig.text(0.005, 0.99, title, color="white", fontsize=11, va="top", ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, facecolor="black")
    plt.close(fig)
    return out_png


# ------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="config/atlas/<name>.yaml, or a path to a spec")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()

    path = a.name if a.name.endswith(".yaml") else os.path.join(CONFIG_DIR, a.name + ".yaml")
    if not os.path.exists(path):
        raise SystemExit(f"no spec at {path}")

    ledger = install_acted_ledger()               # BEFORE the engine builds anything
    import plexus.operators  # noqa: F401
    from plexus.generators.graph_data_generator import data_generate
    from plexus.schema import load

    sim = load(path)
    out_dir = os.path.join(LOG_DIR, sim.name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(path, os.path.join(out_dir, "spec_run.yaml"))

    t0 = time.time()
    data_dir, out = data_generate(sim, "atlas", device=a.device, erase=True)
    wall = time.time() - t0

    all_series, arrays = {}, {}
    for sname, d in out["sets"].items():
        if d.get("pos") is None:
            continue
        all_series[sname] = series_for(np.asarray(d["pos"]), np.asarray(d["occ"]))
        for k, v in all_series[sname].items():
            arrays[f"{sname}__{k}"] = np.asarray(v)

    summary = summarize(all_series)
    summary["frames"] = int(sim.n_frames)
    summary["wall_s"] = round(wall, 1)

    dead = inert(ledger)
    unv = unverifiable(ledger)
    run_id = "A" + hashlib.sha1(
        (sim.name + json.dumps(summary, sort_keys=True)).encode()).hexdigest()[:15]
    diag = {"config": sim.name, "spec": os.path.relpath(path, PLEXUS), "run_id": run_id,
            "device": a.device, "summary": summary,
            "acted": {n: r for n, r in sorted(ledger.items())},
            "inert_operators": dead, "unverifiable_operators": unv,
            "valid_evidence": not dead,
            "why": ("an operator ran and never moved anything" if dead else "every scheduled "
                    "operator acted at least once")}

    with open(os.path.join(out_dir, "diag.json"), "w") as f:
        json.dump(diag, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"summary": summary, "series": all_series}, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "metrics.npz"), **arrays)

    first = next(iter(out["sets"]))
    strip(np.asarray(out["sets"][first]["pos"]), np.asarray(out["sets"][first]["occ"]),
          os.path.join(out_dir, "strip.png"), title=f"{sim.name} · {first}")

    if not a.no_movie:
        from plexus.plot import plot_dataset
        plot_dataset(sim, "atlas", movie=True)
        for fn in sorted(os.listdir(data_dir)):
            if fn.endswith(".mp4"):
                shutil.copyfile(os.path.join(data_dir, fn), os.path.join(out_dir, "movie.mp4"))
                break

    print(f"\n{sim.name}: {summary['frames']} frames in {wall:.1f}s -> {out_dir}")
    for n, r in sorted(ledger.items()):
        mark = "INERT" if n in dead else ("unverifiable" if n in unv else "ok")
        print(f"  {n:<28} calls {r['calls']:>5}  acted {r['acted']:>5}  "
              f"max|delta| {r['moved']:.3g}   {mark}")
    if dead:
        print(f"\n  NOT EVIDENCE: {', '.join(dead)} never moved anything. Fix the spec or the "
              f"operator before reading any number above.")
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
