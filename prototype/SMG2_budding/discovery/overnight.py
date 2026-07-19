"""overnight -- run the mechanism-space discovery loop autonomously for a wall-clock budget.

Each round samples NEW (parameter, seed) points (seed_offset = round) that ACCUMULATE into one
append-only archive, so the emergence / necessity / robustness estimates tighten as evidence grows;
the exploration also DEEPENS over time (node cap and parameter basin grow). After every round the
knowledge ledger is re-distilled and a timestamped line is appended to `analysis_log.md`. Durable
files only -- no auto-commit; every round's evidence is already persisted, so it is Ctrl-C / crash safe.

  python discovery/overnight.py [--hours 8 --max_rounds 200 --metric metric_v1 --max_stage 3]
"""
from __future__ import annotations
import os, sys, time, argparse, datetime, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "pf"))
import loop1_explore as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--max_rounds", type=int, default=200)
    ap.add_argument("--basin", type=int, default=2)
    ap.add_argument("--param_basin", type=int, default=3)
    ap.add_argument("--node_cap", type=int, default=24)
    ap.add_argument("--max_stage", type=int, default=3)
    ap.add_argument("--metric", default="metric_v1")
    a = ap.parse_args()

    out = os.path.join(HERE, "_archive_overnight")
    logf = os.path.join(HERE, "analysis_log.md")
    if not os.path.exists(logf):
        with open(logf, "w") as f:
            f.write("# Overnight discovery log — accumulating mechanism-space evidence\n\n"
                    "> One line per round. The archive (`_archive_overnight/`) is the source of truth; "
                    "`knowledge.md` is the current distilled ledger.\n\n")

    phi0 = os.path.join(HERE, "..", "pf", "_real", "phi0.npy")
    start = time.time(); r = 0
    while (time.time() - start) < a.hours * 3600 and r < a.max_rounds:
        if not os.path.exists(phi0):                            # branch switched away? pause, don't error-spin
            print("[pause] phi0.npy absent (branch switched?) — waiting 60s", flush=True)
            time.sleep(60); continue
        node_cap = a.node_cap + (r // 4) * 4                    # deepen the search every 4 rounds
        pb = a.param_basin + (r // 12)                          # denser parameter basin over time
        t0 = time.time()
        try:
            st = L.explore(basin=a.basin, node_cap=node_cap, max_stage=a.max_stage, param_basin=pb,
                           metric=a.metric, seed_offset=r + 1, out=out)
            line = (f"- round {r:03d}: {st['n_records']} records · {st['n_comps']} comps · "
                    f"Established {st['established']} · Structural {st['structural']} · Open {st['open']} "
                    f"· metric={a.metric} node_cap={node_cap} pb={pb} · {time.time()-t0:.0f}s")
        except Exception as e:                                  # never let one round kill the night
            line = f"- round {r:03d}: ERROR {type(e).__name__}: {str(e)[:80]}"
            traceback.print_exc()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(logf, "a") as f:
            f.write(f"{line} · {stamp}\n")
        print(line, flush=True)
        r += 1
        time.sleep(1)

    print(f"\n=== overnight done: {r} rounds in {(time.time()-start)/3600:.1f} h ===", flush=True)
    print(f"archive: {out}   ledger: {os.path.join(HERE,'knowledge.md')}   log: {logf}")


if __name__ == "__main__":
    main()
