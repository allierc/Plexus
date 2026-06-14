"""Reproduce an archived scenario -- optionally with a NEW seed -- and re-render.

The archive stores the exact spec + seed, so a run is fully reproducible; pass
--seed to draw a different realization of the same dynamics, and --gif/--dot to
visualize it however you like. Output zarr can be fed to any visualizer.

    python reproduce.py aggregate                 # exact reproduction
    python reproduce.py aggregate --seed 7        # new realization
    python reproduce.py aggregate --seed 7 --gif out.gif --dot 0.4
"""

import os
import sys
import torch

from scenario_schema import load
import engine2

HERE = os.path.dirname(__file__)


def _resolve(name):
    for p in (os.path.join(HERE, "scenarios", "archive", name, "scenario.yaml"),
              os.path.join(HERE, "scenarios", f"{name}.yaml"),
              name):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no scenario named {name!r} in archive or scenarios/")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0]
    sc = load(_resolve(name))
    if "--seed" in sys.argv:
        sc.seed = int(sys.argv[sys.argv.index("--seed") + 1])
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"/tmp/{name}_seed{sc.seed}.zarr"
    device = "cpu" if "--cpu" in sys.argv else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[reproduce] {name} seed={sc.seed} device={device}")
    engine2.run(sc, out, device=device)
    print(f"[done] {out}")
    if "--gif" in sys.argv:
        gif = sys.argv[sys.argv.index("--gif") + 1]
        dot = sys.argv[sys.argv.index("--dot") + 1] if "--dot" in sys.argv else "0.5"
        os.system(f"PYTHONPATH={os.environ.get('PYTHONPATH','')} {sys.executable} "
                  f"{os.path.join(HERE,'viz2.py')} {out} {gif} --dot {dot}")


if __name__ == "__main__":
    main()
