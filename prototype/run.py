"""Entrypoint: load + validate a scenario, simulate, write zarr.

    python run.py [scenario.yaml] [out.zarr]
"""

import sys

from scenario_schema import load
from engine import run


def main():
    scenario_path = sys.argv[1] if len(sys.argv) > 1 else "scenario.yaml"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/tissue_proto.zarr"

    sc = load(scenario_path)                       # raises on any spec error
    print(f"[spec ok] {sc.name}: {sum(int(s['n']) for s in sc.sets.values())} nodes, "
          f"{len(sc.operators)} operators, {sc.n_frames} frames")
    path, _ = run(sc, out_path)
    print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
