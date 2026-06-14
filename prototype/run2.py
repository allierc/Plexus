"""Entrypoint for the 2-level MPM scenario.

    python run2.py [scenario_mpm.yaml] [out.zarr] [--cpu]
"""

import sys
import torch

from scenario_schema import load
from engine2 import run


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    scenario = args[0] if len(args) > 0 else "scenario_mpm.yaml"
    out = args[1] if len(args) > 1 else "/tmp/tissue_mpm.zarr"
    device = "cpu" if "--cpu" in sys.argv else ("cuda" if torch.cuda.is_available() else "cpu")
    compile_mpm = "--compile" in sys.argv

    sc = load(scenario)
    Nc = int(sc.sets["cell"]["n"]); ppc = int(sc.sets["particle"]["per_parent"])
    print(f"[spec ok] {sc.name}: {Nc} cells x {ppc} particles = {Nc*ppc} particles, "
          f"{sc.n_frames} frames, device={device}")
    path, _ = run(sc, out, device=device, compile_mpm=compile_mpm)
    print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
