"""One still of a cell spec at t = 0 -- to look at a render choice before it costs cluster hours.

`Plexus_Main.py -o generate` is the way to make a MOVIE; it is a poor way to answer "does this
palette work", because the answer is in the first frame and the question costs a whole run. This
seeds the state and renders exactly one frame.

    python tools/cell_still.py --spec cell/cell_atlas_25 --out /tmp/a.png
    python tools/cell_still.py --spec cell/cell_atlas_cutaway --divide 1
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import plexus.operators  # noqa: F401,E402
from plexus.schema import load  # noqa: E402
from plexus.engine import build, seed  # noqa: E402
from plexus.paths import resolve_config  # noqa: E402
from cell_opacity_montage import panel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="cell/cell_atlas_25")
    ap.add_argument("--out", default=os.path.join(ROOT, "graphs_data", "cell", "cell_still.png"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--divide", type=int, default=4)
    ap.add_argument("--n-grid", type=int, default=None)
    ap.add_argument("--px", type=int, default=1400)
    args = ap.parse_args()

    yaml_file, _pre, _name = resolve_config(args.spec)
    raw = yaml.safe_load(open(yaml_file))
    if args.divide > 1:
        pp = raw["sets"]["mpm_particle"]["per_parent"]
        raw["sets"]["mpm_particle"]["per_parent"] = (
            {k: max(3, v // args.divide) for k, v in pp.items()} if isinstance(pp, dict)
            else max(3, int(pp) // args.divide))
    if args.n_grid:
        for f in raw["fields"].values():
            if "n_grid" in f:
                f["n_grid"] = args.n_grid
    raw["general"]["name"] += "_still"
    raw["plotting"]["render_px"] = args.px
    tmp = os.path.join(tempfile.mkdtemp(prefix="still_"), "spec.yaml")
    yaml.safe_dump(raw, open(tmp, "w"), sort_keys=False)

    sim = load(tmp)
    H = build(sim, device=args.device)
    seed(H, sim, device=args.device)
    img = panel(H, sim, dict(sim.plotting or {}), tempfile.mkdtemp(prefix="still_p_"), args.px)
    if img is None:
        raise RuntimeError("the renderer produced no image")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import imageio.v3 as iio
    iio.imwrite(args.out, img)
    print(f"[still] {img.shape[1]}x{img.shape[0]} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
