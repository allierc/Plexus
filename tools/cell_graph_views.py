"""One top-down PNG per relation: the graph brought forward, everything else faded back.

WHY THESE ARE NOT IN THE MOVIE. The overlay drew each relation over the movie's closing frames,
which puts a tissue-scale graph on top of a picture whose camera and opacities were chosen for the
mechanics -- seen obliquely, through fourteen other compartments, at whatever transparency the
membrane happened to need. A relation between cells is a PLAN, and it reads from above with the
organelle it belongs to opaque and the rest of the cell pushed back.

So this is a separate pass over the finished run:

    one PNG per set in `plotting.graph_sets`
    camera looking straight down the up axis
    the graph's own organelle at full opacity, every other compartment at `--fade`
    edges in that organelle's colour from `plotting.colors`

It reads the RUN, not the spec's initial condition -- the last recorded frame of
`trajectory.npz` -- so the relation is the one the cells ended in. The positions of an organelle
set are only meaningful there because `aggregate_centroid` kept them tracking their own material
points; without it this would draw a graph over frame-0 seed positions and look plausible.

    python tools/cell_graph_views.py --data <run dir> --spec cell/cell_atlas_25_sets
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.paths import resolve_config  # noqa: E402


def radius_edges(P, r, cap=200000):
    """Pairs closer than `r`, each once. A grid bucketing, so it does not build an N^2 matrix."""
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    pairs = tree.query_pairs(r, output_type="ndarray")
    if pairs.shape[0] > cap:
        pairs = pairs[:: pairs.shape[0] // cap + 1]
    return pairs


def cell_diameter(spec):
    """A cell's diameter, READ FROM THE SPEC rather than measured off the cloud.

    The first version estimated it from the membrane's bounding box divided by a cube root of the
    cell count, which for nine cells spread through the box gave 0.43 -- larger than the cell pitch
    it was being compared against, so `auto` always chose per_piece and the guard never fired. The
    number is stated exactly in the model: every organelle set is scattered in a ball of `radius`
    about its cell.
    """
    for s_ in (spec.get("sets") or {}).values():
        if isinstance(s_, dict) and s_.get("parent") == "cell" and "radius" in s_:
            return 2.0 * float(s_["radius"])
    return float("inf")


def cell_pitch(traj):
    """Mean nearest-neighbour distance between CELL centres -- 'about nucleus-nucleus distance'."""
    C = traj.get("cell__pos")
    if C is None or C.shape[1] < 2:
        return None
    P = np.asarray(C[-1], np.float64)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the run directory holding trajectory.npz")
    ap.add_argument("--spec", default=None, help="the spec, for colours and graph_sets")
    ap.add_argument("--radius", type=float, default=None,
                    help="cutoff in world units; default is the mean cell-to-cell distance")
    ap.add_argument("--radius-um", type=float, default=None,
                    help="the cutoff in MICROMETRES, converted with the spec's own `length_um`. "
                         "The world-unit form needs the reader to do that division, which is "
                         "where a cutoff ends up an order of magnitude out.")
    ap.add_argument("--fade", type=float, default=0.06,
                    help="opacity of every compartment that is NOT the graph's own")
    ap.add_argument("--dot", type=float, default=1.2)
    ap.add_argument("--edge-width", type=float, default=1.4)
    ap.add_argument("--edge-opacity", type=float, default=0.85)
    ap.add_argument("--px", type=int, default=1600)
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--sets", default="",
                    help="comma-separated set names, overriding the spec's `graph_sets`")
    ap.add_argument("--color-floor", type=float, default=0.78,
                    help="lift any organelle colour whose brightest channel is below this. The "
                         "palette is chosen for SURFACES lit in a dark scene; the same value as "
                         "an unlit line on black is a different picture, and the rough ER's dark "
                         "teal came out unreadable.")
    ap.add_argument("--balance", type=float, default=0.5,
                    help="how hard the combined view fades a dense relation: opacity scales by "
                         "(fewest edges / this set's edges) ** balance. 0 = no correction, "
                         "1 = every set lays down the same total ink.")
    ap.add_argument("--no-combined", action="store_true",
                    help="skip the graph_all.png that draws every relation at once")
    ap.add_argument("--nodes", default="auto", choices=("auto", "per_cell", "per_piece"),
                    help="what a NODE of the relation is. `per_piece` is every mitochondrion; "
                         "`per_cell` is one node per cell, the centroid of that cell's pieces. "
                         "`auto` picks per_cell when the radius exceeds a cell diameter, because "
                         "at that cutoff per_piece is a complete graph inside every cell.")
    args = ap.parse_args()

    z = np.load(os.path.join(args.data, "trajectory.npz"))
    spec_path = args.spec or os.path.join(args.data, "spec.yaml")
    if not os.path.isfile(spec_path):
        spec_path = resolve_config(spec_path)[0]
    spec = yaml.safe_load(open(spec_path))
    pl = spec.get("plotting") or {}
    gsets = ([v.strip() for v in args.sets.split(",") if v.strip()] or
             pl.get("graph_sets") or (pl.get("graph_overlay") or {}).get("sets") or [])
    if not gsets:
        raise SystemExit("the spec names no `plotting.graph_sets`")
    colors = pl.get("colors") or {}
    up = int(pl.get("up_axis", 2))

    um = float(((spec.get("general") or {}).get("units") or {}).get("length_um", 0)) or None
    if args.radius_um is not None:
        if not um:
            raise SystemExit("--radius-um needs the spec to declare `general.units.length_um`")
        r = args.radius_um / um
    else:
        r = args.radius or cell_pitch(z)
    if r is None:
        raise SystemExit("no cell set with >1 element, so there is no cell-to-cell distance; "
                         "give --radius explicitly")
    print(f"[graphs] radius {r:.5f} world" + (f" = {r * um:.1f} um" if um else ""), flush=True)

    from plexus.render_vtk import offscreen
    offscreen()
    import pyvista as pv
    from matplotlib.colors import to_rgb
    pv.OFF_SCREEN = True
    pv.global_theme.allow_empty_mesh = True

    world = [float(v) for v in (z["world_size"] if "world_size" in z.files else [1, 1, 1])]
    node_sets = [k[:-5] for k in z.files if k.endswith("__pos") and k[:-5].endswith("_node")]

    def lift(c):
        """Scale a colour so its brightest channel reaches `--color-floor`, hue unchanged."""
        m = max(c)
        return c if m >= args.color_floor or m <= 0 else tuple(v * args.color_floor / m for v in c)

    # ---- every relation, computed once ------------------------------------------------------ #
    rel = []
    for g in gsets:
        key = f"{g}__pos"
        if key not in z.files:
            print(f"[graphs] {g}: not recorded, skipped", flush=True)
            continue
        P = np.asarray(z[key][args.frame], np.float64)
        # WHAT A NODE IS, AND WHY IT HAS TO CHANGE WITH THE RADIUS. At a cutoff of the cell-to-cell
        # distance every one of a cell's 77 mitochondria is within reach of every other, so
        # per_piece is a COMPLETE graph per cell -- 49,000 edges that render as a solid block. A
        # cutoff that size is a question about CELLS, so the node becomes the cell.
        mode = args.nodes
        if mode == "auto":
            mode = "per_cell" if r > cell_diameter(spec) else "per_piece"
        n_piece = P.shape[0]
        if mode == "per_cell" and f"{g}__parent" in z.files:
            par = np.asarray(z[f"{g}__parent"]).reshape(-1).astype(np.int64)
            nc = int(par.max()) + 1
            C = np.zeros((nc, 3)); cnt = np.zeros(nc)
            np.add.at(C, par, P); np.add.at(cnt, par, 1.0)
            P = C / np.maximum(cnt, 1)[:, None]
        pairs = radius_edges(P, r)
        col = lift(to_rgb(tuple(colors[g])) if g in colors else (0.9, 0.9, 0.9))
        rel.append(dict(name=g, P=P, pairs=pairs, col=col, mode=mode, n_piece=n_piece))
        print(f"[graphs] {g:20s} {P.shape[0]:>7,} nodes ({mode})  "
              f"{pairs.shape[0]:>9,} edges", flush=True)

    def render(subject, out_name, title, nodes_only=False):
        """`subject` is the list of relations drawn at full strength; all else is faded.

        `nodes_only` draws the GRAPH and nothing else -- the relation's own nodes and its edges,
        with no material points behind them. For a single organelle the cell is useful context;
        for the combined view it is three organelles' worth of context behind three graphs, and
        the graphs are the subject.

        THE DENSE RELATION IS DRAWN FAINTER, in proportion to how much of it there is. Mitochondria
        contribute 13,221 of the combined view's 14,002 edges -- 94% -- so at one opacity for all
        three the picture is the mitochondrial graph with two others lost inside it. Each set's
        edge opacity is scaled by `(fewest edges / its edges) ** balance`: at balance 1 every set
        lays down the same total ink and the mitochondria vanish; at 0 nothing is corrected. The
        square root is the usable middle.
        """
        loud = {d["name"] for d in subject}
        p = pv.Plotter(off_screen=True, window_size=(args.px, args.px), border=False)
        p.set_background("black")
        p.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)
        lo_e = max(1, min(d["pairs"].shape[0] for d in subject))
        if not nodes_only:
            for ns in node_sets:
                base = ns[:-5]
                if base in loud:
                    continue
                Q = np.asarray(z[f"{ns}__pos"][args.frame], np.float32)
                step = max(1, Q.shape[0] // 400_000)
                c = to_rgb(tuple(colors[base])) if base in colors else (0.5, 0.5, 0.5)
                p.add_mesh(pv.PolyData(Q[::step]), color=c, opacity=args.fade,
                           render_points_as_spheres=True, point_size=args.dot * 0.8,
                           lighting=False, show_scalar_bar=False)
        for d in subject:
            n_e = max(1, d["pairs"].shape[0])
            eop = args.edge_opacity
            if nodes_only and len(subject) > 1:
                eop = float(np.clip(eop * (lo_e / n_e) ** args.balance, 0.05, 1.0))
            if nodes_only:
                # the graph's own NODES -- the organelle centroids the relation is between
                p.add_mesh(pv.PolyData(np.asarray(d["P"], np.float32)), color=d["col"],
                           opacity=1.0, render_points_as_spheres=True,
                           point_size=args.dot * 3.0, lighting=False, show_scalar_bar=False)
            else:
                own = f"{d['name']}_node__pos"
                if own in z.files:
                    Q = np.asarray(z[own][args.frame], np.float32)
                    step = max(1, Q.shape[0] // 400_000)
                    p.add_mesh(pv.PolyData(Q[::step]), color=d["col"], opacity=1.0,
                               render_points_as_spheres=True, point_size=args.dot,
                               lighting=False, show_scalar_bar=False)
            if d["pairs"].shape[0]:
                P, pr = d["P"], d["pairs"]
                pts = np.concatenate([P[pr[:, 0]], P[pr[:, 1]]], 0).astype(np.float32)
                m = pr.shape[0]
                lines = np.column_stack([np.full(m, 2, np.int64), np.arange(m),
                                         np.arange(m, 2 * m)]).ravel()
                pd = pv.PolyData(pts); pd.lines = lines
                p.add_mesh(pd, color=d["col"], line_width=args.edge_width,
                           opacity=eop, lighting=False, show_scalar_bar=False)
        centre = np.array(world) / 2.0
        eye = centre.copy(); eye[up] += max(world) * 3.0
        p.camera.position = tuple(eye)
        p.camera.focal_point = tuple(centre)
        vup = np.zeros(3); vup[(up + 1) % 3] = 1.0
        p.camera.up = tuple(vup)
        p.camera.parallel_projection = True
        p.camera.parallel_scale = max(world) * 0.52
        p.add_text(title, position="upper_left", font_size=12, color="white")
        out = os.path.join(args.data, out_name)
        p.screenshot(out)
        p.close()
        print(f"[graphs] -> {out}", flush=True)

    _r = (f"{r * um:.1f} µm" if um else f"{r:.4f}")
    for d in rel:
        render([d], f"graph_{d['name']}.png",
               f"{d['name']}   {d['P'].shape[0]:,} nodes ({d['mode'].replace('_', ' ')}, from "
               f"{d['n_piece']:,} pieces)   {d['pairs'].shape[0]:,} edges at r = {_r}")
    if rel and not args.no_combined:
        render(rel, "graph_all.png",
               "   ".join(f"{d['name']} {d['pairs'].shape[0]:,}" for d in rel)
               + f"   {sum(d['pairs'].shape[0] for d in rel):,} edges at r = {_r}"
               + f"   (opacity balanced, exponent {args.balance:g})",
               nodes_only=True)


if __name__ == "__main__":
    main()
