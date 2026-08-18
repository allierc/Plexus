
"""Dense visualization of Eye G local muscle fibre axes.

Run from:
    Plexus/prototype/eye/archive/eye_G

Examples
--------
Eye G surface-render camera:

    python viz_muscle_fibres_dense.py

Only MR / IO / SR:

    python viz_muscle_fibres_dense.py \
        --muscles MR IO SR

Dense:

    python viz_muscle_fibres_dense.py \
        --n-lines 700

Thinner:

    python viz_muscle_fibres_dense.py \
        --n-lines 700 \
        --line-scale 0.035 \
        --line-width 0.7

Use the exact Eye G surface-view convention but change the angle:

    python viz_muscle_fibres_dense.py \
        --az 16 \
        --el 10

Output:
    viz_muscle_fibres_dense.png


WHAT IS BEING PLOTTED
---------------------
The MPM particles themselves are NOT shown.

For every muscle particle p, the local fibre gradient is reconstructed from
the saved material coordinate s:

    (x_j - x_p) . g_p ~= s_j - s_p

The Eye G implementation forms:

    A_p = sum_j dx_j dx_j^T + ridge I

    b_p = sum_j dx_j ds_j

    g_p = A_p^{-1} b_p

and normalizes:

    f_p = g_p / ||g_p||.

The repository construction uses:

    k = 12
    ridge = 1e-9

The rendered lines are short bidirectional line segments centred on the
corresponding MPM particle.

Their colour is the material coordinate s.

The line has no physical arrow direction because:

    (-f_p)(-f_p)^T = f_p f_p^T.

So the visualization shows the fibre AXIS, not a directed arrow.

IMPORTANT
---------
This script reconstructs the fibre directions from:

    mus_pos + mus_s

in baseline_curves.npz.

It does not use a saved fibre field.

CAMERA
------
The default camera follows the Eye G surface-render convention:

    azimuth   = 16 degrees
    elevation = 10 degrees
    up        = +Y
    orthographic projection
    target    = eye centre

Use --az and --el to change the view.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv


MUSCLE_NAMES = [
    "LR",
    "SR",
    "MR",
    "IR",
    "SO",
    "IO",
]


# ---------------------------------------------------------------------------
# Exact local fibre construction used by Eye G
# ---------------------------------------------------------------------------

def compute_repo_fibres(
    pts: np.ndarray,
    s: np.ndarray,
    k: int = 12,
    ridge: float = 1e-9,
) -> np.ndarray:
    """Reproduce the local fibre construction used by Eye G."""

    pts = np.asarray(
        pts,
        dtype=float,
    )

    s = np.asarray(
        s,
        dtype=float,
    )

    n = len(pts)

    if n < 3:
        return np.zeros(
            (n, 3),
            dtype=float,
        )

    k = min(
        max(int(k), 3),
        n,
    )

    tree = cKDTree(pts)

    _, idx = tree.query(
        pts,
        k=k,
    )

    # Remove self.
    neighbours = idx[:, 1:]

    dx = (
        pts[neighbours]
        - pts[:, None, :]
    )

    ds = (
        s[neighbours]
        - s[:, None]
    )

    # A = sum dx dx^T + ridge I
    A = np.einsum(
        "nki,nkj->nij",
        dx,
        dx,
    )

    A += (
        ridge
        * np.eye(3)[None, :, :]
    )

    # b = sum dx ds
    b = np.einsum(
        "nki,nk->ni",
        dx,
        ds,
    )

    # g = A^-1 b
    g = np.linalg.solve(
        A,
        b[..., None],
    )[..., 0]

    # f = g / ||g||
    norm = np.linalg.norm(
        g,
        axis=1,
        keepdims=True,
    )

    fibre = (
        g
        / np.clip(
            norm,
            1e-12,
            None,
        )
    )

    return fibre


def smooth_material_coord(
    pts: np.ndarray,
    s: np.ndarray,
    k: int = 12,
    iters: int = 8,
    lam: float = 0.5,
) -> np.ndarray:
    """Smooth the material coordinate s, not the fibre field f.

    f = grad(s) / ||grad(s)||, so a noisy s gives a noisy f, and smoothing f
    afterwards patches the symptom in vector space (three noisy numbers) rather
    than the cause (one noisy scalar). This smooths first:

        s -> s~ -> grad(s~) -> f~ = grad(s~) / ||grad(s~)||

    which is `compute_repo_fibres` called on s~ instead of s -- the same
    construction, just fed a cleaner field.

    s~ is obtained by k-NN GRAPH Laplacian smoothing: `iters` rounds of
    averaging each point toward its `k` nearest neighbours in space,

        s <- (1 - lam) s + lam * mean_{j in kNN(p)} s_j

    using the SAME neighbourhood `compute_repo_fibres` differentiates over,
    not an isotropic Gaussian blur. That distinction matters here: a Euclidean
    blur radius can reach across a fold in the strap, where two points close in
    space sit far apart along the fibre, and would blend their very different s
    values into a wrong local gradient. The kNN graph is built from the same
    "nearest in space" relation as the differentiation, so it smooths along the
    geometry the gradient is already trusted to respect.

    The endpoint constraint -- s = 0 at the origin, s = 1 at the insertion --
    is not enforced by pinning particular particles (there is no saved label
    for "the" origin/insertion point, only the scalar field), but by an affine
    RESCALE after smoothing so min(s~) = 0 and max(s~) = 1 exactly, whatever
    smoothing did to the interior. Smoothing can only pull extreme values
    toward the mean, never past it, so the arg-min/arg-max are unchanged in
    identity, only their neighbourhood is denoised.
    """

    pts = np.asarray(pts, dtype=float)
    s = np.asarray(s, dtype=float)
    n = len(pts)

    if n < 3 or iters <= 0:
        return s

    k = min(max(int(k), 3), n)

    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k)
    neighbours = idx[:, 1:]           # drop self

    s_smooth = s.copy()
    for _ in range(int(iters)):
        s_smooth = (1.0 - lam) * s_smooth + lam * s_smooth[neighbours].mean(axis=1)

    lo, hi = s_smooth.min(), s_smooth.max()
    if hi - lo < 1e-12:
        return s
    return (s_smooth - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Capture loading
# ---------------------------------------------------------------------------

def load_capture(
    path: str,
    frame: int,
):
    """Load mus_pos, mus_parent and exact mus_s."""

    z = np.load(
        path,
        allow_pickle=True,
    )

    required = (
        "mus_pos",
        "mus_parent",
        "mus_s",
    )

    missing = [
        key
        for key in required
        if key not in z.files
    ]

    if missing:
        raise RuntimeError(
            f"{path} is missing {missing}.\n"
            "Expected mus_pos, mus_parent and mus_s."
        )

    mus_pos = np.asarray(
        z["mus_pos"],
        dtype=float,
    )

    if mus_pos.ndim == 3:

        frame = min(
            max(frame, 0),
            mus_pos.shape[0] - 1,
        )

        mus_pos = mus_pos[
            frame
        ]

    else:

        frame = 0

    mus_parent = np.asarray(
        z["mus_parent"],
        dtype=int,
    ).reshape(-1)

    mus_s = np.asarray(
        z["mus_s"],
        dtype=float,
    )

    if mus_s.ndim == 2:

        mus_s = mus_s[
            frame
        ]

    mus_s = mus_s.reshape(-1)

    if len(mus_pos) != len(mus_parent):
        raise RuntimeError(
            "mus_pos and mus_parent have different lengths."
        )

    if len(mus_pos) != len(mus_s):
        raise RuntimeError(
            "mus_pos and mus_s have different lengths."
        )

    if "centre" in z.files:

        centre_arr = np.asarray(
            z["centre"],
            dtype=float,
        )

        if centre_arr.ndim == 2:
            centre = centre_arr[
                frame
            ]
        else:
            centre = centre_arr

    else:

        centre = mus_pos.mean(
            axis=0
        )

    return (
        z,
        mus_pos,
        mus_parent,
        mus_s,
        centre,
        frame,
    )


# ---------------------------------------------------------------------------
# Dense particle selection
# ---------------------------------------------------------------------------

def select_dense_indices(
    s: np.ndarray,
    n_lines: int,
    seed: int,
) -> np.ndarray:
    """Select many particles while covering the full s range."""

    n = len(s)

    if n <= n_lines:
        return np.arange(n)

    rng = np.random.default_rng(
        seed
    )

    n_bins = max(
        40,
        int(np.sqrt(n_lines)),
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    chosen = []

    base = n_lines // n_bins
    remainder = n_lines % n_bins

    for i in range(n_bins):

        lo = edges[i]
        hi = edges[i + 1]

        if i == n_bins - 1:

            mask = (
                (s >= lo)
                & (s <= hi)
            )

        else:

            mask = (
                (s >= lo)
                & (s < hi)
            )

        candidates = np.flatnonzero(
            mask
        )

        if len(candidates) == 0:
            continue

        n_pick = base

        if i < remainder:
            n_pick += 1

        n_pick = min(
            n_pick,
            len(candidates),
        )

        if n_pick <= 0:
            continue

        selected = rng.choice(
            candidates,
            size=n_pick,
            replace=False,
        )

        chosen.extend(
            selected.tolist()
        )

    chosen = np.asarray(
        chosen,
        dtype=int,
    )

    if len(chosen) < n_lines:

        available = np.setdiff1d(
            np.arange(n),
            chosen,
            assume_unique=False,
        )

        extra_n = min(
            n_lines - len(chosen),
            len(available),
        )

        if extra_n > 0:

            extra = rng.choice(
                available,
                size=extra_n,
                replace=False,
            )

            chosen = np.concatenate(
                [
                    chosen,
                    extra,
                ]
            )

    return np.sort(
        chosen
    )


# ---------------------------------------------------------------------------
# Eye G surface-render camera
# ---------------------------------------------------------------------------

def set_eyeG_camera(
    plot: pv.Plotter,
    centre: np.ndarray,
    span: float,
    az_deg: float = 16.0,
    el_deg: float = 10.0,
    scale_factor: float = 1.0,
):
    """Set the Eye G surface-render camera convention.

    Eye G uses:

        d = [
            sin(az) cos(el),
            sin(el),
            cos(az) cos(el)
        ]

    and an orthographic camera looking at the eye centre with +Y up.

    `scale_factor` changes only the framing.
    Smaller values zoom in.
    """

    az = np.radians(
        az_deg
    )

    el = np.radians(
        el_deg
    )

    direction = np.array(
        [
            np.sin(az) * np.cos(el),
            np.sin(el),
            np.cos(az) * np.cos(el),
        ],
        dtype=float,
    )

    camera_position = (
        centre
        + direction * 10.0
    )

    plot.camera_position = (
        tuple(camera_position),
        tuple(centre),
        (0.0, 1.0, 0.0),
    )

    plot.camera.parallel_projection = True

    plot.camera.parallel_scale = (
        scale_factor * span
    )


# ---------------------------------------------------------------------------
# Build fibre line mesh
# ---------------------------------------------------------------------------

def build_fibre_lines(
    points: np.ndarray,
    fibres: np.ndarray,
    s: np.ndarray,
    selected: np.ndarray,
    line_scale: float,
    span: float,
):
    """Build all fibre-axis line segments as one PolyData object."""

    origins = points[
        selected
    ]

    directions = fibres[
        selected
    ]

    scalars = s[
        selected
    ]

    valid = (
        np.linalg.norm(
            directions,
            axis=1,
        )
        > 1e-12
    )

    origins = origins[
        valid
    ]

    directions = directions[
        valid
    ]

    scalars = scalars[
        valid
    ]

    if len(origins) == 0:
        return None

    directions = (
        directions
        / np.linalg.norm(
            directions,
            axis=1,
            keepdims=True,
        )
    )

    half_length = (
        0.5
        * line_scale
        * span
    )

    p0 = (
        origins
        - half_length
        * directions
    )

    p1 = (
        origins
        + half_length
        * directions
    )

    n_lines = len(p0)

    vertices = np.empty(
        (
            n_lines * 2,
            3,
        ),
        dtype=float,
    )

    vertices[0::2] = p0
    vertices[1::2] = p1

    lines = np.empty(
        n_lines * 3,
        dtype=np.int64,
    )

    lines[0::3] = 2

    lines[1::3] = np.arange(
        0,
        2 * n_lines,
        2,
    )

    lines[2::3] = np.arange(
        1,
        2 * n_lines,
        2,
    )

    mesh = pv.PolyData(
        vertices,
        lines=lines,
    )

    mesh.cell_data[
        "s"
    ] = scalars

    return mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--npz",
        default="baseline_curves.npz",
    )

    parser.add_argument(
        "--out",
        default="viz_muscle_fibres_dense.png",
    )

    parser.add_argument(
        "--muscles",
        nargs="+",
        default=MUSCLE_NAMES,
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--az",
        type=float,
        default=16.0,
        help="Eye G camera azimuth in degrees",
    )

    parser.add_argument(
        "--el",
        type=float,
        default=10.0,
        help="Eye G camera elevation in degrees",
    )

    parser.add_argument(
        "--zoom",
        type=float,
        default=0.65,
        help="camera parallel scale relative to eye span; smaller = larger view",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=12,
        help="exact Eye G neighbour count",
    )

    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-9,
        help="exact Eye G ridge",
    )

    parser.add_argument(
        "--smooth",
        action="store_true",
        help="smooth the material coordinate s before differentiating it "
             "(s -> grad -> f), instead of differentiating raw s",
    )

    parser.add_argument(
        "--smooth-k",
        type=int,
        default=None,
        help="neighbours for the smoothing graph; default = --k",
    )

    parser.add_argument(
        "--smooth-iters",
        type=int,
        default=8,
        help="rounds of k-NN neighbour averaging applied to s",
    )

    parser.add_argument(
        "--smooth-lambda",
        type=float,
        default=0.5,
        help="step size per round, in [0, 1]; 0 = no change, 1 = replace with "
             "the neighbour mean",
    )

    parser.add_argument(
        "--n-lines",
        type=int,
        default=550,
        help="number of fibre axes rendered per muscle",
    )

    parser.add_argument(
        "--line-scale",
        type=float,
        default=0.040,
        help="fibre line length relative to eye span",
    )

    parser.add_argument(
        "--line-width",
        type=float,
        default=0.75,
        help="thin fibre line width",
    )

    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        default=[1800, 1100],
    )

    args = parser.parse_args()

    requested = [
        name.upper()
        for name in args.muscles
    ]

    invalid = [
        name
        for name in requested
        if name not in MUSCLE_NAMES
    ]

    if invalid:

        raise SystemExit(
            f"Unknown muscles {invalid}; "
            f"expected {MUSCLE_NAMES}"
        )

    # ------------------------------------------------------------------
    # Load capture
    # ------------------------------------------------------------------

    (
        z,
        mus_pos,
        mus_parent,
        mus_s,
        centre,
        frame,
    ) = load_capture(
        args.npz,
        args.frame,
    )

    print()
    print(
        "Eye G dense fibre-axis visualization"
    )
    print(
        f"capture        : {args.npz}"
    )
    print(
        f"frame          : {frame}"
    )
    print(
        f"particles      : {len(mus_pos)}"
    )
    print(
        f"camera         : az={args.az:.1f}, el={args.el:.1f}"
    )
    print(
        f"zoom           : {args.zoom:.3f}"
    )
    print(
        f"k              : {args.k}"
    )
    print(
        f"ridge          : {args.ridge:.1e}"
    )
    print(
        f"smooth s       : {args.smooth}"
        + (f"  (k={args.smooth_k or args.k}, iters={args.smooth_iters}, "
           f"lambda={args.smooth_lambda:.2f})" if args.smooth else "")
    )
    print(
        f"lines/muscle   : {args.n_lines}"
    )
    print(
        "particles shown: NO"
    )
    print(
        "fibre source   : reconstructed from mus_pos + mus_s"
    )
    print()

    # ------------------------------------------------------------------
    # Scene scale
    # ------------------------------------------------------------------

    distance = np.linalg.norm(
        mus_pos - centre,
        axis=1,
    )

    span = max(
        float(
            1.05 * distance.max()
        ),
        0.08,
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    os.environ.setdefault(
        "PYVISTA_OFF_SCREEN",
        "true",
    )

    pv.OFF_SCREEN = True

    plot = pv.Plotter(
        off_screen=True,
        window_size=tuple(
            args.size
        ),
        border=False,
    )

    plot.set_background(
        "black"
    )

    # Exact Eye G surface-view camera.
    set_eyeG_camera(
        plot,
        centre,
        span,
        az_deg=args.az,
        el_deg=args.el,
        scale_factor=args.zoom,
    )

    # ------------------------------------------------------------------
    # Build meshes
    # ------------------------------------------------------------------

    all_meshes = []

    for mi, muscle in enumerate(
        MUSCLE_NAMES
    ):

        if muscle not in requested:
            continue

        ids = np.flatnonzero(
            mus_parent == mi
        )

        if len(ids) == 0:

            print(
                f"{muscle}: no particles"
            )

            continue

        points = mus_pos[
            ids
        ]

        s = np.clip(
            mus_s[
                ids
            ],
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Optionally smooth s BEFORE differentiating it -- s -> grad -> f,
        # not f smoothed after the fact. See smooth_material_coord's docstring.
        # --------------------------------------------------------------

        if args.smooth:
            s = smooth_material_coord(
                points,
                s,
                k=args.smooth_k or args.k,
                iters=args.smooth_iters,
                lam=args.smooth_lambda,
            )

        # --------------------------------------------------------------
        # Exact repository construction
        # --------------------------------------------------------------

        fibres = compute_repo_fibres(
            points,
            s,
            k=args.k,
            ridge=args.ridge,
        )

        # --------------------------------------------------------------
        # Dense selection distributed along s
        # --------------------------------------------------------------

        selected = select_dense_indices(
            s,
            args.n_lines,
            seed=1000 + mi,
        )

        mesh = build_fibre_lines(
            points,
            fibres,
            s,
            selected,
            args.line_scale,
            span,
        )

        if mesh is not None:

            all_meshes.append(
                mesh
            )

        valid = (
            np.linalg.norm(
                fibres,
                axis=1,
            )
            > 1e-12
        )

        print(
            f"{muscle:2s} "
            f"N={len(ids):5d} "
            f"lines={len(selected):4d} "
            f"valid={100.0 * valid.mean():6.2f}% "
            f"s=[{s.min():.3f},{s.max():.3f}]"
        )

    # ------------------------------------------------------------------
    # Add fibres
    # ------------------------------------------------------------------

    for mesh in all_meshes:

        plot.add_mesh(
            mesh,
            scalars="s",
            cmap="viridis",
            clim=(0.0, 1.0),
            line_width=args.line_width,
            lighting=False,
            show_scalar_bar=False,
            render_lines_as_tubes=False,
        )

    # ------------------------------------------------------------------
    # Scalar-bar carrier
    # ------------------------------------------------------------------

    bar = pv.PolyData(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
    )

    bar.point_data[
        "s"
    ] = np.array(
        [0.0, 1.0]
    )

    plot.add_mesh(
        bar,
        scalars="s",
        cmap="viridis",
        clim=(0.0, 1.0),
        opacity=0.0,
        show_scalar_bar=True,
        # vertical, matching viz_muscle_taper.py: the default horizontal bar sits
        # along the bottom and eats into the frame's HEIGHT rather than its width,
        # so the two panels' eye renders end up at different visual scale even at
        # equal canvas height.
        scalar_bar_args={
            "title": "material s",
            "n_labels": 5,
            "color": "white",
            "fmt": "%.2f",
            "vertical": True,
        },
    )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    smooth_line = (
        f"s smoothed: k={args.smooth_k or args.k}, iters={args.smooth_iters}, "
        f"lambda={args.smooth_lambda:.2f}  (rescaled to [0,1] after)"
        if args.smooth else
        "s not smoothed (raw material coordinate)"
    )
    plot.add_text(
        "Eye G - local muscle fibre axes\n"
        f"exact Eye G construction: k={args.k}, ridge={args.ridge:.0e}\n"
        f"{smooth_line}\n"
        "colour = material coordinate s  |  thin lines = fibre axes\n"
        "no MPM particles  |  no muscle labels",
        position="upper_left",
        font_size=16,
        color="white",
    )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    plot.show(
        screenshot=args.out,
        auto_close=False,
    )

    plot.close()

    print()
    print(
        f"Wrote: {args.out}"
    )


if __name__ == "__main__":
    main()


# With this version, the default command:

# ```bash
# python viz_muscle_fibres_dense.py
# ```

# uses the Eye G camera convention directly:

# ```text
# az = 16 deg
# el = 10 deg
# orthographic
# +Y up
# look at eye centre
# ```

# and I set the default `--zoom 0.65` specifically to make the fibre field fill the frame more like the visualization you preferred.

#  python viz_muscle_fibre_construction.py --size 1600 1100 --smooth --smooth-iters 130 --smooth-lambda 0.6 --out viz_muscle_fibres_dense_smooth.png
