
"""Visualise Eye G muscle taper T(s) and local fibre axes.

Run from:
    Plexus/prototype/eye/archive/eye_G

Examples
--------
Taper only:

    python viz_muscle_taper.py

Taper + fibre directions:

    python viz_muscle_taper.py --show-fibres

Bent muscles using the Eye G surface-render camera:

    python viz_muscle_taper.py \
        --muscles SR SO IO \
        --show-fibres \
        --view eyeG \
        --zoom 0.65

More visible fibre axes:

    python viz_muscle_taper.py \
        --show-fibres \
        --n-fibres 20 \
        --fibre-scale 0.18 \
        --zoom 0.65

Reference taper:

    python viz_muscle_taper.py \
        --show-fibres \
        --show-reference

Important
---------
A muscle particle p carries a local fibre direction f_p.

The active stress is

    sigma_act = g_p f_p f_p^T

so f_p itself is a vector, while f_p f_p^T is a 3x3 tensor.

If baseline_curves.npz contains an actual particle-wise fibre field,
this script uses it.

If it does not, the script estimates a local geometric tangent from the
particle geometry. That fallback is explicitly labelled as an estimate and
must NOT be interpreted as the exact p.fibre field used by muscle_contract.

Camera
------
The default "eyeG" camera follows the Eye G surface-render convention:

    azimuth   = 16 degrees
    elevation = 10 degrees
    +Y is up
    orthographic projection
    look at the eye centre

Use --az and --el to change the camera angle.

Use --zoom to control the orthographic framing:
    smaller = larger object
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np


MUSCLE_NAMES = [
    "LR",
    "SR",
    "MR",
    "IR",
    "SO",
    "IO",
]

MUSCLE_COLORS = {
    "LR": "#e76f51",
    "SR": "#2a9d8f",
    "MR": "#457b9d",
    "IR": "#e9c46a",
    "SO": "#9b5de5",
    "IO": "#f15bb5",
}


# ---------------------------------------------------------------------------
# Taper
# ---------------------------------------------------------------------------

def taper(s: np.ndarray) -> np.ndarray:
    """Eye G taper:

        T(s) = sqrt(sin(pi*s))

    clipped to the material-coordinate interval [0, 1].
    """

    s = np.clip(
        np.asarray(s, dtype=float),
        0.0,
        1.0,
    )

    return np.sqrt(
        np.clip(
            np.sin(np.pi * s),
            0.0,
            None,
        )
    )


# ---------------------------------------------------------------------------
# NPZ helpers
# ---------------------------------------------------------------------------

def _pick_key(
    z: np.lib.npyio.NpzFile,
    candidates: Tuple[str, ...],
) -> str | None:
    """Return the first matching key."""

    for key in candidates:
        if key in z.files:
            return key

    return None


def _load_capture(path: str):
    """Load the Eye G renderer capture."""

    z = np.load(
        path,
        allow_pickle=True,
    )

    required = [
        "mus_pos",
        "mus_parent",
    ]

    missing = [
        key
        for key in required
        if key not in z.files
    ]

    if missing:
        raise RuntimeError(
            f"{path} is missing {missing}. "
            "Expected an Eye G capture with mus_pos and mus_parent."
        )

    return z


# ---------------------------------------------------------------------------
# Geometry fallback for s
# ---------------------------------------------------------------------------

def _reconstruct_s(points: np.ndarray) -> np.ndarray:
    """Estimate an approximate along-muscle coordinate from geometry.

    This is only a fallback.

    It uses the first principal component of the particle cloud. It is NOT
    the material coordinate p.s used by muscle_contract.
    """

    points = np.asarray(
        points,
        dtype=float,
    )

    if len(points) < 2:
        return np.zeros(
            len(points),
            dtype=float,
        )

    x = (
        points
        - points.mean(
            axis=0,
            keepdims=True,
        )
    )

    _, _, vh = np.linalg.svd(
        x,
        full_matrices=False,
    )

    axis = vh[0]

    q = x @ axis

    q -= q.min()

    den = q.max()

    if den <= 1e-12:
        return np.zeros_like(q)

    return q / den


# ---------------------------------------------------------------------------
# Geometric tangent fallback
# ---------------------------------------------------------------------------

def _local_tangent(
    points: np.ndarray,
    s: np.ndarray,
    n_vectors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local fibre-axis directions from the bent muscle geometry.

    Strategy
    --------
    1. Divide the muscle into bins along s.
    2. Compute a centre point for each occupied bin.
    3. Estimate the tangent from neighbouring centre points.
    4. Orient the tangent consistently along increasing s.

    Returns
    -------
    positions:
        Representative points on the centreline.

    tangents:
        Unit tangent vectors.

    This is a GEOMETRIC fallback, not p.fibre.
    """

    if len(points) < 3:
        return (
            np.empty((0, 3)),
            np.empty((0, 3)),
        )

    order = np.argsort(s)

    points = points[order]
    s = s[order]

    edges = np.linspace(
        0.0,
        1.0,
        n_vectors + 1,
    )

    centres = []
    centre_s = []

    for i in range(n_vectors):

        lo = edges[i]
        hi = edges[i + 1]

        if i == n_vectors - 1:
            mask = (
                (s >= lo)
                & (s <= hi)
            )
        else:
            mask = (
                (s >= lo)
                & (s < hi)
            )

        idx = np.flatnonzero(mask)

        if len(idx) == 0:
            continue

        centres.append(
            points[idx].mean(axis=0)
        )

        centre_s.append(
            s[idx].mean()
        )

    if len(centres) < 2:
        return (
            np.empty((0, 3)),
            np.empty((0, 3)),
        )

    centres = np.asarray(
        centres,
        dtype=float,
    )

    centre_s = np.asarray(
        centre_s,
        dtype=float,
    )

    tangents = np.zeros_like(
        centres
    )

    for i in range(len(centres)):

        if i == 0:
            d = (
                centres[1]
                - centres[0]
            )

        elif i == len(centres) - 1:
            d = (
                centres[-1]
                - centres[-2]
            )

        else:
            d = (
                centres[i + 1]
                - centres[i - 1]
            )

        norm = np.linalg.norm(d)

        if norm > 1e-12:
            tangents[i] = d / norm

    good = np.linalg.norm(
        tangents,
        axis=1,
    ) > 1e-12

    if not np.all(good):

        valid = np.flatnonzero(good)

        if len(valid) == 0:
            return (
                np.empty((0, 3)),
                np.empty((0, 3)),
            )

        for i in np.flatnonzero(~good):

            nearest = valid[
                np.argmin(
                    np.abs(valid - i)
                )
            ]

            tangents[i] = tangents[
                nearest
            ]

    return centres, tangents


# ---------------------------------------------------------------------------
# Eye G camera
# ---------------------------------------------------------------------------

def _set_eyeG_camera(
    plot,
    centre: np.ndarray,
    span: float,
    az_deg: float,
    el_deg: float,
    zoom: float,
) -> None:
    """Set the Eye G surface-render camera convention."""

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
        + 10.0 * direction
    )

    plot.camera_position = (
        tuple(camera_position),
        tuple(centre),
        (0.0, 1.0, 0.0),
    )

    plot.camera.parallel_projection = True

    plot.camera.parallel_scale = (
        zoom * span
    )


# ---------------------------------------------------------------------------
# Particle display subsampling
# ---------------------------------------------------------------------------

def _subsample(
    idx: np.ndarray,
    nmax: int,
    seed: int = 7,
) -> np.ndarray:
    """Randomly subsample particles for rendering."""

    if len(idx) <= nmax:
        return idx

    rng = np.random.default_rng(
        seed
    )

    return np.sort(
        rng.choice(
            idx,
            size=nmax,
            replace=False,
        )
    )


# ---------------------------------------------------------------------------
# Fibre-field detection
# ---------------------------------------------------------------------------

def _find_fibre_field(
    z: np.lib.npyio.NpzFile,
):
    """Try likely names for a stored particle fibre field."""

    candidates = (
        "mus_fibre",
        "mus_fiber",
        "fibre",
        "fiber",
        "mus_f",
        "muscle_fibre",
        "muscle_fiber",
        "fibre_dir",
        "fiber_dir",
        "mus_fibre_dir",
        "mus_fiber_dir",
    )

    return _pick_key(
        z,
        candidates,
    )


# ---------------------------------------------------------------------------
# Plot arrows
# ---------------------------------------------------------------------------

def _add_fibre_arrow(
    plot,
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    color: str = "white",
):
    """Draw one visible fibre-direction arrow."""

    direction = np.asarray(
        direction,
        dtype=float,
    )

    norm = np.linalg.norm(
        direction
    )

    if norm <= 1e-12:
        return

    direction /= norm

    start = (
        origin
        - 0.5 * length * direction
    )

    arrow = pv.Arrow(
        start=start,
        direction=direction,
        scale=length,
        tip_length=0.30,
        tip_radius=0.10,
        shaft_radius=0.035,
        resolution=20,
    )

    plot.add_mesh(
        arrow,
        color=color,
        lighting=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument(
        "--npz",
        default="baseline_curves.npz",
    )

    ap.add_argument(
        "--out",
        default="viz_muscle_taper.png",
    )

    ap.add_argument(
        "--view",
        choices=(
            "eyeG",
            "anterior",
            "lateral",
            "oblique",
        ),
        default="eyeG",
    )

    ap.add_argument(
        "--az",
        type=float,
        default=16.0,
        help="Eye G camera azimuth in degrees",
    )

    ap.add_argument(
        "--el",
        type=float,
        default=10.0,
        help="Eye G camera elevation in degrees",
    )

    ap.add_argument(
        "--zoom",
        type=float,
        default=0.65,
        help="orthographic camera scale; smaller = larger scene",
    )

    ap.add_argument(
        "--muscles",
        nargs="+",
        default=MUSCLE_NAMES,
    )

    ap.add_argument(
        "--size",
        nargs=2,
        type=int,
        default=[1600, 1100],
    )

    ap.add_argument(
        "--n",
        type=int,
        default=18000,
        help="maximum displayed particles per muscle",
    )

    ap.add_argument(
        "--frame",
        type=int,
        default=0,
        help="frame for time-resolved captures",
    )

    ap.add_argument(
        "--show-reference",
        action="store_true",
    )

    ap.add_argument(
        "--show-fibres",
        action="store_true",
        help=(
            "show actual particle fibre directions if available; "
            "otherwise show geometric tangent estimates"
        ),
    )

    ap.add_argument(
        "--n-fibres",
        type=int,
        default=12,
        help="number of fibre vectors per muscle",
    )

    ap.add_argument(
        "--fibre-scale",
        type=float,
        default=0.16,
        help="length of fibre arrows relative to eye span",
    )

    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load PyVista
    # ------------------------------------------------------------------

    os.environ.setdefault(
        "PYVISTA_OFF_SCREEN",
        "true",
    )

    import pyvista as pv

    pv.OFF_SCREEN = True

    # ------------------------------------------------------------------
    # Load capture
    # ------------------------------------------------------------------

    z = _load_capture(
        args.npz
    )

    print("\nNPZ contents:")

    for key in z.files:

        arr = np.asarray(
            z[key]
        )

        print(
            f"  {key:24s} shape={arr.shape}"
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    mus_pos = np.asarray(
        z["mus_pos"],
        dtype=float,
    )

    if mus_pos.ndim == 3:

        frame = min(
            max(args.frame, 0),
            mus_pos.shape[0] - 1,
        )

        mus_pos = mus_pos[
            frame
        ]

    else:

        frame = 0

    # ------------------------------------------------------------------
    # Parent IDs
    # ------------------------------------------------------------------

    parent = np.asarray(
        z["mus_parent"],
        dtype=int,
    ).reshape(-1)

    if len(parent) != len(mus_pos):

        raise RuntimeError(
            "mus_parent and mus_pos have incompatible sizes."
        )

    # ------------------------------------------------------------------
    # Requested muscles
    # ------------------------------------------------------------------

    requested = [
        name.upper()
        for name in args.muscles
    ]

    unknown = [
        name
        for name in requested
        if name not in MUSCLE_NAMES
    ]

    if unknown:

        raise SystemExit(
            f"Unknown muscles {unknown}; "
            f"expected {MUSCLE_NAMES}"
        )

    # ------------------------------------------------------------------
    # Exact material coordinate s
    # ------------------------------------------------------------------

    s_key = _pick_key(
        z,
        (
            "mus_s",
            "s",
            "muscle_s",
            "fiber_s",
            "fibre_s",
        ),
    )

    exact_s = None

    if s_key is not None:

        candidate = np.asarray(
            z[s_key],
            dtype=float,
        )

        if candidate.ndim == 2:
            candidate = candidate[
                frame
            ]

        candidate = candidate.reshape(-1)

        if len(candidate) == len(mus_pos):

            exact_s = candidate

            print(
                f"\nExact material coordinate: {s_key}"
            )

        else:

            print(
                f"\nWARNING: {s_key} has incompatible shape "
                f"{candidate.shape}"
            )

    # ------------------------------------------------------------------
    # Actual particle fibre field
    # ------------------------------------------------------------------

    fibre_key = _find_fibre_field(z)

    exact_fibres = None

    if fibre_key is not None:

        candidate = np.asarray(
            z[fibre_key],
            dtype=float,
        )

        if candidate.ndim == 3:
            candidate = candidate[
                frame
            ]

        if candidate.shape == mus_pos.shape:

            exact_fibres = candidate

            print(
                f"Exact particle fibre field: {fibre_key}"
            )

        else:

            print(
                f"WARNING: {fibre_key} has shape "
                f"{candidate.shape}; expected "
                f"{mus_pos.shape}"
            )

    else:

        print(
            "\nNO PARTICLE FIBRE FIELD FOUND IN CAPTURE."
        )

    # ------------------------------------------------------------------
    # Centre
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Scene size
    # ------------------------------------------------------------------

    span = 1.05 * np.max(
        np.linalg.norm(
            mus_pos - centre,
            axis=1,
        )
    )

    span = max(
        float(span),
        0.08,
    )

    # ------------------------------------------------------------------
    # Plotter
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    if args.view == "eyeG":

        _set_eyeG_camera(
            plot,
            centre,
            span,
            az_deg=args.az,
            el_deg=args.el,
            zoom=args.zoom,
        )

    elif args.view == "anterior":

        plot.camera_position = (
            tuple(
                centre
                + np.array(
                    [0.0, 0.0, span]
                )
            ),
            tuple(centre),
            (0.0, 1.0, 0.0),
        )

        plot.camera.parallel_projection = True
        plot.camera.parallel_scale = (
            args.zoom * span
        )

    elif args.view == "lateral":

        plot.camera_position = (
            tuple(
                centre
                + np.array(
                    [span, 0.0, 0.0]
                )
            ),
            tuple(centre),
            (0.0, 1.0, 0.0),
        )

        plot.camera.parallel_projection = True
        plot.camera.parallel_scale = (
            args.zoom * span
        )

    else:

        forward = np.array(
            [0.65, 0.35, 0.68],
            dtype=float,
        )

        forward /= np.linalg.norm(
            forward
        )

        plot.camera_position = (
            tuple(
                centre
                + span * forward
            ),
            tuple(centre),
            (0.0, 1.0, 0.0),
        )

        plot.camera.parallel_projection = True
        plot.camera.parallel_scale = (
            args.zoom * span
        )

    # ------------------------------------------------------------------
    # Render every muscle
    # ------------------------------------------------------------------

    for mi, name in enumerate(
        MUSCLE_NAMES
    ):

        if name not in requested:
            continue

        ids_all = np.flatnonzero(
            parent == mi
        )

        if len(ids_all) == 0:

            print(
                f"WARNING: no particles for {name}"
            )

            continue

        points_all = mus_pos[
            ids_all
        ]

        # --------------------------------------------------------------
        # s
        # --------------------------------------------------------------

        if exact_s is not None:

            s_all = np.clip(
                exact_s[ids_all],
                0.0,
                1.0,
            )

            s_source = (
                "exact material s"
            )

        else:

            s_all = _reconstruct_s(
                points_all
            )

            s_source = (
                "geometry-reconstructed s"
            )

        # --------------------------------------------------------------
        # Particle cloud
        # --------------------------------------------------------------

        ids_display = _subsample(
            ids_all,
            args.n,
            seed=100 + mi,
        )

        local_lookup = {
            global_id: local_id
            for local_id, global_id
            in enumerate(ids_all)
        }

        local_display = np.asarray(
            [
                local_lookup[g]
                for g in ids_display
            ],
            dtype=int,
        )

        points_display = mus_pos[
            ids_display
        ]

        s_display = s_all[
            local_display
        ]

        T = taper(
            s_display
        )

        cloud = pv.PolyData(
            points_display
        )

        cloud["T"] = T

        plot.add_mesh(
            cloud,
            scalars="T",
            cmap="viridis",
            clim=(0.0, 1.0),
            point_size=5.5,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )

        # --------------------------------------------------------------
        # Fibre direction visualization
        # --------------------------------------------------------------

        if args.show_fibres:

            if exact_fibres is not None:

                fibre_points = (
                    points_all
                )

                fibre_vectors = (
                    exact_fibres[
                        ids_all
                    ]
                )

                fibre_source = (
                    "ACTUAL p.fibre"
                )

            else:

                fibre_points, fibre_vectors = (
                    _local_tangent(
                        points_all,
                        s_all,
                        args.n_fibres,
                    )
                )

                fibre_source = (
                    "GEOMETRIC TANGENT FALLBACK"
                )

            if len(fibre_points) > 0:

                if (
                    len(fibre_points)
                    > args.n_fibres
                ):

                    selection = np.linspace(
                        0,
                        len(fibre_points) - 1,
                        args.n_fibres,
                        dtype=int,
                    )

                    fibre_points = (
                        fibre_points[
                            selection
                        ]
                    )

                    fibre_vectors = (
                        fibre_vectors[
                            selection
                        ]
                    )

                norms = np.linalg.norm(
                    fibre_vectors,
                    axis=1,
                )

                good = (
                    norms > 1e-12
                )

                fibre_points = (
                    fibre_points[good]
                )

                fibre_vectors = (
                    fibre_vectors[good]
                    / norms[
                        good,
                        None,
                    ]
                )

                arrow_length = (
                    args.fibre_scale
                    * span
                )

                for origin, direction in zip(
                    fibre_points,
                    fibre_vectors,
                ):

                    _add_fibre_arrow(
                        plot,
                        origin,
                        direction,
                        arrow_length,
                        color="white",
                    )

                print(
                    f"{name:2s}: "
                    f"{fibre_source}, "
                    f"{len(fibre_points)} vectors"
                )

            else:

                print(
                    f"{name:2s}: "
                    "no fibre vectors available"
                )

        # --------------------------------------------------------------
        # Taper report
        # --------------------------------------------------------------

        print(
            f"{name:2s}: "
            f"N={len(ids_all):6d}  "
            f"s=[{s_all.min():.3f}, "
            f"{s_all.max():.3f}]  "
            f"T=[{T.min():.3f}, "
            f"{T.max():.3f}]  "
            f"{s_source}"
        )

    # ------------------------------------------------------------------
    # Reference taper
    # ------------------------------------------------------------------

    if args.show_reference:

        sref = np.linspace(
            0.0,
            1.0,
            200,
        )

        tref = taper(
            sref
        )

        x0 = (
            centre
            + np.array(
                [
                    -0.42 * span,
                    -0.38 * span,
                    0.72 * span,
                ]
            )
        )

        curve_points = (
            x0
            + np.stack(
                [
                    0.32 * span
                    * (sref - 0.5),

                    0.24 * span
                    * (tref - 0.5),

                    np.zeros_like(
                        sref
                    ),
                ],
                axis=1,
            )
        )

        curve = pv.Spline(
            curve_points,
            200,
        )

        plot.add_mesh(
            curve,
            color="white",
            line_width=4,
        )

        plot.add_point_labels(
            np.asarray(
                [
                    x0
                    + np.array(
                        [
                            0.0,
                            0.16 * span,
                            0.0,
                        ]
                    )
                ]
            ),
            [
                "T(s) = sqrt(sin(pi s))"
            ],
            font_size=12,
            text_color="white",
            shape_opacity=0.0,
            show_points=False,
        )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    if exact_s is not None:

        s_title = (
            f"exact material s: {s_key}"
        )

    else:

        s_title = (
            "WARNING: geometry-reconstructed s"
        )

    if args.show_fibres:

        if exact_fibres is not None:

            fibre_title = (
                f"\nWHITE arrows = actual {fibre_key}"
            )

        else:

            fibre_title = (
                "\nWHITE arrows = geometric tangent "
                "(NO saved p.fibre in NPZ)"
            )

    else:

        fibre_title = ""

    plot.add_text(
        "Eye G - T(s) and local muscle fibre axes"
        + "\n"
        + s_title
        + fibre_title,
        position="upper_left",
        font_size=16,
        color="white",
    )

    # ------------------------------------------------------------------
    # Scalar bar
    # ------------------------------------------------------------------

    plot.add_scalar_bar(
        title="T(s)",
        n_labels=5,
        vertical=True,
        color="white",
        fmt="%.2f",
    )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    plot.show(
        screenshot=args.out,
        auto_close=False,
    )

    plot.close()

    print(
        f"\nWrote {args.out}"
    )


if __name__ == "__main__":
    main()
# ```

# For the view matching the Eye G surface renderer, use:

# ```bash
# python viz_muscle_taper.py --view eyeG --az 16 --el 10 --zoom 0.65 --show-fibres
# ```

# The main change relative to your uploaded version is the camera; the rest stays aligned with the script you supplied.
