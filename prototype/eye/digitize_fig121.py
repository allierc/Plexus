"""digitize_fig121 -- measure the zebrafish oculomotor plant off Fig. 12.1A of

    Tulenko & Currie, "Zebrafish Myology", ch. 12 of *The Zebrafish in Biomedical
    Research* (2020), p.116.  Panel A is redrawn from the camera lucida of
    Easter & Nicola (1996: fig 11): a VENTRAL view of a 96 hpf larva's right eye.

Why this file exists: the plant in `eye_anatomy.py` was built from the MAMMALIAN
plan (four recti from an annulus of Zinn, obliques inserting behind the equator,
a trochlea for the SO).  The fish is not built that way, and panel A says so.
Everything below is measured off the panel, in panel pixels, and then reduced to
dimensionless ratios of the globe's own semi-axes -- so the numbers survive any
choice of world scale.

READING THE PANEL
    image +x  ->  MEDIAL      (the head's midline is off the right edge)
    image +y  ->  CAUDAL      ("post" on the compass points down)
    out of page -> VENTRAL    ("vent" on the compass points at the reader)
so the eye's optic axis lies along image -x (LATERAL, out of the head), and the
plane of the drawing is the eye's HORIZONTAL section.

THE DASHED OUTLINES ARE THE THIRD DIMENSION.  The caption says dashed lines are
deeper structures: in a ventral view "deeper" means DORSAL, i.e. behind the globe.
So SO and SR (dashed) run over the globe's dorsal face and IO, IR, LR (solid) over
its ventral face.  That is what lets a single projection give 3-D insertions: an
insertion sits ON the globe surface, so its unknown dorso-ventral component is
recovered as  u_y = +-sqrt(1 - u_x^2 - u_z^2),  the sign taken from dashed/solid.

Outputs (into archive/eye_F/):
    fig121_panelA.png            the source panel, as extracted from the PDF
    fig121_digitized.png         the panel with the fitted outline + landmarks
    fig121_measurements.json     every number below, px and normalized
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy import ndimage as ndi

PDF = "/workspace/Plexus/papers/zebrafish_eye.pdf"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", "eye_F")

# --------------------------------------------------------------------------- #
#  The six bands, as the connected components of the panel's pink fill.
#
#  The bands are drawn with a black outline, so where one muscle crosses another
#  the stroke cuts the lower one in two and each muscle arrives as 1-3 fragments.
#  Only the fragment->muscle assignment below is read by eye; every number that
#  follows (insertion, origin, centreline, width, area) is measured from the
#  fragments themselves.  The assignment is the one the chapter describes:
#
#      SO  {1, 6}     origin at the rostral orbit; 6 is DASHED -> it runs dorsal
#      SR  {9,13,14}  14 is the free belly, 9 and 13 DASHED -> dorsal, like SO
#      MR  {2,10}     cut in two where IO crosses it: "the medial rectus extends
#                     BETWEEN THE OBLIQUES, inserting on the anteromedial surface"
#      IO  {4}        one long band, rostral orbit -> ventral face
#      IR  {15}       posterior orbit -> ventral face, just caudal to IO
#      LR  {17}       from far caudal, outside the orbit, to the caudal sclera
# --------------------------------------------------------------------------- #
FRAGMENTS = {
    "SO": dict(parts=[1, 6], side="dorsal"),
    "SR": dict(parts=[9, 13, 14], side="dorsal"),
    "MR": dict(parts=[2, 10], side="ventral"),
    "IO": dict(parts=[4], side="ventral"),
    "IR": dict(parts=[15], side="ventral"),
    "LR": dict(parts=[17], side="ventral"),
}

# The dorso-ventral height of each ORIGIN, in equatorial semi-axes.  This is the
# one thing a single ventral projection cannot give: the drawing fixes each
# origin's rostro-caudal and medio-lateral position but not its height.  Taken
# from the muscle each origin serves -- SO/SR run onto the dorsal face and so
# leave the skull above the horizontal plane, IO/IR below it, MR and LR in it.
ORIGIN_HEIGHT = {"SO": 0.45, "SR": 0.40, "MR": 0.0, "IO": -0.45, "IR": -0.40, "LR": -0.15}

# The physical scale.  Easter & Nicola (1996) give the larval eye's diameter; at
# 96 hpf the globe is ~0.25 mm across its equator.  Only used to print micrometres.
EYE_EQUATORIAL_DIAMETER_UM = 250.0


# --------------------------------------------------------------------------- #
def extract_panel() -> np.ndarray:
    """Panel A of Fig. 12.1, as RGB float, straight out of the PDF's image xref."""
    import fitz

    doc = fitz.open(PDF)
    xref = doc[1].get_images(full=True)[0][0]          # the only image on p.116
    pix = fitz.Pixmap(doc, xref)
    im = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return im[:, :535, :3].astype(float)               # panel A; B,C are the photos


def masks(A: np.ndarray) -> dict:
    r, g, b = A[..., 0], A[..., 1], A[..., 2]
    disk = np.zeros((7, 7), bool)
    yy, xx = np.mgrid[-3:4, -3:4]
    disk[yy ** 2 + xx ** 2 <= 9] = True

    dark = (r < 85) & (g < 85) & (b < 85)
    pig = ndi.binary_opening(dark, disk)               # drop strokes and text
    lab, n = ndi.label(pig)
    pig = lab == (int(np.argmax(ndi.sum(pig, lab, range(1, n + 1)))) + 1)

    light = (r > 215) & (g > 215) & (b > 215)
    lab2, n2 = ndi.label(ndi.binary_opening(light, disk))
    lens, best = None, 0
    for k in range(1, n2 + 1):
        m = lab2 == k
        ys, xs = np.nonzero(m)
        if m.sum() < 3000 or m.sum() < best:
            continue
        if xs.max() > A.shape[1] - 4 or ys.min() < 3 or ys.max() > A.shape[0] - 4:
            continue                                   # touches a panel edge -> background
        lens, best = m, m.sum()

    globe = ndi.binary_closing(ndi.binary_fill_holes(pig | lens), np.ones((15, 15), bool))
    gray = (abs(r - g) < 14) & (abs(g - b) < 14) & (r > 195) & (r < 240)
    paper = (r > 245) & (g > 245) & (b > 245)
    # The lens is drawn as light as the head background, so it reads as background
    # too -- but it is INSIDE the globe, and letting it count as "outside" would put
    # free-margin points on the pupil rim and pull the lateral outline in. Excluding
    # the globe's own interior fixes that without touching the paper beside the
    # cornea, which is the only place the lateral margin can be measured at all.
    outside = (gray | paper) & ~globe
    return dict(pig=pig, lens=lens, globe=globe, outside=outside)


def outline_profile(globe: np.ndarray, outside: np.ndarray, n_harm: int = 4):
    """r(phi) of the globe's outline, as a Fourier series through its FREE margin.

    Only the margin that faces background is used: where a muscle band is drawn
    over the eye the outline is hidden, and those pixels are skipped. Fitting a
    smooth periodic r(phi) interpolates across those gaps instead of inventing
    a boundary there.  4 harmonics is enough to carry the fore-aft asymmetry and
    not enough to chase the pen strokes.
    """
    ys, xs = np.nonzero(globe)
    cx, cy = xs.mean(), ys.mean()
    bnd = globe & ~ndi.binary_erosion(globe, np.ones((3, 3), bool))
    free = bnd & ndi.binary_dilation(outside, np.ones((7, 7), bool))
    fy, fx = np.nonzero(free)
    fx, fy = fx.astype(float), fy.astype(float)
    keep = np.hypot(fx - cx, fy - cy) < 340             # drop stray head-outline pixels
    fx, fy = fx[keep], fy[keep]

    def fit(cx, cy):
        rad = np.hypot(fx - cx, fy - cy)
        phi = np.arctan2(fy - cy, fx - cx)              # 0 = medial, +pi/2 = caudal
        cols = [np.ones_like(phi)]
        for k in range(1, n_harm + 1):
            cols += [np.cos(k * phi), np.sin(k * phi)]
        coef, *_ = np.linalg.lstsq(np.stack(cols, 1), rad, rcond=None)

        def r_of(p, coef=coef):
            p = np.atleast_1d(p)
            out = np.full_like(p, coef[0], dtype=float)
            for k in range(1, n_harm + 1):
                out += coef[2 * k - 1] * np.cos(k * p) + coef[2 * k] * np.sin(k * p)
            return out
        return coef, r_of, float(np.sqrt(((rad - r_of(phi)) ** 2).mean()))

    # The mask's own centroid is biased laterally: the muscle bands hide the medial
    # margin, so those pixels are missing from the mask. Re-centre on the AREA
    # CENTROID OF THE FITTED OUTLINE instead, which does not care about occlusion,
    # and refit until it stops moving. Without this the medial and lateral
    # semi-diameters differ by 40% of the axial semi-axis, which is fit bias, not eye.
    for _ in range(24):
        coef, r_of, rms = fit(cx, cy)
        p = np.linspace(-np.pi, np.pi, 1441)[:-1]
        rr = r_of(p)
        px, py = cx + rr * np.cos(p), cy + rr * np.sin(p)
        cross = px * np.roll(py, -1) - np.roll(px, -1) * py
        area = 0.5 * cross.sum()
        gx = ((px + np.roll(px, -1)) * cross).sum() / (6 * area)
        gy = ((py + np.roll(py, -1)) * cross).sum() / (6 * area)
        if abs(gx - cx) < 0.05 and abs(gy - cy) < 0.05:
            break
        cx, cy = gx, gy
    coef, r_of, rms = fit(cx, cy)
    return (cx, cy), coef, r_of, rms, len(fx)


def _suture(m: np.ndarray):
    """Rejoin a muscle's fragments across the band that was drawn on top of it.

    A morphological closing would need a radius as large as the widest crossing
    (25 px for MR, where IO runs over it) and would fatten the whole silhouette by
    that much. Instead the two nearest points of the two nearest fragments are
    joined by a bar.

    THE BAR CARRIES THE MUSCLE'S OWN WIDTH. A fragment tapers to nothing at the cut,
    so the width measured AT the cut is near zero; taking it would pinch the
    reconstructed muscle exactly where it is thickest in life -- and a pinched
    silhouette becomes a pinched cross-section, i.e. a weak spot that buckles when
    the muscle contracts. The half-width used at each end is therefore the widest the
    fragment gets within `look` px of the contact, and the bar tapers between the two.
    """
    bars, look = [], 34
    for _ in range(6):
        lab, n = ndi.label(m)
        if n <= 1:
            break
        first = lab == 1
        rest = m & ~first
        d_to_first, idx = ndi.distance_transform_edt(~first, return_indices=True)
        ys, xs = np.nonzero(rest)
        j = int(np.argmin(d_to_first[ys, xs]))
        b = (ys[j], xs[j])
        a = (int(idx[0][b]), int(idx[1][b]))
        edt = ndi.distance_transform_edt(m)

        def local_half(pt, side):
            """Half-width to give the bar at this end.

            Two candidates, and the wider wins. The LOCAL one is how wide the
            fragment is within `look` px of the contact -- right when the band was
            cut across its belly. But a band is often cut at a tapering corner
            (MR's wedge, SO's tip), where the local value is a few pixels and using
            it would neck the muscle down to a third of its section exactly where
            it is hidden. The CHARACTERISTIC one -- the 90th percentile of the whole
            fragment's half-width -- is what the muscle is away from its own ends.
            """
            yy, xx = np.mgrid[max(pt[0] - look, 0):pt[0] + look,
                              max(pt[1] - look, 0):pt[1] + look]
            win = side[max(pt[0] - look, 0):pt[0] + look, max(pt[1] - look, 0):pt[1] + look]
            near = win & (np.hypot(yy - pt[0], xx - pt[1]) <= look)
            e = edt[max(pt[0] - look, 0):pt[0] + look, max(pt[1] - look, 0):pt[1] + look][near]
            local = float(e.max()) if e.size else 0.0
            charac = float(np.percentile(edt[side], 90)) if side.any() else 0.0
            return max(2.5, local, 0.85 * charac)

        ha, hb = local_half(a, first), local_half(b, rest)
        steps = int(max(abs(a[0] - b[0]), abs(a[1] - b[1]))) + 1
        yy = np.linspace(a[0], b[0], steps)
        xx = np.linspace(a[1], b[1], steps)
        hh = np.linspace(ha, hb, steps)
        m = m.copy()
        rmax = int(math.ceil(max(ha, hb)))
        sy, sx = np.mgrid[-rmax:rmax + 1, -rmax:rmax + 1]
        rr = np.hypot(sy, sx)
        for y0, x0, h in zip(yy, xx, hh):
            keep = rr <= h
            py = np.clip((y0 + sy[keep]).astype(int), 0, m.shape[0] - 1)
            px = np.clip((x0 + sx[keep]).astype(int), 0, m.shape[1] - 1)
            m[py, px] = True
        bars.append(dict(gap_px=round(float(math.hypot(a[0] - b[0], a[1] - b[1])), 1),
                         half_width_px=[round(ha, 1), round(hb, 1)]))
    return m, bars


def band_masks(A: np.ndarray) -> dict:
    """The six muscle silhouettes, as boolean masks in panel pixels.

    The pink fill is labelled, the fragments listed in FRAGMENTS are unioned per
    muscle, and the union is closed: the black outline that cut the band is ~6 px
    wide, so a closing bridges the two halves of MR (and of SO, SR) back into one
    body without inventing tissue anywhere else.  A final dilation gives back the
    outline's own width, which the fill does not include but the muscle does.
    """
    r, g, b = A[..., 0], A[..., 1], A[..., 2]
    pink = ndi.binary_opening((r - g > 32) & (r - b > 15) & (r > 135), np.ones((3, 3), bool))
    lab, n = ndi.label(pink)
    letter = np.zeros((15, 15), bool)
    yy, xx = np.mgrid[-7:8, -7:8]
    letter[yy ** 2 + xx ** 2 <= 49] = True
    out, bridged = {}, {}
    for key, F in FRAGMENTS.items():
        m = np.zeros_like(pink)
        for p in F["parts"]:
            m |= lab == p
        # THE PANEL'S OWN LABELS ARE HOLES. "MR", "SO" and the rest are lettered in
        # dark ink straight onto the band, so the pink fill has letter-shaped bites
        # out of it and the muscle appears to neck down to a third of its width
        # exactly under its own name. Filling the holes, then closing at a little
        # over the stroke width, then filling again, puts the tissue back; the
        # closing costs at most 7 px of concavity on a band 25-60 px wide.
        m = ndi.binary_fill_holes(m)
        m = ndi.binary_fill_holes(ndi.binary_closing(m, letter))
        m, sutures = _suture(m)
        out[key] = ndi.binary_dilation(m, np.ones((3, 3), bool))
        bridged[key] = sutures
    out["_bridge_radius"] = bridged
    return out


def _geodesic(mask: np.ndarray, seed: tuple) -> np.ndarray:
    """Distance from `seed` measured INSIDE `mask` (8-connected BFS, sqrt2 diagonals).

    Euclidean distance is the wrong ruler for a muscle that curves around a globe:
    LR's two ends are 1.4 equatorial semi-axes apart in a straight line but the band
    between them bows. Walking the mask gives the length the tissue actually has.
    """
    INF = np.inf
    d = np.full(mask.shape, INF)
    sy, sx = seed
    d[sy, sx] = 0.0
    # Two-pass chamfer sweep, iterated: cheap, exact enough for ordering + length.
    off = [(-1, -1, 1.4142), (-1, 0, 1.0), (-1, 1, 1.4142), (0, -1, 1.0),
           (0, 1, 1.0), (1, -1, 1.4142), (1, 0, 1.0), (1, 1, 1.4142)]
    for _ in range(64):
        changed = False
        for rev in (False, True):
            rows = range(mask.shape[0] - 1, -1, -1) if rev else range(mask.shape[0])
            for y in rows:
                row = mask[y]
                if not row.any():
                    continue
                xs = np.nonzero(row)[0]
                for x in (xs[::-1] if rev else xs):
                    best = d[y, x]
                    for dy, dx, w in off:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx]:
                            v = d[ny, nx] + w
                            if v < best:
                                best = v
                    if best < d[y, x]:
                        d[y, x] = best
                        changed = True
        if not changed:
            break
    return d


def trace_bands(A, masks, centre, a_ax, a_eq):
    """Centreline, width profile and the two attachments of each muscle.

    Ordering runs from the INSERTION (the end that sits on the globe) to the
    ORIGIN (the cut end at the skull): the seed is the band pixel that lies
    deepest inside the globe's outline, and the origin is then the geodesically
    farthest pixel from it. Nothing here is read off the panel by hand.
    """
    cx, cy = centre
    out = {}
    for key, m in masks.items():
        if key.startswith("_"):
            continue
        ys, xs = np.nonzero(m)
        # The band's two ENDS by double sweep: farthest-from-anywhere, then
        # farthest-from-that. Picking the pixel merely closest to the globe centre
        # would land mid-band for LR, whose band passes right under the globe.
        far = lambda seed: np.unravel_index(
            np.where(np.isfinite(dd := _geodesic(m, seed)) & m, dd, -1).argmax(), m.shape)
        e1 = far((int(ys[0]), int(xs[0])))
        e2 = far(e1)
        rho = lambda e: math.hypot((e[0] - cy) / a_eq, (cx - e[1]) / a_ax)
        seed, other = (e1, e2) if rho(e1) <= rho(e2) else (e2, e1)   # insertion is the end on the globe
        d = _geodesic(m, seed)
        fin = np.isfinite(d) & m
        L = float(d[fin].max())
        oy, ox = other
        edt = ndi.distance_transform_edt(m)
        n_band = max(int(L // 12), 6)
        cl, wid, sv = [], [], []
        for i in range(n_band):
            lo, hi = L * i / n_band, L * (i + 1) / n_band
            sel = fin & (d >= lo) & (d <= hi)
            if sel.sum() < 8:
                continue
            by, bx = np.nonzero(sel)
            cl.append([bx.mean(), by.mean()])
            wid.append(2.0 * np.percentile(edt[sel], 90))
            sv.append(0.5 * (lo + hi) / L)
        out[key] = dict(
            centreline_px=np.asarray(cl), width_px=np.asarray(wid), s=np.asarray(sv),
            insertion_px=(float(seed[1]), float(seed[0])), origin_px=(float(ox), float(oy)),
            geodesic_length_px=L, area_px=int(m.sum()))
    return out


def lift_to_globe(pts_px, side, centre, a_ax, a_eq, r_of, h_origin):
    """Panel pixels -> 3-D globe coordinates (caudal, dorsal, lateral), in equatorial
    semi-axes, with the third dimension recovered from the projection.

    A point drawn OVER the globe is on the globe: the ventral camera projected it
    along the dorso-ventral axis, so lifting it back along that same axis inverts
    the projection exactly --  u_d = +-sqrt(1 - u_c^2 - u_l^2), the sign from
    whether the band was drawn dashed (dorsal) or solid (ventral).

    THE MUSCLE GOES TANGENT WHEN IT REACHES ITS ORIGIN'S HEIGHT. Inverting the
    projection naively pins the whole band to the globe, and since the silhouette
    is the globe's HORIZONTAL great circle, every band that reaches the rim is
    dragged back up to the equator and then has to dive again to reach its origin:
    a fold, up to a 129-degree kink in SR, and it is an artifact of the projection,
    not anatomy. A muscle leaves the sclera tangentially and runs straight from
    there, so the wrap is cut off at the origin's own height and the free belly
    holds it. That single clamp -- never past `h_origin`, on the side away from the
    insertion -- removes every fold and keeps every real feature (LR still dips
    under the globe on its way across).
    """
    # u_c, u_l are ELLIPSOID-NORMALIZED (each by its own semi-axis) so that the
    # globe is the unit sphere and the missing coordinate is a plain Pythagoras.
    # The value returned is physical again, in equatorial semi-axes.
    cx, cy = centre
    pts_px = np.atleast_2d(np.asarray(pts_px, float))
    u_c = (pts_px[:, 1] - cy) / a_eq
    u_l = (cx - pts_px[:, 0]) / a_ax
    # the measured outline, in the same normalized units, along each point's bearing
    phi = np.arctan2(pts_px[:, 1] - cy, pts_px[:, 0] - cx)
    r_out = r_of(phi)
    r_pt = np.hypot(pts_px[:, 0] - cx, pts_px[:, 1] - cy)
    on = r_pt <= r_out
    sgn = 1.0 if side == "dorsal" else -1.0
    u_d = np.zeros_like(u_c)
    u_d[on] = sgn * np.sqrt(np.clip(1.0 - u_c[on] ** 2 - u_l[on] ** 2, 0.0, None))
    if (~on).any():
        beyond = (r_pt[~on] - r_out[~on]) / a_eq
        far = max(float(beyond.max()), 1e-6)
        u_d[~on] = h_origin * np.clip(beyond / far, 0.0, 1.0)
    return np.stack([u_c, u_d, u_l * a_ax / a_eq], 1)


def _theta_deg(u_phys, a_ax, a_eq):
    """Polar angle from the OPTIC AXIS, measured on the globe as a sphere.

    `u_phys` is in equatorial semi-axes; on a flattened globe the angle a muscle
    subtends at the centre is not the angle its insertion makes on the unit
    sphere, and it is the latter that the spec's insertion directions want.
    """
    n = np.array([u_phys[0], u_phys[1], u_phys[2] * a_eq / a_ax], float)
    n /= max(float(np.linalg.norm(n)), 1e-12)
    return float(math.degrees(math.acos(np.clip(n[2], -1.0, 1.0))))


def main():
    os.makedirs(OUT, exist_ok=True)
    A = extract_panel()
    from PIL import Image
    Image.fromarray(A.astype(np.uint8)).save(os.path.join(OUT, "fig121_panelA.png"))

    M = masks(A)
    (cx, cy), coef, r_of, rms, n_free = outline_profile(M["globe"], M["outside"])

    # the four cardinal semi-diameters of the horizontal section
    R = {"medial": float(r_of(0.0)[0]), "caudal": float(r_of(np.pi / 2)[0]),
         "lateral": float(r_of(np.pi)[0]), "rostral": float(r_of(-np.pi / 2)[0])}
    a_axial = 0.5 * (R["lateral"] + R["medial"])       # along the optic axis
    a_equat = 0.5 * (R["rostral"] + R["caudal"])       # in the equatorial plane
    px_um = EYE_EQUATORIAL_DIAMETER_UM / (2 * a_equat)

    ly, lx = np.nonzero(M["lens"])
    lens_r = float(np.sqrt(M["lens"].sum() / np.pi))
    lens_c = (float(lx.mean()), float(ly.mean()))

    # the six bands: silhouettes -> centrelines -> 3-D attachments
    bands = band_masks(A)
    tr = trace_bands(A, bands, (cx, cy), a_axial, a_equat)
    ins, traces = {}, {}
    for key, T in tr.items():
        side = FRAGMENTS[key]["side"]
        h0 = ORIGIN_HEIGHT[key]
        cl3 = lift_to_globe(T["centreline_px"], side, (cx, cy), a_axial, a_equat, r_of, h0)
        u = lift_to_globe([T["insertion_px"]], side, (cx, cy), a_axial, a_equat, r_of, h0)[0]
        o = lift_to_globe([T["origin_px"]], side, (cx, cy), a_axial, a_equat, r_of, h0)[0]
        o[1] = h0                                      # the origin's height is anatomical
        nrm = float(np.linalg.norm(u))
        u_hat = u / nrm if nrm > 0 else u
        ins[key] = dict(
            side=side, fragments=FRAGMENTS[key]["parts"],
            insertion_px=[round(v, 1) for v in T["insertion_px"]],
            origin_px=[round(v, 1) for v in T["origin_px"]],
            u_caudal=round(float(u_hat[0]), 4), u_dorsal=round(float(u_hat[1]), 4),
            u_lateral=round(float(u_hat[2]), 4),
            radius_in_globe_units=round(nrm, 4),
            theta_from_optic_axis_deg=round(_theta_deg(u_hat, a_axial, a_equat), 1),
            origin_caudal=round(float(o[0]), 4), origin_dorsal=round(float(o[1]), 4),
            origin_lateral=round(float(o[2]), 4),
            length_along_band=round(float(T["geodesic_length_px"] / a_equat), 3),
            length_um=round(float(T["geodesic_length_px"] * px_um), 1),
            mean_width=round(float(T["width_px"].mean() / a_equat), 4),
            mean_width_um=round(float(T["width_px"].mean() * px_um), 1),
            max_width_um=round(float(T["width_px"].max() * px_um), 1),
            silhouette_area_px=T["area_px"],
        )
        traces[key] = dict(mask=bands[key], centreline=cl3, width=T["width_px"] / a_equat,
                           s=T["s"], side=side, h_origin=h0)

    # the trace asset the MPM seeder reads: the silhouettes themselves, plus what
    # is needed to lift any point inside them into the globe's frame.
    np.savez_compressed(
        os.path.join(OUT, "fig121_muscle_trace.npz"),
        keys=np.array(list(traces.keys())),
        centre_px=np.array([cx, cy]), a_axial_px=a_axial, a_equatorial_px=a_equat,
        outline_harmonics_px=np.asarray(coef), um_per_px=px_um,
        **{f"mask_{k}": v["mask"] for k, v in traces.items()},
        **{f"centreline_{k}": v["centreline"] for k, v in traces.items()},
        **{f"width_{k}": v["width"] for k, v in traces.items()},
        **{f"s_{k}": v["s"] for k, v in traces.items()},
        sides=np.array([traces[k]["side"] for k in traces]),
        h_origin=np.array([traces[k]["h_origin"] for k in traces]))

    meas = dict(
        source=dict(
            pdf=PDF, page=116, figure="12.1A",
            chapter="Tulenko & Currie 2020, Zebrafish Myology, ch.12",
            redrawn_from="Easter & Nicola 1996: fig 11 (camera lucida, 96 hpf larva)",
            view="ventral; +x_img medial, +y_img caudal, out-of-page ventral",
            dashed_means="deeper = DORSAL, i.e. behind the globe in this view"),
        globe=dict(
            centre_px=[round(cx, 1), round(cy, 1)],
            semi_diameters_px={k: round(v, 1) for k, v in R.items()},
            axial_semi_axis_px=round(a_axial, 1),
            equatorial_semi_axis_px=round(a_equat, 1),
            axial_over_equatorial=round(a_axial / a_equat, 4),
            fore_aft_asymmetry=round((R["caudal"] - R["rostral"]) / a_equat, 4),
            medio_lateral_asymmetry=round((R["medial"] - R["lateral"]) / a_axial, 4),
            harmonic_coefficients_px=[round(float(c), 2) for c in coef],
            outline_fit_rms_px=round(rms, 2), n_free_margin_px=n_free,
            scale_um_per_px=round(px_um, 4),
            equatorial_diameter_um=EYE_EQUATORIAL_DIAMETER_UM,
            axial_diameter_um=round(2 * a_axial * px_um, 1)),
        lens=dict(
            centre_px=[round(lens_c[0], 1), round(lens_c[1], 1)],
            radius_px=round(lens_r, 1),
            radius_over_equatorial_semi_axis=round(lens_r / a_equat, 4),
            centre_lateral_offset_in_axial_semi_axes=round((cx - lens_c[0]) / a_axial, 4),
            lateral_pole_reach=round(((cx - lens_c[0]) + lens_r) / a_axial, 4),
            diameter_um=round(2 * lens_r * px_um, 1)),
        muscles=ins,
    )
    with open(os.path.join(OUT, "fig121_measurements.json"), "w") as fh:
        json.dump(meas, fh, indent=2)

    _overlay(A, (cx, cy), r_of, a_axial, a_equat, lens_c, lens_r, ins, bands, tr)
    return meas


MCOL = {"SO": "#c58cff", "SR": "#ff5c5c", "MR": "#ffd24d",
        "IO": "#ff9c42", "IR": "#7ee081", "LR": "#4da3ff"}


def _overlay(A, c, r_of, a_axial, a_equat, lens_c, lens_r, ins, bands, tr):
    """The panel, with everything that was measured off it drawn back on top.

    Two panels: the globe fit on the left, the six traced silhouettes on the
    right. If a line here does not sit on the ink underneath it, the number it
    produced is wrong -- that is the whole point of keeping this figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    cx, cy = c
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 10.4), dpi=150)
    p = np.linspace(-np.pi, np.pi, 721)
    rr = r_of(p)

    ax = axs[0]
    ax.imshow(A.astype(np.uint8))
    ax.plot(cx + rr * np.cos(p), cy + rr * np.sin(p), "-", color="#00e5ff", lw=2.0,
            label="measured outline (4 harmonics, rms 4.7 px)")
    ax.plot(cx + a_axial * np.cos(p), cy + a_equat * np.sin(p), "--", color="#ffd24d", lw=1.4,
            label="fitted ellipse  axial:equatorial = %.3f" % (a_axial / a_equat))
    ax.add_patch(Circle(lens_c, lens_r, fill=False, ec="#7ee081", lw=2.0))
    ax.plot([], [], "-", color="#7ee081", lw=2,
            label="lens  r = %.2f a_eq, reaches the cornea" % (lens_r / a_equat))
    ax.plot(cx, cy, "+", color="#00e5ff", ms=16, mew=2.5)
    ax.annotate("", xy=(cx - a_axial * 1.25, cy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=1.4))
    ax.text(cx - a_axial * 1.22, cy - 12, "optic axis (lateral)", color="#00e5ff", fontsize=9)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax.set_title("globe: %.0f x %.0f um section, flattened to %.2f of its equator\n"
                 "(eye_anatomy.py assumed 0.82 -- the modelled eye was too spherical)"
                 % (2 * a_axial * (EYE_EQUATORIAL_DIAMETER_UM / (2 * a_equat)),
                    EYE_EQUATORIAL_DIAMETER_UM, a_axial / a_equat), fontsize=10)

    ax = axs[1]
    ax.imshow((0.45 * A + 0.55 * 255).astype(np.uint8))
    ax.plot(cx + rr * np.cos(p), cy + rr * np.sin(p), "-", color="#888888", lw=1.2)
    for key, T in tr.items():
        col = MCOL[key]
        d = ins[key]
        ax.contour(bands[key].astype(float), levels=[0.5], colors=[col], linewidths=1.6)
        cl = T["centreline_px"]
        ax.plot(cl[:, 0], cl[:, 1], "-", color=col, lw=1.0, alpha=0.9)
        ax.plot(*d["insertion_px"], "o", mfc="none", mec=col, ms=13, mew=2.4)
        ax.plot(*d["origin_px"], "s", mfc=col, mec="k", ms=7, mew=0.7)
        ax.annotate("%s %s  %+.2f dv" % (key, "dorsal" if d["side"] == "dorsal" else "ventral",
                                         d["u_dorsal"]),
                    d["insertion_px"], textcoords="offset points", xytext=(-6, -14),
                    color=col, fontsize=8.5, fontweight="bold", ha="right",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.8))
    ax.plot([], [], "o", mfc="none", mec="k", ms=10, label="insertion (band end on the globe)")
    ax.plot([], [], "s", mfc="k", mec="k", ms=7, label="origin (geodesically farthest end)")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax.set_title("the six bands, traced: silhouette, centreline, attachments\n"
                 "'dv' = dorso-ventral coordinate recovered from the projection",
                 fontsize=10)

    for ax in axs:
        ax.set_xlim(0, 535)
        ax.set_ylim(696, 0)
        ax.axis("off")
    fig.suptitle("Fig. 12.1A (Tulenko & Currie 2020, after Easter & Nicola 1996) digitized"
                 " -- 96 hpf larval right eye, ventral view", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(os.path.join(OUT, "fig121_digitized.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    m = main()
    G, L = m["globe"], m["lens"]
    print("globe  axial/equatorial = %.3f   (semi %.1f x %.1f px)" %
          (G["axial_over_equatorial"], G["axial_semi_axis_px"], G["equatorial_semi_axis_px"]))
    print("       cardinal semi-diameters px:", G["semi_diameters_px"])
    print("       fore-aft asym %.3f  medio-lateral asym %.3f  fit rms %.2f px" %
          (G["fore_aft_asymmetry"], G["medio_lateral_asymmetry"], G["outline_fit_rms_px"]))
    print("lens   r = %.3f a_eq, centre %.3f a_ax lateral, reaches %.3f a_ax (1.0 = cornea)" %
          (L["radius_over_equatorial_semi_axis"], L["centre_lateral_offset_in_axial_semi_axes"],
           L["lateral_pole_reach"]))
    print("\ninsertion direction on the globe (equatorial semi-axes), then the origin:")
    print("%-4s %7s %7s %7s  %5s %6s   %7s %7s %7s   %6s %6s" %
          ("m", "caud", "dors", "lat", "|u|", "theta", "o_caud", "o_dors", "o_lat",
           "len_um", "wid_um"))
    for k, d in m["muscles"].items():
        print("%-4s %7.3f %7.3f %7.3f  %5.3f %6.1f   %7.3f %7.3f %7.3f   %6.1f %6.1f" %
              (k, d["u_caudal"], d["u_dorsal"], d["u_lateral"], d["radius_in_globe_units"],
               d["theta_from_optic_axis_deg"], d["origin_caudal"], d["origin_dorsal"],
               d["origin_lateral"], d["length_um"], d["mean_width_um"]))
