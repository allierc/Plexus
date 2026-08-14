"""fish_anatomy -- the oculomotor plant AS MEASURED, not as guessed.

`eye_anatomy.py` describes a MAMMAL: four recti from a common annulus of Zinn deep
in the orbit, obliques inserting behind the equator, a trochlea to redirect the
superior oblique, and a globe flattened to 0.82 of its equator. A zebrafish has
none of that. From Tulenko & Currie (2020, ch.12, p.116) and Kasprick et al.
(2011, PLoS ONE 6:e27095):

  * the two OBLIQUES arise together from the ROSTRAL orbit -- the anterior ethmoid
    plate -- and insert on the DORSAL (SO) and VENTRAL (IO) faces of the globe.
    There is no trochlea: the SO pulls from in front, directly.
  * SR, IR and MR arise together from a plate on the CAUDAL orbit. SR and IR insert
    "just caudal to" SO and IO; MR runs forward BETWEEN the two obliques to insert
    on the anteromedial surface.
  * the LATERAL RECTUS is the odd one out: it originates OUTSIDE the orbit,
    immediately posterior to the diencephalon, turns nearly 90 degrees laterally,
    passes lateral to the SR/MR/IR origin, and inserts on the CAUDAL sclera.
  * in the ADULT the insertions have migrated to the sclera-corneal junction and
    SO now shares its insertion with SR, IO with IR (Kasprick fig 2, arrowheads).
    That overlap is a LATE feature -- it is not there in the 96 hpf larva.

Every number below is measured by `digitize_fig121.py` off Fig. 12.1A (a camera
lucida of a 96 hpf larva, ventral view) and read back from
`archive/eye_F/fig121_measurements.json` + `fig121_muscle_trace.npz`. Nothing here
is a round number chosen to look plausible.

THE FRAME IS UNCHANGED from `eye_anatomy`, so every operator still reads the same
axes:  +x caudal (temporal), +y dorsal (superior), +z LATERAL = the optic axis.
For a laterally-eyed fish "anterior along the optic axis" means out of the head,
which is what +z already was.

WHAT THE TRACE IS USED FOR. Not for placing particles: the muscles are still built
by `muscle_ops.strap_path`, the belly-plus-arc-of-contact strap that gives them
their shape. What the trace supplies is the numbers that strap needs and that were
previously guessed -- where each muscle attaches at both ends, how wide it is, and
how flat the globe it wraps is. Lifting the traced silhouette itself onto the globe
was tried and dropped: a ventral projection is singular at the globe's outline, so
every band that reached the rim came back with a fold in it (up to a 129-degree
kink in SR) that was an artifact of inverting the projection, not anatomy.

This module holds no dynamics and registers no operator. It is the CONFIG of a
plant; `eye_spec.build_spec(plant="fish")` reads it and writes the numbers into the
spec, where they belong.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.path.join(HERE, "archive", "eye_F")
MEASUREMENTS = os.path.join(TRACE_DIR, "fig121_measurements.json")
TRACE_NPZ = os.path.join(TRACE_DIR, "fig121_muscle_trace.npz")

MUSCLE_KEYS = ["LR", "SR", "MR", "IR", "SO", "IO"]      # the order eye_anatomy uses
N_MUSCLE = 6
LONG_NAME = {"LR": "lateral rectus", "SR": "superior rectus", "MR": "medial rectus",
             "IR": "inferior rectus", "SO": "superior oblique", "IO": "inferior oblique"}
CRANIAL_NERVE = {"LR": "VI", "SR": "III", "MR": "III", "IR": "III", "SO": "IV", "IO": "III"}
COLOR = {"LR": "#4da3ff", "SR": "#ff5c5c", "MR": "#ffd24d",
         "IR": "#7ee081", "SO": "#c58cff", "IO": "#ff9c42"}

# ---- the globe, in world units (unchanged placement, measured shape) --------- #
GLOBE_CENTER = (0.50, 0.50, 0.46)
A_EQ = 0.115                                   # equatorial semi-axis (x = caudal, y = dorsal)

_M = None
_T = None


def measurements() -> dict:
    global _M
    if _M is None:
        with open(MEASUREMENTS) as fh:
            _M = json.load(fh)
    return _M


def trace() -> dict:
    global _T
    if _T is None:
        _T = dict(np.load(TRACE_NPZ, allow_pickle=False))
    return _T


def axial_ratio() -> float:
    """Axial / equatorial semi-axis of the globe, measured: 0.676.

    `eye_anatomy.AXIAL_RATIO` is 0.82, i.e. the modelled eye was ~20% too round
    along the optic axis. A teleost eye is a flattened ovoid because the lens, not
    the globe's depth, does the focusing.
    """
    return float(measurements()["globe"]["axial_over_equatorial"])


def lens() -> dict:
    """The lens: radius 0.35 of the equatorial semi-axis, centred 0.49 of the AXIAL
    semi-axis out along the optic axis -- so its outer surface reaches 1.007 of the
    axial semi-axis. The zebrafish lens is spherical, huge, and touches the cornea;
    it is not a mammalian lens sitting behind an anterior chamber."""
    L = measurements()["lens"]
    return dict(radius=float(L["radius_over_equatorial_semi_axis"]),
                center_axial=float(L["centre_lateral_offset_in_axial_semi_axes"]),
                reach=float(L["lateral_pole_reach"]))


def insertion_dirs(plant="larva") -> np.ndarray:
    """[6,3] unit insertion directions on the globe-as-a-sphere, in MUSCLE_KEYS order."""
    return np.stack([_unit(_insertion(k, plant)) for k in MUSCLE_KEYS])


def insertions(plant="larva") -> np.ndarray:
    """[6,3] insertion points in globe-local units (equatorial semi-axes)."""
    return np.stack([_insertion(k, plant) for k in MUSCLE_KEYS])


def origins(plant="larva") -> np.ndarray:
    """[6,3] muscle origins in globe-local units (equatorial semi-axes)."""
    return np.stack([_origin(k, plant) for k in MUSCLE_KEYS])


def origins_world(plant="larva") -> np.ndarray:
    return np.asarray(GLOBE_CENTER, float)[None, :] + A_EQ * origins(plant)


def peak_tensions() -> np.ndarray:
    """Relative peak tension per muscle: ALL ONES, on purpose.

    Force is active stress x physiological cross-section, and `strap_widths` gives
    each muscle its measured width against one common thickness -- so the geometry
    already carries every muscle's force. SR ends up twice the pull of MR because it
    is drawn twice as wide, not because a table says so. `eye_anatomy` set these
    weights to 1.00-1.30 by hand; here there is nothing left for them to do.
    """
    return np.ones(N_MUSCLE)


def widths_um() -> np.ndarray:
    return np.array([measurements()["muscles"][k]["mean_width_um"] for k in MUSCLE_KEYS], float)


# --------------------------------------------------------------------------- #
#  the four plants
# --------------------------------------------------------------------------- #
#  larva  -- exactly what Fig. 12.1A shows, at 96 hpf
#  adult  -- Kasprick's adult: insertions migrated to the sclera-corneal junction,
#            SO sharing SR's dorsal insertion and IO sharing IR's ventral one
PLANTS = ("larva", "adult")

# Half-angle from the optic axis of the sclera-corneal junction. The cornea is the
# clear window over the lens; the lens subtends asin(0.35 a_eq / 1.0 a_eq) plus the
# limbal margin, which puts the junction near 60 deg -- the value the adult
# insertions are placed at.
SC_JUNCTION_DEG = 60.0


def _insertion(key, plant):
    m = measurements()["muscles"][key]
    u = np.array([m["u_caudal"], m["u_dorsal"], m["u_lateral"]], float)
    if plant == "larva":
        return u
    if plant != "adult":
        raise ValueError(f"unknown plant {plant!r}; expected one of {PLANTS}")
    # ADULT: four insertion stations on the sclera-corneal junction -- caudal (LR),
    # rostral (MR), dorsal (SR+SO), ventral (IR+IO) -- at SC_JUNCTION_DEG from the
    # optic axis. The pairs SHARE a station, which is the overlap Kasprick reports.
    station = {"LR": (1.0, 0.0), "MR": (-1.0, 0.0),
               "SR": (0.0, 1.0), "SO": (0.0, 1.0),
               "IR": (0.0, -1.0), "IO": (0.0, -1.0)}[key]
    th = math.radians(SC_JUNCTION_DEG)
    v = np.array([station[0] * math.sin(th), station[1] * math.sin(th), math.cos(th)])
    # keep a small separation inside a shared station so the two tendons are
    # adjacent rather than coincident: the rectus sits just caudal of its oblique.
    if key in ("SR", "IR"):
        v[0] += 0.16
    elif key in ("SO", "IO"):
        v[0] -= 0.16
    return _unit(v)


def _origin(key, plant):
    m = measurements()["muscles"][key]
    return np.array([m["origin_caudal"], m["origin_dorsal"], m["origin_lateral"]], float)


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


# --------------------------------------------------------------------------- #
#  what the strap generator needs: widths, thickness
# --------------------------------------------------------------------------- #
EQUATORIAL_SEMI_AXIS_UM = 125.0                # half of the measured 250 um equator


def strap_widths(min_width=0.020) -> np.ndarray:
    """[6] belly width per muscle, in world units, in MUSCLE_KEYS order.

    The measured widths are 10-20 um, i.e. 0.009-0.019 world units, and the MLS-MPM
    cell at n_grid=112 is 0.0089 -- so the narrowest muscle (MR) would be ONE CELL
    across and would simply fall through the grid that is supposed to couple it to
    the sclera. Every width is therefore scaled by ONE common factor until the
    narrowest spans `min_width`. A common factor is the point: it leaves the RATIOS
    measured, and the ratios are what set the relative forces, since a muscle's
    force is its cross-section.
    """
    w = widths_um() / EQUATORIAL_SEMI_AXIS_UM * A_EQ
    return w * max(1.0, float(min_width / w.min()))


def strap_thickness(widths=None) -> float:
    """ONE thickness for all six, 0.62 of the mean width.

    Thickness is set by how many myofibres deep a muscle is, and Kasprick finds much
    the same count in every extraocular muscle -- so it is width, not thickness, that
    distinguishes them. Keeping the thickness common therefore leaves the modelled
    cross-sections proportional to the measured ones.
    """
    w = strap_widths() if widths is None else np.asarray(widths, float)
    return 0.62 * float(np.mean(w))



def summary() -> str:
    M = measurements()
    out = [f"fish plant, measured off {M['source']['figure']} "
           f"({M['source']['redrawn_from']})",
           f"  globe   axial/equatorial {axial_ratio():.3f}   "
           f"({M['globe']['axial_diameter_um']:.0f} x "
           f"{M['globe']['equatorial_diameter_um']:.0f} um)",
           f"  lens    r {lens()['radius']:.3f} a_eq, reaches {lens()['reach']:.3f} of the "
           f"axial semi-axis (1.0 = cornea)",
           "  muscle   insertion (caud, dors, lat)      origin                    len_um  wid_um"]
    for k in MUSCLE_KEYS:
        m = M["muscles"][k]
        out.append("  %-3s %-6s (%6.3f,%6.3f,%6.3f)  (%6.3f,%6.3f,%6.3f)  %6.1f %6.1f"
                   % (k, CRANIAL_NERVE[k], m["u_caudal"], m["u_dorsal"], m["u_lateral"],
                      m["origin_caudal"], m["origin_dorsal"], m["origin_lateral"],
                      m["length_um"], m["mean_width_um"]))
    return "\n".join(out)


if __name__ == "__main__":
    print(summary())
    w = strap_widths()
    print("\nstrap widths (world units, ratios measured, common scale to clear the grid):")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in zip(MUSCLE_KEYS, w)))
    print(f"  one thickness for all six: {strap_thickness():.4f}")
    print("  relative cross-section (= relative force): "
          + "  ".join(f"{k}={v:.2f}" for k, v in zip(MUSCLE_KEYS, w / w.mean())))
    for plant in PLANTS:
        d = insertion_dirs(plant)
        print(f"\n  {plant}: insertion directions")
        for k, v in zip(MUSCLE_KEYS, d):
            print("    %-3s (%+.3f, %+.3f, %+.3f)  theta_from_optic_axis %5.1f deg"
                  % (k, v[0], v[1], v[2], math.degrees(math.acos(max(-1, min(1, v[2]))))))
