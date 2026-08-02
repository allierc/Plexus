"""eye_anatomy -- the ANATOMICAL CONSTANTS of the oculomotor plant (one source of truth).

This module holds no dynamics. It holds the numbers that the operators in `eye_ops.py`
and the renderer in `run_eye.py` both need, so the plant is described ONCE.

Coordinate frame (right eye, head-fixed, world units; the globe centre is the origin
of the local frame and `GLOBE_CENTER` in world coordinates):

    +x   temporal / lateral   (toward the ear; for a zebrafish, toward the tail-ward
                               edge of the laterally-placed eye)
    +y   superior             (up)
    +z   anterior             (forward, along the optic axis at primary gaze)

Rotations follow the right-hand rule about those axes, which gives the clinical
vocabulary directly:

    rotation about +y   ->  gaze moves to +x   ->  ABduction   (horizontal, `h`)
    rotation about -x   ->  gaze moves to +y   ->  elevation   (vertical, `v`)
    rotation about +z   ->  12 o'clock rolls toward -x (nasal) -> INtorsion (`t`)

THE GLOBE IS AN OVOID, not a sphere. Teleost (zebrafish) eyes are flattened along the
optic axis: the equatorial semi-axis `A_EQ` exceeds the axial semi-axis `C_AX`. The
particles are seeded as a uniform ball of radius `A_EQ` by the `mpm_particle` entity and
then affinely squashed z -> AXIAL_RATIO*z at frame 0 (`eye_anatomy` operator), which
preserves uniform density and leaves F = I undeformed. The bony cup, by contrast, stays
SPHERICAL with radius >= A_EQ -- an ovoid cannot rotate inside a matching ovoid socket,
and in the animal the gap is filled by orbital fat, not bone.

THE SIX EXTRAOCULAR MUSCLES. Four recti (lateral / superior / medial / inferior) arise
together from the orbital apex (the annulus of Zinn) and insert on the sclera ANTERIOR to
the equator; two obliques provide torsion and insert POSTERIOR to the equator, pulling
ANTERIORLY -- the superior oblique from the trochlea (a pulley on the superomedial orbital
wall, so its EFFECTIVE origin is the trochlea, not the apex), the inferior oblique from the
anteromedial orbital floor.

The muscle actions are NOT tabulated here. Each muscle's rotation axis is computed at run
time from its geometry as  n_hat x u_hat  (insertion radius x line of action), so the
textbook primary/secondary/tertiary actions must EMERGE. With the constants below they do:

    muscle  torque axis (x, y, z)        reading
    LR      ( 0.03,  1.00, -0.05)        pure abduction
    MR      ( 0.02, -1.00,  0.03)        pure adduction
    SR      (-0.90, -0.17,  0.39)        elevation > intorsion > adduction
    IR      ( 0.90, -0.22, -0.37)        depression > extorsion > adduction
    SO      ( 0.61,  0.14,  0.78)        intorsion > depression > abduction
    IO      (-0.66,  0.18, -0.73)        extorsion > elevation > abduction

The medial tilt of the orbital apex (ORBIT_APEX x-component) is what gives the vertical
recti their torsional and horizontal components; a co-axial apex would make SR/IR pure
elevators/depressors, which is anatomically wrong.
"""
from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- #
#  The globe (ovoid) and its seat
# --------------------------------------------------------------------------- #
GLOBE_CENTER = (0.50, 0.50, 0.46)     # world position of the globe centre
A_EQ = 0.115                          # equatorial semi-axis (x, y)
AXIAL_RATIO = 0.82                    # axial / equatorial -> the ovoid flattening
C_AX = A_EQ * AXIAL_RATIO             # axial semi-axis (z, the optic axis)

CUP_RADIUS = A_EQ + 0.007             # bony cup: SPHERICAL, so the ovoid can rotate in it
CUP_APERTURE_DEG = 62.0               # half-angle of the anterior opening (palpebral aperture)

# radial bands of the globe, in units of the normalized ball radius (pre-squash).
# The `layers` of the spec set the stiffness; these are the labels the renderer uses.
R_VITREOUS = 0.60                     # soft gel core
R_INNER = 0.88                        # choroid / retina
# outside R_INNER: sclera (stiff shell)
R_SHELL = 0.94                        # the outermost band the cosmetic render draws

# the zebrafish lens: large, hard, and pushed ANTERIORLY (it nearly touches the cornea)
LENS_CENTER = (0.0, 0.0, 0.40)        # in units of the normalized ball radius
LENS_RADIUS = 0.34
LENS_YOUNGS = 520.0                   # much stiffer than the sclera -- a rigid ball optic

# cosmetic anterior regions, by polar angle from +z (the optic axis), on the shell
PUPIL_DEG = 30.0                      # black pupil (the photo's large round black disc)
IRIS_DEG = 44.0                       # silvery iris ring around it
# golden iridophore flecks: azimuths (deg, from +x toward +y) of the bright specks
IRIS_FLECK_DEG = (58.0, 128.0, 208.0, 300.0)
IRIS_FLECK_WIDTH_DEG = 13.0

# --------------------------------------------------------------------------- #
#  Muscle geometry.  Positions are in units of the EQUATORIAL semi-axis A_EQ,
#  relative to the globe centre, in the (temporal, superior, anterior) frame.
# --------------------------------------------------------------------------- #
ORBIT_APEX = (-1.10, -0.10, -2.60)    # annulus of Zinn: deep, and tilted ~23 deg MEDIALLY
TROCHLEA = (-0.95, 1.15, 0.55)        # the superior oblique's pulley, superomedial + anterior
IO_ORIGIN = (-0.95, -1.05, 0.60)      # inferior oblique: anteromedial orbital floor (maxilla)

# name, long name, insertion (theta from +z, phi from +x toward +y, both deg),
# effective origin, peak tension (relative), colour
MUSCLES = [
    dict(key="LR", name="lateral rectus",   theta=62.0, phi=0.0,     origin=ORBIT_APEX,
         tension=1.05, cn="VI",  color="#4da3ff"),
    dict(key="SR", name="superior rectus",  theta=66.0, phi=90.0,    origin=ORBIT_APEX,
         tension=1.00, cn="III", color="#ff5c5c"),
    dict(key="MR", name="medial rectus",    theta=55.0, phi=180.0,   origin=ORBIT_APEX,
         tension=1.10, cn="III", color="#ffd24d"),
    dict(key="IR", name="inferior rectus",  theta=60.0, phi=270.0,   origin=ORBIT_APEX,
         tension=1.00, cn="III", color="#7ee081"),
    dict(key="SO", name="superior oblique", theta=122.0, phi=52.0,   origin=TROCHLEA,
         tension=1.30, cn="IV",  color="#c58cff"),
    dict(key="IO", name="inferior oblique", theta=128.0, phi=-48.0,  origin=IO_ORIGIN,
         tension=1.30, cn="III", color="#ff9c42"),
]

MUSCLE_KEYS = [m["key"] for m in MUSCLES]
N_MUSCLE = len(MUSCLES)

INSERTION_PATCH_DEG = 22.0            # angular half-width of the tendon insertion arc
WRAP = 0.85                           # fraction of the line-of-action's NORMAL component
                                      # removed -> the muscle wraps the sclera (arc of
                                      # contact) instead of pulling through the globe


def unit_from_spherical(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Unit vector at polar angle `theta` from +z, azimuth `phi` from +x toward +y."""
    th, ph = math.radians(theta_deg), math.radians(phi_deg)
    return np.array([math.sin(th) * math.cos(ph),
                     math.sin(th) * math.sin(ph),
                     math.cos(th)], dtype=np.float64)


def insertion_dirs() -> np.ndarray:
    """[6, 3] unit insertion directions in the globe's REST frame (pre-squash sphere)."""
    return np.stack([unit_from_spherical(m["theta"], m["phi"]) for m in MUSCLES])


ANNULUS_RING = 0.42          # radius of the annulus of Zinn, in units of A_EQ


def origins_world() -> np.ndarray:
    """[6, 3] effective origin in WORLD coordinates.

    The four recti do not arise from a point: they arise from the ANNULUS OF ZINN, a fibrous
    RING at the orbital apex, each from the sector facing its own insertion. Offsetting each
    rectus origin by `ANNULUS_RING` in its own azimuthal direction is both more accurate and
    what stops all four from projecting onto a single blob. The obliques keep their own
    origins (trochlea, orbital floor)."""
    c = np.asarray(GLOBE_CENTER, float)
    out = []
    for m in MUSCLES:
        o = c + A_EQ * np.asarray(m["origin"], float)
        if m["origin"] is ORBIT_APEX:
            ph = math.radians(m["phi"])
            out.append(o + A_EQ * ANNULUS_RING * np.array([math.cos(ph), math.sin(ph), 0.0]))
        else:
            out.append(o)
    return np.stack(out)


def peak_tensions() -> np.ndarray:
    return np.array([m["tension"] for m in MUSCLES], float)


def squash(v: np.ndarray) -> np.ndarray:
    """Apply the ovoid flattening to a vector/array of local (globe-frame) offsets."""
    out = np.asarray(v, float).copy()
    out[..., 2] *= AXIAL_RATIO
    return out


def belly_centers() -> np.ndarray:
    """[6, 3] a rough world position for each muscle BELLY -- only used to seed the
    `muscle` parent set, since `muscle_morphogenesis` then places every point exactly."""
    c = np.asarray(GLOBE_CENTER, float)
    ins = insertion_dirs() * A_EQ + c
    org = origins_world()
    return 0.5 * (ins + org)
