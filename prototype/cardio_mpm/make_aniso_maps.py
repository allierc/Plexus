"""make_aniso_maps.py -- structured anisotropy maps for the cardio_mpm aniso spec.

Reproduces the p1_aniso symmetry-breaking STRUCTURE (prototype/cardio/archive/3_p1_aniso) in the
Plexus cardio_mpm world: NO FitzHugh-Nagumo, NO rotary force -- just a global synchronous pulse on
a sheet whose STIFFNESS, GAIN and FIBRE ORIENTATION are spatially structured. The loops then have to
emerge from the material structure (anisotropic active shortening along a spatially-varying fibre),
not from an imposed rotation. The fibre/stiffness/gain are analytic PATTERNS (wavelength/angle/phase)
-- the few parameters that should later be LEARNED (a UNet is too coarse to synthesise this texture).

Writes 3 TIFFs -> $GraphData/graphs_data/material/ :
  map_aniso_stiffness.tif  scalar [0,1] (image field -> youngs via apply_material_map)
  map_aniso_gain.tif       scalar [0,1] (image field -> per-particle gain)
  dir_aniso_fiber.tif      angle/(2pi) in [0,0.5] (vector_grid -> unit fibre axis cos/sin)

aniso_field / fiber_field are copied verbatim from cardio_stage2.py so the texture matches p1_aniso.
Arrays are [row=y(top->bottom), col=x]; the image/vector_grid fields flip vertical (image-top ->
domain-top), same convention as scripts/make_material_maps.py.  Run:  python make_aniso_maps.py
"""
import os
import numpy as np
import tifffile

MAT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/material"
N = 128                                            # grid res (p1_aniso was 137; wavelengths in pixels)


def aniso_field(Hy, Wx, wl, angle=0.0, phase=0.0):
    """Smooth scalar in [0,1] (verbatim from cardio_stage2.aniso_field). wl scalar or [wx,wy]."""
    yy, xx = np.meshgrid(np.arange(Hy), np.arange(Wx), indexing="ij")
    ca, sa = np.cos(angle), np.sin(angle)
    xr, yr = ca * xx + sa * yy, -sa * xx + ca * yy
    wx, wy = (wl if isinstance(wl, (list, tuple)) else (wl, wl))
    f = (np.cos(2 * np.pi * xr / wx + phase) * np.cos(2 * np.pi * yr / wy + 0.5 * phase)
         + 0.5 * np.cos(2 * np.pi * (xr / wx + yr / wy) + phase))
    return ((f - f.min()) / (f.max() - f.min() + 1e-9)).astype(np.float32)


def fiber_field(Hy, Wx, wavelength=16, angle=0.6, amp=np.pi, base=0.0):
    """Per-node fibre ANGLE (rad), smooth mode (verbatim from cardio_stage2.fiber_field)."""
    return (base + amp * aniso_field(Hy, Wx, wavelength, angle, 0.0)).astype(np.float32)


# p1_aniso patterns: stiffness wl=[8,26] phase 0.7 ; gain wl=[26,8] (perpendicular) ; fibre wl 16 angle 0.6
stiff = aniso_field(N, N, [8, 26], angle=0.0, phase=0.7)        # [0,1] striped
gain = aniso_field(N, N, [26, 8], angle=0.0, phase=0.0)         # [0,1] striped, perpendicular
fiber = fiber_field(N, N, wavelength=16, angle=0.6, amp=np.pi)  # angle in [0, pi]
fiber_norm = (fiber / (2 * np.pi)).astype("float32")            # vector_grid: th = val*2pi -> recovers angle

os.makedirs(MAT, exist_ok=True)
for name, arr in [("map_aniso_stiffness", stiff), ("map_aniso_gain", gain), ("dir_aniso_fiber", fiber_norm)]:
    tifffile.imwrite(os.path.join(MAT, f"{name}.tif"), arr.astype("float32"))
    print(f"[tiff] {name}.tif  range [{arr.min():.3f}, {arr.max():.3f}]  shape={arr.shape}")
print(f"fibre angle range: [{fiber.min():.2f}, {fiber.max():.2f}] rad (mod pi axis)")
