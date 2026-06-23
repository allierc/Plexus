"""material_map -- a static image FIELD read from a TIFF, plus `apply_material_map`,
the field -> set operator that writes per-particle MATERIAL parameters from it.

This is the Plexus way to give a tissue a heterogeneous, image-defined material: an
image (TIFF/PNG) becomes a scalar field over the domain (`image` field), and
`apply_material_map` samples it at each particle and turns the intensity into a
per-particle parameter -- Young's modulus E_i -> Lame mu_i, la_i (target: youngs), or
any named per-particle buffer (e.g. a future per-particle contraction `gain`). It is
NOT a force: it sets the material the MLS-MPM stress law (mpm_strain / p2g) then reads,
so a stiffness map breaks the contraction's symmetry WITHOUT changing MPM physics.

    fields:  {stiffness_map: {frame: image, source: material/map.tif}}
    ops:     {op: apply_material_map, at: mpm_particle, from: stiffness_map,
              target: youngs, min: 20, max: 200}

Anisotropy (E_x != E_y, fibre angle) would need a tensor map + an anisotropic stress
law -- a later step; this does the scalar case the current fixed-corotated MPM supports.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as Fnn

from plexus.models.base import Field, Exchange
from plexus.models.registry import register_field, register_operator
from plexus.paths import graphs_data_path


@register_field("image", frame="image")
class ImageField(Field):
    """A 1-channel scalar field read from a 2D image (TIFF/PNG), normalised to [0,1].
    A STATIC map (no dynamics): it holds only its grid `[1, nx, ny]` and the
    world<->pixel geometry, sampled by `apply_material_map`. Same orientation
    convention as `VideoField` (flip vertical so image-top maps to domain-top)."""

    def __init__(self, name, source=None, res=None, width=1.0, device="cpu",
                 normalize=True, **kw):
        super().__init__(name)                                  # binds to no set (no couples_to)
        if source is None:
            raise ValueError(f"image field {name!r} needs a `source:` (path to a .tif/.png)")
        import tifffile
        path = source if os.path.isabs(source) else graphs_data_path(source)
        img = tifffile.imread(path).astype("float32")          # [ny, nx] (image rows top->bottom)
        if img.ndim == 3:                                      # collapse any channels to grayscale
            img = img.mean(axis=-1)
        img = img[::-1, :].copy()                              # flip vertical: image-top -> domain-top
        if normalize:
            lo, hi = float(img.min()), float(img.max())
            img = (img - lo) / (hi - lo + 1e-9)                # -> [0,1]
        v = torch.tensor(img, device=device).permute(1, 0).contiguous()   # [ny,nx] -> [nx,ny]
        self.C = 1
        self.nx, self.ny = int(v.shape[0]), int(v.shape[1])
        self.width = float(width)
        self.R = self.nx / self.width                          # pixels per world unit (x)
        self.register_buffer("grid", v[None])                  # [1, nx, ny]

    def pix(self, x, y):
        gx = (x.clamp(0, self.width - 1e-6) / self.width * self.nx).long().clamp(0, self.nx - 1)
        gy = (y.clamp(0, 1 - 1e-6) * self.ny).long().clamp(0, self.ny - 1)
        return gx, gy


@register_field("vector_grid", frame="vector_grid")
class VectorGrid(Field):
    """A 2-channel UNIT-VECTOR field d(x) = (dx, dy) read from a TIFF -- the contraction
    DIRECTION / active-stress-orientation map. A 2-channel TIFF `[ny,nx,2]` is read as
    (dx, dy); a 1-channel TIFF as an angle theta in [0,1]->[0,2pi) -> (cos, sin). Every
    vector is normalised to unit length. Same vertical-flip convention as ImageField."""

    def __init__(self, name, source=None, res=None, width=1.0, device="cpu", **kw):
        super().__init__(name)
        if source is None:
            raise ValueError(f"vector_grid field {name!r} needs a `source:` (path to a .tif)")
        import tifffile
        import numpy as np
        path = source if os.path.isabs(source) else graphs_data_path(source)
        img = tifffile.imread(path).astype("float32")
        img = img[::-1].copy()                                 # flip vertical (image-top -> domain-top)
        if img.ndim == 2:                                      # angle map theta in [0,1] -> [0,2pi)
            th = img * (2 * np.pi)
            dx, dy = np.cos(th), np.sin(th)
        else:                                                  # [ny,nx,2] vector map (dx, dy), [-1,1]
            dx, dy = img[..., 0], img[..., 1]
        v = np.stack([dx, dy], axis=0)                         # [2, ny, nx]
        n = np.sqrt(v[0] ** 2 + v[1] ** 2); n[n < 1e-9] = 1.0
        v = (v / n).astype("float32")                          # unit vectors
        vt = torch.tensor(v, device=device).permute(0, 2, 1).contiguous()   # [2, nx, ny]
        self.C = 2
        self.nx, self.ny = int(vt.shape[1]), int(vt.shape[2])
        self.width = float(width)
        self.R = self.nx / self.width
        self.register_buffer("grid", vt)                       # [2, nx, ny]


@register_operator("apply_material_map", level="particle", kind="exchange")
class ApplyMaterialMap(Exchange):
    """field -> set: sample the map at each particle and write a per-particle material
    parameter. `target: youngs` maps intensity in [0,1] to E in [min,max] and sets the
    Lame buffers mu/la (the MPM stress law reads them); any other `target` is written as
    a per-particle buffer of that name. Mutates per-particle buffers, returns {}."""

    PREDICTION = None                              # sets material, emits no force
    REQUIRES_PARAMS = ["from", "target"]
    SUPPORTED_DIMS = [2, 3]
    MECHANISM_TAGS = ["material_map", "heterogeneous_stiffness", "symmetry_breaking"]
    PARAM_ROLES = {"min": "param_lo", "max": "param_hi", "target": "material_parameter"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.target = str(params.get("target", "youngs"))
        self.lo = float(params.get("min", 20.0))
        self.hi = float(params.get("max", 200.0))
        self.channel = int(params.get("channel", 0))
        self.at = params.get("_at", "mpm_particle")

    def _sample(self, H, lvl):
        """Bilinear-sample the map field at the particle positions -> intensity [N]."""
        pos = lvl.get("pos")
        g = H.fields[self.field_name].grid[self.channel]            # [nx, ny]
        W = float(getattr(H, "world_width", 1.0))
        gxn = (pos[:, 0] / W) * 2 - 1
        gyn = (pos[:, 1] / 1.0) * 2 - 1
        grid = torch.stack([gyn, gxn], -1)[None, None]             # grid_sample expects (x=ny, y=nx)
        val = Fnn.grid_sample(g[None, None], grid, mode="bilinear",
                              padding_mode="border", align_corners=True)[0, 0, 0]
        return val.clamp(0.0, 1.0)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        mapped = self.lo + self._sample(H, lvl) * (self.hi - self.lo)   # intensity -> [lo, hi]
        if self.target == "youngs":
            from plexus.models.entities import _lame
            mu, la = _lame(mapped)
            liquid = getattr(lvl, "is_liquid", None)
            if liquid is not None:                                 # liquid keeps zero shear modulus
                mu = torch.where(liquid, torch.zeros_like(mu), mu)
            lvl.mu, lvl.la = mu, la                                # MPM stress reads these
            if "youngs" in getattr(lvl, "_buffers", {}):
                lvl.youngs = mapped
            else:
                lvl.register_buffer("youngs", mapped)
        else:
            if self.target in getattr(lvl, "_buffers", {}):
                setattr(lvl, self.target, mapped)
            else:
                lvl.register_buffer(self.target, mapped)
        return {}
