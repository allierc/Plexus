"""segmentation_seed -- a measured instance segmentation becomes the CELL level of the hierarchy.

WHAT THIS IS FOR
================================================================================================
`apply_material_map` gives a tissue a heterogeneous material from a greyscale image: every particle
samples an intensity and becomes a little stiffer or softer than its neighbour. That is a CONTINUUM
with a pattern painted on it. It has no cells in it.

This operator does something different. It reads an INSTANCE SEGMENTATION -- an integer label map,
one id per cell -- and uses it to populate the middle level of the hierarchy the paper describes:

    tissue  ->  cells  ->  material points

A cell is then a real entity with its own identity, its own material, and its own set of particles,
rather than a region of a map. Particles belonging to the same cell share a Young's modulus exactly;
particles either side of a boundary do not. That is the difference between a stiffness gradient and
a tissue made of cells, and it is the difference the MPM sees.

TWO THINGS IT HAS TO GET RIGHT, AND BOTH ARE EASY TO GET WRONG
------------------------------------------------------------------------------------------------
LABELS MAY NOT BE INTERPOLATED. Cell 7 next to cell 12 must not produce cell 9 or 10 in between.
`ImageField` normalises to [0,1] and is sampled bilinearly, which is right for a stiffness map and
silently catastrophic for a label map. Hence `label_image`: no normalisation, nearest neighbour,
integers in and integers out.

A CELL IS NOT A DISC. The engine scatters a contained set in a ball about its parent, which is
right for a generic child and wrong for a cell whose shape was measured. This operator re-seeds:
each cell entity moves to the centroid of its own label, and each of its particles is placed at a
random point INSIDE that label. The shapes in the simulation are then the shapes in the microscope.

It runs ONCE. Seeding is not a per-frame force -- it establishes the configuration and then gets
out of the way, so every call after the first returns immediately.

    fields:  {cells: {frame: label_image, source: material/cardio_cells_label.tif}}
    ops:     {op: seed_from_segmentation, at: mpm_particle, from: cells,
              cell_set: cell, youngs_min: 40, youngs_max: 220, props: material/props.json}
"""
from __future__ import annotations

import json
import os

import torch

from plexus.models.base import Field, Exchange
from plexus.models.registry import register_field, register_operator
from plexus.paths import graphs_data_path


@register_field("label_image", frame="label_image")
class LabelImageField(Field):
    """An integer instance map read from a TIFF. NOT normalised, NEVER interpolated.

    The one job it has that `image` cannot do: return the id that is actually there. Bilinear
    weights between label 7 and label 12 are a number that means nothing and points at a cell that
    may not exist, so `sample_label` indexes rather than interpolates.
    """

    def __init__(self, name, source=None, res=None, width=1.0, device="cpu", **kw):
        super().__init__(name)
        if source is None:
            raise ValueError(f"label_image field {name!r} needs a `source:` (path to a label .tif)")
        import tifffile
        path = source if os.path.isabs(source) else graphs_data_path(source)
        img = tifffile.imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        img = img[::-1, :].copy()                       # image-top -> domain-top, as ImageField
        v = torch.tensor(img.astype("int64"), device=device).permute(1, 0).contiguous()
        self.C = 1
        self.nx, self.ny = int(v.shape[0]), int(v.shape[1])
        self.width = float(width)
        self.R = self.nx / self.width
        self.register_buffer("grid", v[None])           # [1, nx, ny] int64 labels
        self.n_labels = int(v.max())

    def sample_label(self, pos):
        """[N,2] world positions -> [N] integer label, nearest neighbour."""
        x = pos[:, 0].clamp(0, self.width - 1e-6) / self.width * self.nx
        y = pos[:, 1].clamp(0, self.width - 1e-6) / self.width * self.ny
        gx = x.long().clamp(0, self.nx - 1)
        gy = y.long().clamp(0, self.ny - 1)
        return self.grid[0][gx, gy]


@register_operator("seed_from_segmentation", family="seed", set="particle", kind="exchange")
class SeedFromSegmentation(Exchange):
    """Populate tissue -> cell -> particle from a measured instance segmentation. Runs once."""

    EMIT = None
    # STRUCTURAL, not dynamics. It establishes the configuration -- where the cells are and which
    # particles belong to them -- and writes the state buffer directly to do it. The engine's
    # integration invariant forbids that for a force, correctly, so the exemption is declared here
    # and paid for by running exactly once.
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["from"]
    SUPPORTED_DIMS = [2]
    MECHANISM_TAGS = ["instance_segmentation", "cell_identity", "heterogeneous_material"]
    PARAM_ROLES = {"youngs_min": "param_lo", "youngs_max": "param_hi",
                   "from": "label_field", "cell_set": "middle_level"}
    REFERENCE = "instance segmentation measured from the beat; see prototype/cardio_cells"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.at = params.get("_at", "mpm_particle")
        self.cell_set = params.get("cell_set", "cell")
        self.y_lo = float(params.get("youngs_min", 40.0))
        self.y_hi = float(params.get("youngs_max", 220.0))
        self.props = params.get("props")               # optional measured per-cell json
        self.jitter = float(params.get("jitter", 0.0))
        self._done = False

    def _cell_values(self, n_cells, device):
        """Per-cell Young's modulus: from the MEASURED beat when a props file is given, else a
        deterministic spread so the tissue is heterogeneous but reproducible.

        Measured is the interesting case. A cell that moved little in the recording is either stiff
        or weakly contractile; mapping amplitude to stiffness INVERSELY is one hypothesis about
        which, it is stated here rather than hidden, and the alternative (amplitude -> contraction
        gain) is the same one line the other way round.
        """
        if self.props:
            path = self.props if os.path.isabs(self.props) else graphs_data_path(self.props)
            if os.path.exists(path):
                d = json.load(open(path))
                amp = torch.tensor([d.get(str(k), {}).get("amp", float("nan"))
                                    for k in range(1, n_cells + 1)], device=device)
                good = torch.isfinite(amp)
                if good.any():
                    a = amp.clone()
                    a[~good] = a[good].median()
                    lo, hi = torch.quantile(a, 0.05), torch.quantile(a, 0.95)
                    u = ((a - lo) / (hi - lo + 1e-9)).clamp(0, 1)
                    return self.y_lo + (1.0 - u) * (self.y_hi - self.y_lo), "measured beat amplitude"
        g = torch.Generator(device="cpu").manual_seed(12345)
        u = torch.rand(n_cells, generator=g).to(device)
        return self.y_lo + u * (self.y_hi - self.y_lo), "deterministic spread (no props file)"

    def forward(self, H, mask=None):
        if self._done:
            return {}
        self._done = True
        lvl = H.level(self.at)
        fld = H.fields[self.field_name]
        dev = lvl.state.device
        px0, px1 = lvl.state_schema["pos"]
        n_cells = int(fld.n_labels)

        # ---- where each label lives, in world coordinates ---------------------------------
        gridl = fld.grid[0]                                     # [nx,ny] int64
        nx, ny = gridl.shape
        gx, gy = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev),
                                indexing="ij")
        flat = gridl.reshape(-1)
        wx = (gx.reshape(-1).double() + 0.5) / nx * fld.width
        wy = (gy.reshape(-1).double() + 0.5) / ny * fld.width
        inside = flat > 0
        lab_in, wx_in, wy_in = flat[inside], wx[inside], wy[inside]
        cnt = torch.bincount(lab_in, minlength=n_cells + 1).clamp(min=1)
        cx = torch.bincount(lab_in, weights=wx_in, minlength=n_cells + 1) / cnt
        cy = torch.bincount(lab_in, weights=wy_in, minlength=n_cells + 1) / cnt

        # ---- the CELL level moves onto its own segmented cell -----------------------------
        moved_cells = 0
        if self.cell_set in H.levels:
            cl = H.level(self.cell_set)
            cx0, cx1 = cl.state_schema["pos"]
            m = min(cl.n, n_cells)
            st = cl.state.clone()
            st[:m, cx0] = cx[1:m + 1].float()
            st[:m, cx0 + 1] = cy[1:m + 1].float()
            cl.state = st
            moved_cells = m
            if cl.n != n_cells:
                print(f"  [seed_from_segmentation] the {self.cell_set!r} set has {cl.n} entities "
                      f"and the map has {n_cells} cells -- seeding {m}. Declare "
                      f"per_parent: {n_cells} to use all of them.", flush=True)

        # ---- each particle is placed INSIDE its own cell's mask ---------------------------
        # ordering by label makes the members of one cell contiguous, so a particle can be given a
        # pixel of its OWN cell by index arithmetic instead of a python loop over 472 cells
        order = torch.argsort(lab_in)
        lab_s, wx_s, wy_s = lab_in[order], wx_in[order], wy_in[order]
        start = torch.cumsum(torch.bincount(lab_s, minlength=n_cells + 1), 0) - \
            torch.bincount(lab_s, minlength=n_cells + 1)

        pidx = lvl.parent if lvl.parent is not None else torch.zeros(lvl.n, dtype=torch.long,
                                                                     device=dev)
        pcell = (pidx % n_cells) + 1 if lvl.parent is not None else None
        if pcell is None or moved_cells == 0:
            # no declared cell level: assign each particle the label it already sits on
            pos = lvl.state[:, px0:px1]
            cid = fld.sample_label(pos)
        else:
            cid = pcell.clamp(1, n_cells)
            g = torch.Generator(device="cpu").manual_seed(777)
            u = torch.rand(lvl.n, generator=g).to(dev)
            k = (start[cid] + (u * cnt[cid].float()).long().clamp(max=0 + cnt[cid] - 1))
            k = k.clamp(0, lab_s.numel() - 1)
            newpos = torch.stack([wx_s[k].float(), wy_s[k].float()], 1)
            if self.jitter > 0:
                newpos = newpos + (torch.rand_like(newpos) - 0.5) * self.jitter
            st = lvl.state.clone(); st[:, px0:px1] = newpos; lvl.state = st

        # ---- one material per cell, shared exactly by its particles -----------------------
        yc, how = self._cell_values(n_cells, dev)
        y_all = torch.cat([yc[:1], yc])                          # index 0 = background, unused
        p_y = y_all[cid.clamp(0, n_cells)]
        from plexus.models.entities import _lame
        mu, la = _lame(p_y)
        liquid = getattr(lvl, "is_liquid", None)
        if liquid is not None:
            mu = torch.where(liquid, torch.zeros_like(mu), mu)
        lvl.mu, lvl.la = mu, la
        for nm, val in (("youngs", p_y), ("cell_id", cid.float())):
            if nm in getattr(lvl, "_buffers", {}):
                setattr(lvl, nm, val)
            else:
                lvl.register_buffer(nm, val)

        print(f"  [seed_from_segmentation] {n_cells} cells from {self.field_name!r}; "
              f"{lvl.n} particles ({lvl.n / max(n_cells,1):.0f} per cell); "
              f"youngs {float(yc.min()):.0f}-{float(yc.max()):.0f} from {how}; "
              f"cell centres seeded: {moved_cells}", flush=True)
        return {}
