"""The cell as a COMPOSITION: one cell, many subcellular compartments, each compartment a
cloud of material points.

This module exists to exercise the claim of plexus2 sec. Hierarchy -- that a parent may hold
several distinct sets at once, and that a level is a heterogeneous collection rather than one
biological category -- on the object that claim is usually stated about, a eukaryotic cell.
The reference picture is `figures/cell_atlas.png`: an interactive cytology viewer that reports
a cell as 4,177 modelled PIECES distributed over 15 compartments (421 plasma-membrane pieces,
116 cytoskeletal filaments, 138 nuclear-envelope pieces, 31 chromatin strands, 2 nucleoli,
11 rough-ER cisternae, 3 smooth-ER cisternae, 14 Golgi cisternae, 77 mitochondria, and six
further compartments the screenshot's list scrolls past). A piece is not a cell and not a
material point; it is the intermediate level the atlas actually counts, and Plexus has to be
able to name it.

    cell            1 element              the body
      compartment   813 elements           a PIECE of an organelle -- one membrane patch, one
                                           mitochondrion, one Golgi cisterna. Carries `types`,
                                           one type per compartment of the atlas.
        mpm_particle  813,000 elements     material points, 1,000 per piece

Three levels, two containment maps, and the whole thing is declared in `sets:` -- the engine's
`build` already walks a chain of any depth, because pass 2 resolves children in declaration
order and a child's parent only has to exist by the time its own row is reached.

In the order they appear below:

    compartment      entity     a piece of an organelle: position, velocity, nothing else
    seed_cell_atlas  seed       lay the atlas out once, at x_0: piece centres, piece frames,
                                the material points inside each piece, and the per-particle
                                volume the piece's geometry implies
    aggregate_centroid aggregate  a parent's position as the mass-weighted mean of its
                                children's -- sum_pi, scheduled once per containment map

WHAT THE SEED DOES AND WHY IT IS NOT A `sets:` KEY. The engine's own placement scatters a
contained set uniformly in a ball about its parent, which is the right default and is exactly
wrong here: an atlas is not a fuzzball. A plasma-membrane piece belongs ON a sphere, tangent to
it, and its 1,000 points fill a curved patch; a mitochondrion is a capsule somewhere in the
cytoplasm but never inside the nucleus; a chromatin strand is a random walk confined to the
nucleus. That is a MECHANISM for establishing x_0 -- reusable, parameterised, shared between
specs -- which is what `seed:` is for (plexus2, "The seed window").

EVERY LENGTH IN THE ATLAS TABLE IS A FRACTION OF THE CELL RADIUS, never a world coordinate, so
one number (`cell_radius`, in world units) rescales the whole cell and the atlas stays a
statement about proportions. `size: 0.105` on the plasma membrane means "a patch whose radius
is 0.105 OF THE CELL'S RADIUS"; `r: 0.42` on the nuclear envelope means "a shell at 0.42 OF THE
CELL'S RADIUS".
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Aggregate, Lateral, Seed
from plexus.models.registry import register_entity, register_operator
from plexus.models.state import spatial_schema


# --------------------------------------------------------------------------- the entity
@register_entity(
    "compartment", "organelle", depth=1,
    state_schema=spatial_schema,                 # dim -> StateSchema (pos|vel, D-wide each)
    render={"color_by": "node_type", "arrows": None},
)
class Compartment:
    """One PIECE of a subcellular compartment: a single membrane patch, a single mitochondrion,
    a single Golgi cisterna.

    It is a set in its own right rather than a label on the material points, because the atlas
    counts pieces and because the two levels carry different state: a piece has a position, an
    orientation and an identity (`node_type` says which compartment it belongs to), while a
    material point has a deformation gradient and a mass. Collapsing them would make "how many
    mitochondria" unanswerable without clustering the point cloud.

    `depth=1` is the same scale hint `cell` carries, and that is not a contradiction: depth is a
    hint used only in `Level.__repr__`, while the containment chain that the engine and the
    operators actually traverse is `parent`/`parent_name`. Nothing dispatches on depth.
    """


# --------------------------------------------------------------------------- the atlas table
#
# GEOMETRY ONLY. The COUNTS and the MATERIALS live on `sets.compartment.types` in the spec --
# the counts because `_assign_types` is what turns them into `node_type`, the materials because
# `MPMParticle.provision` reads `youngs`/`material` off the parent's types at build time, before
# any seed runs. Stating either of them here as well would be a second chance to disagree with
# the spec, which is the failure mode `bulk_modulus` beside `youngs` is refused for.
#
# Fields, all lengths as a fraction of `cell_radius`:
#   place  shell   piece centres on a sphere of radius `r`, laid out on a Fibonacci spiral
#          ball    piece centres uniform in the shell between radii `r_in` and `r_out`
#   axis   radial  the piece's `e3` points away from the cell centre
#          random  `e3` is an isotropic random direction
#   shape  patch     a curved cap of a sphere: a disc of radius `size` in the tangent plane,
#                    `thickness` deep radially. `e3` is the patch NORMAL.
#          sheet     a flat disc of radius `size`, `thickness` deep. `e3` is the sheet NORMAL.
#          rod       a cylinder of half-length `size` and radius `thickness`. `e3` is the AXIS.
#          ball      a solid ball of radius `size`.
#          filament  a persistent random walk of contour length `size` and radius `thickness`;
#                    `persistence` in [0, 1] is 0 for a freely-jointed coil and 1 for a straight
#                    rod. `e3` is the direction of the FIRST segment.
#          cisterna  a BOWED, FENESTRATED disc: radius `size`, `thickness` deep, sagging by
#                    `bow` x `size` along `e3`, with `holes` circular fenestrations of radius
#                    `hole_r` x `size` punched through it. Rough ER.
#          tubule    a BRANCHING TUBULAR NETWORK: a tree of `segments` tubes of radius
#                    `thickness`, each `size`/`segments` long, branching with probability
#                    `branch`. Smooth ER.
#          stack     `layers` bowed cisternae stacked along `e3`, separated by `gap` x `size`
#                    and tapering to `taper` of the first by the last. Golgi.
#          mito      a BENT capsule of half-length `size` and radius `thickness`, curved by
#                    `bend` (0 straight, 1 a half-circle), with `cristae` transverse inner
#                    lamellae taking `crista_frac` of the points. Mitochondrion.
#          barrel    `blades` rods on a circle of radius `size`, each `length` x `size` long and
#                    `thickness` thick, tilted by `skew` -- the 9 triplet microtubules of a
#                    centriole.
#
# WHY THE ORGANELLE SHAPES ARE NOT ALL DISCS AND CAPSULES. They were, and the picture said so: a
# Golgi drawn as one flat disc is indistinguishable from an ER cisterna drawn as one flat disc,
# and a mitochondrion drawn as a straight cylinder is indistinguishable from a piece of
# cytoskeleton. The shapes below are what makes the atlas an atlas rather than nine colours --
# a Golgi is a STACK, an ER is a bowed fenestrated sheet or a branching tube network, a
# mitochondrion is bent and has cristae, a centriole is a nine-fold barrel. Each is still a
# closed-form point sampler with a closed-form volume, so `p_vol` stays exact.
#
# The ten compartments below are the nine the screenshot's list shows, plus the centriole. The
# remaining five of the atlas's fifteen are not invented here -- add them to the spec's `types:`
# and to this table (or to the operator's `geometry:` override) when the list is known.
ATLAS: dict[str, dict] = {
    # the bounding shell: 421 patches on the cell surface. `size` is set just above the
    # equal-area spacing of a 421-point Fibonacci spiral (2/sqrt(421) = 0.0975 of the radius),
    # so neighbouring patches overlap by about 8% of their width and the shell is continuous
    # rather than perforated -- a perforated shell is not a membrane, it is a colander.
    "plasma_membrane": dict(place="shell", r=1.00, axis="radial",
                            shape="patch", size=0.105, thickness=0.05),
    # filaments spanning the cytoplasm, laid RADIALLY and nearly straight (persistence 0.88), each
    # centred between the nuclear envelope and the membrane. A contour length of 0.60 R centred in
    # the band 0.62-0.78 R reaches roughly 0.35 R to 1.02 R -- so the outer ends ANCHOR IN THE
    # MEMBRANE rather than stopping short of it, which is what a cytoskeleton does and what a
    # filament floating free in the cytoplasm does not. The inner ends stop outside the nuclear
    # envelope (0.42 R + its thickness), so the skeleton does not thread the nucleus.
    "cytoskeleton":    dict(place="ball", r_in=0.62, r_out=0.78, axis="radial",
                            shape="filament", size=0.60, thickness=0.006, persistence=0.88),
    # the nuclear ENVELOPE, not the nuclear volume: 138 patches on a sphere at 0.42 of the cell
    # radius. Same Fibonacci-spacing argument as the plasma membrane (2/sqrt(138) = 0.170 of
    # THAT sphere's radius = 0.0715 of the cell radius).
    "nucleus":         dict(place="shell", r=0.42, axis="radial",
                            shape="patch", size=0.078, thickness=0.040),
    # chromatin: coiled strands inside the envelope (persistence 0.25 -- a loose coil).
    "chromatin":       dict(place="ball", r_in=0.05, r_out=0.24, axis="random",
                            shape="filament", size=0.34, thickness=0.018, persistence=0.25),
    "nucleolus":       dict(place="ball", r_in=0.06, r_out=0.20, axis="random",
                            shape="ball", size=0.085),
    # ROUGH ER: broad bowed cisternae wrapped around the nucleus, perforated. The `axis: radial`
    # puts each sheet's NORMAL along the radius, so the sheet itself lies tangentially -- which is
    # how the reference figure shows the teal ER, as curved shells rather than as discs at
    # arbitrary angles. `holes` is what makes it read as ER and not as a plate.
    "rough_er":        dict(place="ball", r_in=0.52, r_out=0.70, axis="radial",
                            shape="cisterna", size=0.26, thickness=0.012,
                            bow=0.35, holes=9, hole_r=0.16),
    # SMOOTH ER: the tubular half of the same organelle, so a different SHAPE and not a smaller
    # sheet. A branching network of tubes is what distinguishes smooth from rough under EM.
    "smooth_er":       dict(place="ball", r_in=0.54, r_out=0.78, axis="random",
                            shape="tubule", size=0.90, thickness=0.010,
                            segments=16, branch=0.45, persistence=0.55),
    # GOLGI: a stack of cisternae, cis to trans, each smaller than the last. One disc was a plate;
    # five bowed discs 0.11 x size apart is the organelle.
    "golgi":           dict(place="ball", r_in=0.52, r_out=0.68, axis="radial",
                            shape="stack", size=0.075, thickness=0.008,
                            layers=6, gap=0.30, bow=0.40, taper=0.62),
    # MITOCHONDRIA: bent capsules with transverse cristae, anywhere in the cytoplasm and with no
    # preferred direction. `crista_frac` of the points go on the inner lamellae, which is what
    # makes a cut-open mitochondrion look like one.
    "mitochondria":    dict(place="ball", r_in=0.52, r_out=0.80, axis="random",
                            shape="mito", size=0.12, thickness=0.025,
                            bend=0.55, cristae=7, crista_frac=0.40),
    # CENTRIOLE: two of them, near the nucleus, as nine-fold barrels of triplet microtubules.
    # Placed close together because a centrosome is a PAIR at right angles; the right angle is not
    # imposed here (each barrel takes its own random axis), which is the one piece of centrosome
    # anatomy this shape does not yet carry.
    "centriole":       dict(place="ball", r_in=0.50, r_out=0.58, axis="random",
                            shape="barrel", size=0.011, thickness=0.0026,
                            blades=9, length=4.5, skew=0.30),
    # THE CROWD. Five protein species filling the whole cell -- one PIECE each, a ball of the
    # cell's own radius, drawn as dots. They are not an organelle and they are the reason a cell is
    # not hollow: a real cytoplasm is 20-30% protein by volume, and a shell with vacuum inside
    # pancakes on impact instead of bouncing, which is exactly what the first run did.
    #
    # ONE PIECE EACH, NOT A CLOUD OF PIECES, because a protein species is not counted in pieces --
    # it is a concentration. `volume_frac` is what keeps five co-located balls from claiming five
    # cell volumes of mass between them.
    **{f"protein_{k}": dict(place="ball", r_in=0.0, r_out=0.0, axis="random",
                            shape="ball", size=0.95, volume_frac=0.12)
       for k in ("a", "b", "c", "d", "e")},
}

_SHAPES = ("patch", "sheet", "rod", "ball", "filament",
           "cisterna", "tubule", "stack", "mito", "barrel")
_PLACES = ("shell", "ball")
_AXES = ("radial", "random")


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _ortho(e3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Any orthonormal (e1, e2) completing `e3` to a right-handed frame, per row.

    The seed vector is swapped away from `e3` when the two are nearly parallel, because a cross
    product of nearly parallel vectors is dominated by round-off and the frame it yields spins
    with the last bit of the input -- which would make a patch's orientation, and therefore the
    picture, non-reproducible for exactly the pieces sitting on the x axis.
    """
    a = torch.zeros_like(e3); a[:, 0] = 1.0
    b = torch.zeros_like(e3); b[:, 1] = 1.0
    a = torch.where((e3[:, 0].abs() > 0.9)[:, None], b, a)
    e1 = _unit(torch.cross(a, e3, dim=1))
    e2 = torch.cross(e3, e1, dim=1)
    return e1, e2


def _fibonacci(n: int, gen: torch.Generator) -> torch.Tensor:
    """`n` near-uniformly spaced unit vectors (the spherical Fibonacci spiral).

    NOT `randn`-normalised directions, and the difference is the whole appearance of the
    membrane: independent random directions clump, so at 421 points a random shell has bald
    patches several patch-widths across and a piece count that means nothing locally. The
    spiral's nearest-neighbour spacing is uniform to a few percent, which is what lets `size`
    be set from the count.
    """
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = (1.0 - z * z).clamp_min(0.0).sqrt()
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    d = torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=1).float()
    # A RANDOM ROTATION PER RUN, so the spiral's own pole (a visible seam where the spacing is
    # least uniform) is not always at +z and the seam cannot be mistaken for a result.
    q = torch.randn(3, 3, generator=gen)
    q, _ = torch.linalg.qr(q)
    return _unit(d @ q)


def _ball_radii(n: int, r_in: float, r_out: float, gen: torch.Generator) -> torch.Tensor:
    """`n` radii uniform IN VOLUME between `r_in` and `r_out` (the 3-ball's r^2 dr weight)."""
    u = torch.rand(n, generator=gen)
    return (r_in ** 3 + u * (r_out ** 3 - r_in ** 3)) ** (1.0 / 3.0)


def _in_disc(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """`n` points uniform in the UNIT disc, as (a, b). sqrt(u) because the area element is r dr."""
    rho = torch.rand(n, generator=gen).sqrt()
    th = torch.rand(n, generator=gen) * (2.0 * math.pi)
    return rho * torch.cos(th), rho * torch.sin(th)


def _in_ball(n: int, gen: torch.Generator) -> torch.Tensor:
    """`n` points uniform in the UNIT ball [n, 3]."""
    d = _unit(torch.randn(n, 3, generator=gen))
    return d * torch.rand(n, 1, generator=gen).pow(1.0 / 3.0)


# --------------------------------------------------------------------------- the seed
@register_operator("seed_cell_atlas", family="seed", set="compartment", kind="seed")
class SeedCellAtlas(Seed):
    """Lay a cell atlas out once, at x_0: where every compartment PIECE sits, how it is oriented,
    where its material points go, and what each of those points is worth in volume.

    compartment + mpm_particle -> compartment + mpm_particle. Reads the cell's position through
    the containment map `compartment -> cell`, and the pieces' identities from the compartment
    set's `node_type`; writes `pos` on both sets and `p_vol` / `mass` on the particles.

    The geometry of each compartment is a row of `ATLAS` above, overridable per compartment from
    the spec's `geometry:` block. A compartment named in `types:` with no row in either is a
    hard error rather than a default sphere: a silently spherical Golgi is a picture of a cell
    that has no Golgi in it, and nothing downstream would say so.

    WHY IT ALSO WRITES `p_vol` AND `mass`, which the entity's own provision already set. The
    provision has one number to work with -- the child set's `radius` -- so it gives every
    material point the same volume, hence the same mass. That is right for a set of identical
    bodies and wrong for an atlas: a plasma-membrane patch and a Golgi cisterna both get 1,000
    points, but the patch encloses about 5x the volume, so equal masses would make the Golgi
    5x as dense as the membrane and the cell would sort itself by compartment under gravity.
    Here each piece's volume follows from the shape it was actually built as,

        p_vol = V_piece / points_per_piece,        mass = p_vol * density,

    with V_piece the closed-form volume of the patch / sheet / rod / ball / filament. Written in
    place (`copy_`), never reassigned, so a CUDA-graph capture keeps pointing at the same storage.

    ONE CELL. The piece counts come from `types.count` on the compartment set, which
    `_assign_types` spreads over the WHOLE compartment level; with several parent cells the
    per-cell counts would be a multinomial draw rather than the atlas, so a parent set with more
    than one element is refused here instead of quietly producing nine cells with different
    organelle inventories.

    Reference: none -- a geometric initial condition, not a mechanism. Plexus (this work).
    The piece counts are read from the Interactive Cytology "Cell Atlas" viewer
    (`figures/cell_atlas.png`), which reports 4,177 modelled pieces over 15 compartments.
    """
    EMIT = None                       # writes state at x_0; there is no integrable delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["particles", "cell_radius"]
    MECHANISM_TAGS = ["cell_atlas", "compartment_seeding", "hierarchy", "ultrastructure"]
    PARAM_ROLES = {"particles": "material_point_set", "cell_radius": "cell_radius_world",
                   "geometry": "per_compartment_geometry", "centre": "cell_centre_override",
                   "seed": "rng_seed"}
    REFERENCE = "Plexus (this work); piece counts from the Cell Atlas viewer (figures/cell_atlas.png)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "compartment")
        self.particles = str(params["particles"])          # the child set filling each piece
        self.cell_radius = float(params["cell_radius"])    # world units; every atlas length scales by it
        self.centre = params.get("centre", None)           # override the parent cell's own position
        self.geometry = dict(params.get("geometry", {}) or {})
        # ONE LAUNCH VELOCITY PER CELL, and it has to be written here rather than declared on a
        # set. `vel_init` on a CONTAINED set draws one vector per PARENT and shares it among that
        # parent's children -- which for `mpm_particle` means one vector per COMPARTMENT PIECE, so
        # a 1,104-piece cell would leave frame 0 as 1,104 fragments flying apart. The velocity is a
        # property of the CELL, two levels up, and the only thing that knows the cell a material
        # point belongs to is the containment chain this operator already walks.
        self.vel_init = float(params.get("vel_init", 0.0))     # max speed, world units per time
        self.vel_axes = params.get("vel_init_axes", None)      # e.g. [0, 2]: launch in the plane
        self.seed = int(params.get("seed", 0))

    # -- geometry of one compartment, merged from the table and the spec ------------------- #
    def _row(self, name: str) -> dict:
        row = dict(ATLAS.get(name, {}))
        row.update(self.geometry.get(name, {}) or {})
        if not row:
            raise ValueError(
                f"seed_cell_atlas: compartment {name!r} has no geometry. It is not in the "
                f"built-in atlas ({', '.join(sorted(ATLAS))}) and the operator line declares no "
                f"`geometry: {{{name}: ...}}` for it. Give it one -- a compartment with no shape "
                f"would be seeded as an undifferentiated ball, which is a picture of a cell that "
                f"does not contain it.")
        for key, allowed in (("place", _PLACES), ("shape", _SHAPES), ("axis", _AXES)):
            v = row.get(key, "ball" if key != "axis" else "random")
            if v not in allowed:
                raise ValueError(f"seed_cell_atlas: compartment {name!r} has {key}={v!r}, "
                                 f"not one of {allowed}")
            row[key] = v
        return row

    @staticmethod
    def _piece_volume(row: dict, R: float) -> float:
        """The closed-form volume of ONE piece, in world units cubed (lengths are fractions of R).

        `volume_frac` scales the result, and it exists for the one case the closed forms cannot
        express: several bodies OCCUPYING THE SAME REGION, each holding a share of it. Five protein
        species crowding the cytoplasm are five balls of the cell's own radius, so taking each at
        its full geometric volume would give the cell five times its own mass. `volume_frac: 0.12`
        says this species fills 12% of the ball it is drawn in; the five together are a 60%
        crowded cytosol, which is roughly what a real one is. It changes only `p_vol` and hence
        the mass -- the points are still drawn throughout the region, because that is where the
        molecules are.
        """
        s = float(row.get("size", 0.05)) * R
        t = float(row.get("thickness", 0.02)) * R
        shape = row["shape"]
        vf = float(row.get("volume_frac", 1.0))
        if vf != 1.0:
            return vf * SeedCellAtlas._piece_volume({**row, "volume_frac": 1.0}, R)
        if shape in ("patch", "sheet"):
            return math.pi * s * s * t                     # a disc of radius s, t deep
        if shape == "rod":
            return math.pi * t * t * (2.0 * s)             # a cylinder of half-length s, radius t
        if shape == "ball":
            return (4.0 / 3.0) * math.pi * s ** 3
        if shape == "filament":
            return math.pi * t * t * s                     # contour length s, radius t
        if shape == "cisterna":
            # a disc minus its fenestrations. `holes` discs of radius `hole_r` x size, and the
            # fraction they remove is capped below 1 because a sheet that is all hole has no volume
            # and would put p_vol at zero -- which is a division by zero in the density, not a thin
            # membrane.
            _h = min(0.8, int(row.get("holes", 0)) * float(row.get("hole_r", 0.0)) ** 2)
            return math.pi * s * s * t * (1.0 - _h)
        if shape == "tubule":
            return math.pi * t * t * s                     # total contour length s, radius t
        if shape == "stack":
            # `layers` discs tapering geometrically from radius s to `taper` x s
            n = int(row.get("layers", 5)); tap = float(row.get("taper", 1.0))
            fr = [1.0 - (1.0 - tap) * (i / max(n - 1, 1)) for i in range(n)]
            return math.pi * t * s * s * sum(f * f for f in fr)
        if shape == "mito":
            # the outer capsule; the cristae are INSIDE it, so they add no volume -- they are a
            # rearrangement of where the points sit, not extra material.
            return math.pi * t * t * (2.0 * s) + (4.0 / 3.0) * math.pi * t ** 3
        if shape == "barrel":
            # `blades` rods of radius t and length `length` x s, on a circle of radius s
            return (int(row.get("blades", 9)) * math.pi * t * t
                    * float(row.get("length", 2.0)) * s)
        raise ValueError(f"unhandled shape {shape!r}")

    # -- where the pieces go ---------------------------------------------------------------- #
    def _place(self, row: dict, n: int, R: float, gen) -> tuple[torch.Tensor, torch.Tensor]:
        """(offset from the cell centre [n,3], outward unit direction [n,3]) for `n` pieces."""
        if row["place"] == "shell":
            d = _fibonacci(n, gen)
            r = torch.full((n, 1), float(row.get("r", 1.0)) * R)
            # a small radial jitter, a twentieth of the shell radius, so the spiral does not read
            # as a lattice -- a perfectly regular membrane looks manufactured, and a regular
            # lattice of MPM bodies also rings at one frequency when it is struck.
            r = r + (torch.rand(n, 1, generator=gen) - 0.5) * 0.05 * float(row.get("r", 1.0)) * R
            return d * r, d
        d = _unit(torch.randn(n, 3, generator=gen))
        r = _ball_radii(n, float(row.get("r_in", 0.0)), float(row.get("r_out", 0.9)), gen) * R
        return d * r[:, None], d

    # -- where the points inside one piece go ------------------------------------------------ #
    def _fill(self, row: dict, n_pieces: int, per: int, R: float,
              centroid: torch.Tensor, e1, e2, e3, gen) -> torch.Tensor:
        """World positions [n_pieces * per, 3] of the material points of these pieces.

        Row-major: piece p owns rows [p*per, (p+1)*per), which is the layout `build` gives a
        contained set (`arange(n_parent).repeat_interleave(per)`), so the result lines up with
        `mpm_particle.parent` without a scatter.
        """
        N = n_pieces * per
        s = float(row.get("size", 0.05)) * R
        t = float(row.get("thickness", 0.02)) * R
        shape = row["shape"]
        rep = lambda v: v.repeat_interleave(per, dim=0)                       # noqa: E731

        if shape == "ball":
            return rep(centroid) + _in_ball(N, gen) * s

        if shape == "rod":
            a, b = _in_disc(N, gen)
            z = (torch.rand(N, generator=gen) * 2.0 - 1.0) * s
            return rep(centroid) + rep(e1) * (a * t)[:, None] + rep(e2) * (b * t)[:, None] \
                + rep(e3) * z[:, None]

        if shape == "sheet":
            a, b = _in_disc(N, gen)
            z = (torch.rand(N, generator=gen) - 0.5) * t
            return rep(centroid) + rep(e1) * (a * s)[:, None] + rep(e2) * (b * s)[:, None] \
                + rep(e3) * z[:, None]

        if shape == "patch":
            # A CURVED CAP, NOT A FLAT DISC. The piece is a patch OF A SPHERE, so a point is
            # placed in the tangent plane and then pushed back out to the shell radius: the 421
            # patches then meet edge to edge on the sphere instead of each cutting a chord
            # through it, which at 0.105 of the radius is a gap of 0.0055 R between neighbours --
            # a hole a grid cell wide, i.e. a leaking membrane.
            a, b = _in_disc(N, gen)
            tang = rep(e1) * (a * s)[:, None] + rep(e2) * (b * s)[:, None]
            depth = (torch.rand(N, generator=gen) - 0.5) * t
            rad = centroid.norm(dim=1, keepdim=True)                # the shell radius of each piece
            dirn = _unit(rep(centroid) + tang)
            return dirn * (rep(rad) + depth[:, None])

        if shape == "filament":
            # A PERSISTENT RANDOM WALK per piece, then points sprinkled along it. `persistence`
            # p mixes each new segment direction as p * previous + (1 - p) * random: p = 1 is a
            # straight rod (a microtubule), p = 0 a freely jointed coil (chromatin).
            n_seg = int(row.get("segments", 12))
            p = float(row.get("persistence", 0.5))
            step = s / n_seg
            d = e3.clone()                                     # the first segment: the piece axis
            nodes = [torch.zeros(n_pieces, 3)]
            for _ in range(n_seg):
                nd = _unit(p * d + (1.0 - p) * _unit(torch.randn(n_pieces, 3, generator=gen)))
                nodes.append(nodes[-1] + nd * step)
                d = nd
            nodes = torch.stack(nodes, dim=1)                  # [n_pieces, n_seg+1, 3]
            # CENTRED ON THE PIECE, NOT STARTED AT IT. The walk above begins at the origin of the
            # piece's frame and only ever moves away from it, so an uncentred filament is a piece
            # whose declared centre is one of its ENDS: a 0.70 R cytoskeletal filament placed at
            # 0.90 R reached 1.60 R and the cell had 116 spines sticking out of it (measured: the
            # particle bounding box ran to 0.873 in a cell whose surface is at 0.80). Subtracting
            # the polyline's own centroid makes `place` mean the same thing for a filament as it
            # does for a rod or a ball.
            nodes = nodes - nodes.mean(dim=1, keepdim=True)
            u = torch.rand(N, generator=gen) * n_seg
            i = u.floor().long().clamp(max=n_seg - 1)
            f = (u - i)[:, None]
            pid = torch.arange(n_pieces).repeat_interleave(per)
            a0 = nodes[pid, i]; a1 = nodes[pid, i + 1]
            return rep(centroid) + a0 + f * (a1 - a0) + _in_ball(N, gen) * t

        if shape == "cisterna":
            # A BOWED, FENESTRATED SHEET. The bow is a paraboloid in the sheet's own plane --
            # z += bow * s * (1 - rho^2) -- so the cisterna is a shallow dish rather than a plate,
            # which is what a stack of them wrapped round a nucleus has to be to nest.
            #
            # THE HOLES ARE REJECTION-SAMPLED, and rejection rather than a mask because a masked
            # point has to go somewhere: setting it to the centre piles material at r = 0, and
            # dropping it leaves the piece with fewer points than `p_vol` was divided by. Re-drawing
            # keeps the count exact and the density flat.
            bow = float(row.get("bow", 0.0))
            nh, hr = int(row.get("holes", 0)), float(row.get("hole_r", 0.0))
            hc = torch.zeros(n_pieces, max(nh, 1), 2)
            if nh:
                # hole centres inside the disc, kept off the rim so a fenestration is a hole and
                # not a bite out of the edge
                _hr = torch.rand(n_pieces, nh, generator=gen).sqrt() * (1.0 - hr)
                _ht = torch.rand(n_pieces, nh, generator=gen) * (2.0 * math.pi)
                hc = torch.stack([_hr * torch.cos(_ht), _hr * torch.sin(_ht)], dim=2)
            pid = torch.arange(n_pieces).repeat_interleave(per)
            a = torch.zeros(N); b = torch.zeros(N)
            todo = torch.arange(N)
            for _ in range(32):
                _a, _b = _in_disc(int(todo.numel()), gen)
                ok = torch.ones(int(todo.numel()), dtype=torch.bool)
                if nh:
                    d2 = ((_a[:, None] - hc[pid[todo], :, 0]) ** 2
                          + (_b[:, None] - hc[pid[todo], :, 1]) ** 2)
                    ok = (d2 > hr * hr).all(dim=1)
                a[todo[ok]] = _a[ok]; b[todo[ok]] = _b[ok]
                todo = todo[~ok]
                if todo.numel() == 0:
                    break
            rho2 = a * a + b * b
            z = (torch.rand(N, generator=gen) - 0.5) * t + bow * s * (1.0 - rho2)
            return rep(centroid) + rep(e1) * (a * s)[:, None] + rep(e2) * (b * s)[:, None] \
                + rep(e3) * z[:, None]

        if shape == "tubule":
            # A BRANCHING TUBE NETWORK. A tree is grown per piece: node 0 at the centre, and each
            # new segment departs from a RANDOMLY CHOSEN existing node with probability `branch`
            # and from the previous tip otherwise. Points then pick a segment uniformly and sit
            # inside a tube of radius `thickness` about it -- so the point density follows contour
            # length, which is what `p_vol = pi t^2 s / per` assumes.
            n_seg = int(row.get("segments", 12))
            p = float(row.get("persistence", 0.5))
            br = float(row.get("branch", 0.0))
            step = s / n_seg
            nodes = [torch.zeros(n_pieces, 3)]
            dirs = [e3.clone()]
            segs = []                                       # (from_node, to_node)
            tip = torch.zeros(n_pieces, dtype=torch.long)
            for k in range(n_seg):
                # which node this segment leaves from: the tip, or an earlier node (a branch)
                if br > 0 and k > 1 and float(torch.rand(1, generator=gen)) < br:
                    frm = torch.randint(0, len(nodes), (n_pieces,), generator=gen)
                else:
                    frm = tip
                d0 = torch.stack(dirs, dim=1)[torch.arange(n_pieces), frm.clamp(max=len(dirs) - 1)]
                nd = _unit(p * d0 + (1.0 - p) * _unit(torch.randn(n_pieces, 3, generator=gen)))
                base = torch.stack(nodes, dim=1)[torch.arange(n_pieces), frm]
                nodes.append(base + nd * step)
                dirs.append(nd)
                segs.append((frm, torch.full((n_pieces,), len(nodes) - 1, dtype=torch.long)))
                tip = torch.full((n_pieces,), len(nodes) - 1, dtype=torch.long)
            NODES = torch.stack(nodes, dim=1)               # [n_pieces, n_seg+1, 3]
            NODES = NODES - NODES.mean(dim=1, keepdim=True)  # centred on the piece, as `filament`
            A = torch.stack([f for f, _ in segs], dim=1)     # [n_pieces, n_seg]
            B = torch.stack([t_ for _, t_ in segs], dim=1)
            pid = torch.arange(n_pieces).repeat_interleave(per)
            si = torch.randint(0, n_seg, (N,), generator=gen)
            f = torch.rand(N, 1, generator=gen)
            a0 = NODES[pid, A[pid, si]]; a1 = NODES[pid, B[pid, si]]
            return rep(centroid) + a0 + f * (a1 - a0) + _in_ball(N, gen) * t

        if shape == "stack":
            # `layers` BOWED CISTERNAE along e3, tapering cis -> trans. Each point picks a layer
            # uniformly, so every cisterna gets the same COUNT while the outer ones have more
            # volume -- which makes the trans face slightly denser, and is the price of keeping
            # this one closed-form draw instead of a per-layer loop.
            nl = int(row.get("layers", 5))
            gap = float(row.get("gap", 0.12)) * s
            bow = float(row.get("bow", 0.0))
            tap = float(row.get("taper", 1.0))
            li = torch.randint(0, nl, (N,), generator=gen)
            frac = 1.0 - (1.0 - tap) * (li.float() / max(nl - 1, 1))
            a, b = _in_disc(N, gen)
            rho2 = a * a + b * b
            z = (li.float() - (nl - 1) / 2.0) * gap + (torch.rand(N, generator=gen) - 0.5) * t \
                + bow * s * (1.0 - rho2)
            return rep(centroid) + rep(e1) * (a * s * frac)[:, None] \
                + rep(e2) * (b * s * frac)[:, None] + rep(e3) * z[:, None]

        if shape == "mito":
            # A BENT CAPSULE WITH CRISTAE. The centre line is an arc of angle `bend` * pi in the
            # (e3, e1) plane, so `bend: 0` is the straight rod this used to be and `bend: 1` is a
            # half-circle. `crista_frac` of the points are placed on transverse LAMELLAE -- discs
            # normal to the centre line at `cristae` evenly spaced stations, filling 0.8 of the
            # local radius -- and the rest fill the outer membrane shell. That split is why a
            # sectioned mitochondrion here shows internal structure instead of a solid sausage.
            bend = float(row.get("bend", 0.0))
            ncr = int(row.get("cristae", 0))
            cfr = float(row.get("crista_frac", 0.0)) if ncr else 0.0
            u = torch.rand(N, generator=gen) * 2.0 - 1.0            # -1..1 along the arc
            is_cr = torch.rand(N, generator=gen) < cfr
            if ncr:
                # snap the crista points onto their station, so the lamellae are discrete sheets
                st = torch.randint(0, ncr, (N,), generator=gen).float()
                u = torch.where(is_cr, (st / max(ncr - 1, 1)) * 1.8 - 0.9, u)
            ang = u * bend * math.pi * 0.5
            # arc of radius rc such that the arc length is 2s; bend -> 0 recovers the straight rod
            rc = (s / max(bend * math.pi * 0.5, 1e-6)) if bend > 1e-6 else 0.0
            if bend > 1e-6:
                along = rc * torch.sin(ang)
                lateral = rc * (1.0 - torch.cos(ang))
            else:
                along = u * s
                lateral = torch.zeros_like(u)
            # cross-section: an annulus for the outer membrane, a filled disc for a crista
            ca, cb = _in_disc(N, gen)
            crho = torch.sqrt(ca * ca + cb * cb).clamp_min(1e-9)
            shell = 0.72 + 0.28 * torch.rand(N, generator=gen)      # outer membrane band
            scale = torch.where(is_cr, torch.rand(N, generator=gen).sqrt() * 0.80, shell)
            ca, cb = ca / crho * scale, cb / crho * scale
            # e1 carries the bend, so the cross-section rides e1 (offset by `lateral`) and e2
            return rep(centroid) + rep(e3) * along[:, None] \
                + rep(e1) * ((lateral + ca * t))[:, None] + rep(e2) * (cb * t)[:, None]

        if shape == "barrel":
            # NINE TRIPLET MICROTUBULES ON A CIRCLE -- the centriole's defining nine-fold symmetry,
            # which no capsule or ball can express. Each blade is a rod of radius `thickness`
            # parallel to e3, tilted tangentially by `skew` so the barrel has the pinwheel the EM
            # cross-sections show.
            nb = int(row.get("blades", 9))
            L = float(row.get("length", 2.0)) * s
            skew = float(row.get("skew", 0.0))
            bi = torch.randint(0, nb, (N,), generator=gen).float()
            th = bi / nb * 2.0 * math.pi
            z = (torch.rand(N, generator=gen) - 0.5) * L
            th = th + skew * (z / max(L, 1e-9))                     # the pinwheel tilt
            ca, cb = _in_disc(N, gen)
            x = torch.cos(th) * s + ca * t
            y = torch.sin(th) * s + cb * t
            return rep(centroid) + rep(e1) * x[:, None] + rep(e2) * y[:, None] + rep(e3) * z[:, None]

        raise ValueError(f"unhandled shape {shape!r}")

    # -- the seed itself ---------------------------------------------------------------------- #
    def forward(self, H, mask=None):
        lvl = H.level(self.at)                                 # the compartment pieces
        plvl = H.level(self.particles)                         # the material points
        dev = lvl.state.device
        R = self.cell_radius
        gen = torch.Generator(device="cpu").manual_seed(self.seed)

        if getattr(plvl, "parent_name", None) != lvl.name:
            raise ValueError(
                f"seed_cell_atlas: `particles: {self.particles}` has parent "
                f"{getattr(plvl, 'parent_name', None)!r}, not {lvl.name!r}. The points that fill "
                f"a piece must be CONTAINED in it -- that containment map is what the seed writes "
                f"through and what the renderer colours by.")
        names = list(getattr(lvl, "type_names", []) or [])
        nt = getattr(lvl, "node_type", None)
        if not names or nt is None:
            raise ValueError(
                f"seed_cell_atlas: set {lvl.name!r} declares no `types:`. The atlas IS the type "
                f"table -- one type per compartment, its `count:` the number of pieces -- so a "
                f"compartment set without one has nothing to lay out.")
        # ONE ATLAS PER CELL. `centres` is [n_cells, 3] and `owner` says which cell each PIECE
        # belongs to, so a scene of 100 cells is 100 independent layouts sharing one set -- which
        # is the point of the containment map, and needs no per-cell set.
        #
        # THIS USED TO REFUSE ANY CELL COUNT BUT ONE, and the refusal was correct at the time:
        # `types.count` was assigned over the whole compartment level with a single permutation, so
        # 25 cells received a MULTINOMIAL DRAW of each organelle rather than the atlas. That is
        # fixed where it belonged, in `_assign_types` -- a count on a CONTAINED set is per parent,
        # tiled and shuffled inside each parent's block -- so the inventory is now identical from
        # cell to cell and only the placement's randomness differs.
        pname = getattr(lvl, "parent_name", None)
        if self.centre is not None:
            centres = torch.tensor([[float(v) for v in self.centre]])
            owner = torch.zeros(lvl.n, dtype=torch.long)
        elif pname is not None:
            par = H.level(pname)
            centres = par.get("pos")[:, :3].detach().cpu()
            owner = lvl.parent.detach().cpu()
        else:
            raise ValueError("seed_cell_atlas: set has no `parent:` and the operator line gives "
                             "no `centre:` -- there is nothing to place the atlas around.")
        n_cells = int(centres.shape[0])

        # HOW MANY POINTS EACH PIECE OWNS, AND WHERE ITS ROWS ARE -- read off the containment map
        # rather than assumed uniform. `per_parent` may be a mapping from compartment type to
        # count, because 50 points is enough for a plasma-membrane patch and a nuclear-envelope
        # patch wants 50,000; the blocks are then unequal, so the piece's rows are
        # `[offset[p], offset[p] + counts[p])` and not `[p*per, (p+1)*per)`.
        pcpu = plvl.parent.detach().cpu()
        counts = torch.bincount(pcpu, minlength=lvl.n)
        if int(counts.sum()) != plvl.n or int((counts == 0).sum()):
            raise ValueError(
                f"seed_cell_atlas: the containment map of {plvl.name!r} does not cover "
                f"{lvl.name!r} -- {int((counts == 0).sum())} piece(s) own no material points. "
                f"A compartment with no substance is not a compartment.")
        offset = torch.cat([torch.zeros(1, dtype=counts.dtype), counts.cumsum(0)[:-1]])

        cpos = torch.zeros(lvl.n, 3)
        ppos = torch.zeros(plvl.n, 3)
        pvol = torch.zeros(plvl.n)
        ntc = nt.detach().cpu()
        report = []
        tally = {}
        for cid in range(n_cells):
            centre = centres[cid]
            in_cell = (owner == cid)
            for tid, name in enumerate(names):
                idx = torch.nonzero(in_cell & (ntc == tid), as_tuple=False).flatten()
                k = int(idx.numel())
                if k == 0:
                    continue
                row = self._row(name)
                # EVERY PIECE OF ONE COMPARTMENT HAS THE SAME POINT COUNT, because `per_parent` keys on
                # the parent's TYPE and a compartment is exactly one type. That is what lets the fill
                # below stay a single vectorised call per compartment instead of a loop over pieces.
                per = int(counts[idx[0]])
                if not bool((counts[idx] == per).all()):
                    raise ValueError(
                        f"seed_cell_atlas: pieces of compartment {name!r} own different numbers of "
                        f"material points ({int(counts[idx].min())}..{int(counts[idx].max())}). "
                        f"`per_parent` varies by parent TYPE, and every piece here is one type.")
                off, dirn = self._place(row, k, R, gen)
                e3 = dirn if row["axis"] == "radial" else _unit(torch.randn(k, 3, generator=gen))
                e1, e2 = _ortho(e3)
                cpos[idx] = centre + off
                # the piece's points, in the piece's own frame about its centre-offset, then shifted
                # to world by the cell centre. `_fill` works in cell-centred coordinates because the
                # `patch` shape needs the shell radius, which is only meaningful about that centre.
                pts = self._fill(row, k, per, R, off, e1, e2, e3, gen) + centre
                rows = (offset[idx][:, None] + torch.arange(per)).flatten()
                ppos[rows] = pts
                v = self._piece_volume(row, R)
                pvol[rows] = v / per
                # THE RADIAL EXTENT, MEASURED, not derived from the table. `place` and `shape`
                # are two independent statements and a compartment sits where their SUM puts it,
                # which is how 116 cytoskeletal filaments came to stick 0.6 R out of a cell whose
                # every parameter looked reasonable in isolation. It makes "the mitochondria are
                # in the cytoplasm" a checkable claim.
                #
                # REPORTED FOR THE FIRST CELL ONLY. The inventory is identical from cell to cell
                # -- the counts are per parent -- so 100 copies of the same ten lines is a
                # thousand lines of log saying one thing.
                rad = (pts - centre).norm(dim=1) / R
                tally[name] = tally.get(name, 0) + k * per
                if cid == 0:
                    report.append(
                        f"{name:16s} x{k:<4d} {row['shape']:9s} {per:>6,}/piece "
                        f"({k * per:>8,} pts/cell)  r in [{rad.min():.2f}, {rad.max():.2f}] R  "
                        f"V={v:.3e}  spacing={(v / per) ** (1 / 3):.4f}")

        px0, px1 = lvl.state_schema["pos"]
        lvl.state[:, px0:px1] = cpos.to(dev)
        if "vel" in lvl.state_schema:
            vx0, vx1 = lvl.state_schema["vel"]
            lvl.state[:, vx0:vx1] = 0.0
        qx0, qx1 = plvl.state_schema["pos"]
        plvl.state[:, qx0:qx1] = ppos.to(dev)
        if "vel" in plvl.state_schema:
            vx0, vx1 = plvl.state_schema["vel"]
            if self.vel_init > 0:
                # ISOTROPIC IN DIRECTION, UNIFORM IN SPEED, one draw per cell, then broadcast down
                # BOTH containment maps -- piece by `owner`, point by the piece it belongs to. A
                # cell translates; it does not come apart.
                dirs = _unit(torch.randn(n_cells, 3, generator=gen))
                spd = torch.rand(n_cells, 1, generator=gen) * self.vel_init
                vcell = dirs * spd
                if self.vel_axes is not None:
                    keep = torch.zeros(3)
                    for ax in self.vel_axes:
                        keep[int(ax)] = 1.0
                    vcell = vcell * keep
                per_piece = vcell[owner]                       # [n_pieces, 3]
                plvl.state[:, vx0:vx1] = per_piece[pcpu].to(dev)
                if "vel" in lvl.state_schema:
                    lx0, lx1 = lvl.state_schema["vel"]
                    lvl.state[:, lx0:lx1] = per_piece.to(dev)
                print(f"[seed_cell_atlas] launch: |v| <= {self.vel_init:g} per cell, isotropic"
                      + (f", axes {self.vel_axes}" if self.vel_axes else "")
                      + f"; mean speed {float(spd.mean()):.4g}", flush=True)
            else:
                plvl.state[:, vx0:vx1] = 0.0

        # in place, so a captured CUDA graph keeps pointing at the same storage
        if hasattr(plvl, "p_vol"):
            plvl.p_vol.copy_(pvol.to(dev))
            if hasattr(plvl, "density"):
                plvl.mass.copy_(plvl.p_vol * plvl.density)
            else:
                plvl.mass.copy_(plvl.p_vol)
        print(f"[seed_cell_atlas] {n_cells} cell(s), {lvl.n:,} pieces / {plvl.n:,} material "
              f"points ({int(counts.min()):,}-{int(counts.max()):,} per piece), cell radius {R}; "
              f"{plvl.n // max(n_cells, 1):,} points per cell", flush=True)
        for line in report:
            print(f"[seed_cell_atlas]   {line}", flush=True)
        return {}


# --------------------------------------------------------------------------- the aggregate
@register_operator("aggregate_centroid", family="hierarchy", set="compartment", kind="aggregate")
class AggregateCentroid(Aggregate):
    """A parent's position as the mass-weighted mean of its children's: sum_pi, and nothing else.

    children -> parent along the containment map `child.parent`. Reads the children's `pos` (and
    `mass`, when the child set carries one); writes the parent's `pos`.

        x_p = ( sum_{c in pi^-1(p)} m_c x_c ) / ( sum_{c in pi^-1(p)} m_c )

    with m_c the child's mass, or 1 for a set with no mass -- so an unweighted centroid is the
    same operator on a set whose elements weigh the same, not a second one.

    IT CROSSES ANY NUMBER OF LEVELS, because pi is a function and a composition of functions is
    a function: `H.lift_index(child, at)` composes the containment maps between them into one
    map with the same fibre structure, so `index_add_` along it counts every descendant exactly
    once whether it is a child, a grandchild or a great-great-grandchild. `child:` may therefore
    name any set below `at:`, and the operator is depth-generic without a recursive special case.

    Both spellings are available and they are not equivalent. In the cell atlas,

        aggregate_centroid at: compartment   child: mpm_particle   points -> piece
        aggregate_centroid at: cell          child: compartment    pieces -> cell

    walks the chain one hop at a time and leaves a meaningful position on EVERY level, whereas

        aggregate_centroid at: cell          child: mpm_particle   points -> cell   (one hop)

    reaches the top directly and leaves the compartments frozen. Use the first when the
    intermediate level is itself observed, the second when it is only structure. Without either,
    the piece and cell positions stay at their seeded values while the material points deform
    away from them, so `cell.pos` reports where the cell STARTED for the whole run.

    A mass-weighted mean also composes correctly across hops -- the intermediate level's weight
    is the aggregated child mass -- only if the intermediate carries that mass. A `compartment`
    does not, so the two-hop form weights pieces equally while the one-hop form weights points
    equally; they differ, and which is wanted is a modelling choice, not a numerical one.

    `MAY_MUTATE_INTEGRATED_STATE` because it writes `pos`, which is an integrated block on a
    spatial set. That is safe exactly when nothing integrates the parent -- the case here, where
    the only operator touching a compartment is `gravity`, whose `EMIT` is `mpm_acceleration`
    and is therefore routed to the MPM substep rather than to the engine's integrator. A parent
    that IS integrated would have this operator fighting its integrator every tick.

    Reference: none -- the Aggregate family of plexus2 sec. "The operator algebra". Plexus (this work).
    """
    KIND = "aggregate"
    EMIT = None                                   # writes the parent's state; returns no delta
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["child"]
    INPUTS = ["child"]; OUTPUTS = ["parent"]; READS = ["pos"]; WRITES = ["pos"]
    MAPS = ["parent"]
    MECHANISM_TAGS = ["aggregate", "centroid", "cross_scale"]
    PARAM_ROLES = {"child": "descendant_set", "weight": "mass_weighted_or_uniform"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        # `child:` AND NOT `from:`. `from:` is a reserved spec key that names a source FIELD, and
        # the schema rejects it when it names a set -- which is the right refusal: an Exchange
        # reads a field, an Aggregate reads a contained SET, and spelling both `from` would make
        # the two indistinguishable in a spec.
        self.child = params.get("child")
        self.weighted = bool(params.get("weight", True))

    def forward(self, H, mask=None):
        parent = H.level(self.at)
        child = H.level(self.child)
        if parent.name not in H.ancestors(child.name):
            raise ValueError(
                f"aggregate_centroid: {self.child!r} is not contained in {parent.name!r} at any "
                f"depth (its containment chain is "
                f"{' -> '.join(H.ancestors(child.name)) or '(none)'}). Aggregate travels the "
                f"containment map; without one there is no partition to sum over.")
        pidx = H.lift_index(child.name, parent.name)   # composed pi, however many hops away
        x = child.get("pos")
        w = child.occ if getattr(child, "occ", None) is not None else torch.ones_like(x[:, 0])
        if self.weighted and hasattr(child, "mass"):
            w = w * child.mass
        num = torch.zeros(parent.n, x.shape[1], device=x.device, dtype=x.dtype)
        den = torch.zeros(parent.n, device=x.device, dtype=x.dtype)
        num.index_add_(0, pidx, x * w[:, None])
        den.index_add_(0, pidx, w)
        centroid = num / den.clamp_min(1e-12)[:, None]
        px0, px1 = parent.state_schema["pos"]
        if torch.is_grad_enabled():
            # THE DIFFERENTIABLE PATH: clone, so the tape keeps the old state alive.
            st = parent.state.clone()
            st[:, px0:px1] = torch.where(den[:, None] > 0, centroid, st[:, px0:px1])
            parent.state = st
        else:
            # THE FORWARD PATH WRITES IN PLACE, AND MUST. Reassigning `parent.state` allocates a
            # new tensor, and the engine's graph signature is the set of buffer ADDRESSES the
            # captured substep baked in: one reassignment anywhere in the tick invalidates the
            # capture and the run finishes eager. Measured on the 48,780-point smoke test, the
            # clone path printed "a state buffer was reallocated after the substep was captured;
            # dropping the CUDA graph" on the first tick -- for a write to `compartment`, a set
            # the substep never touches.
            old = parent.state[:, px0:px1]
            parent.state[:, px0:px1] = torch.where(den[:, None] > 0, centroid, old)
        return {}


# --------------------------------------------------------------------------- signalling
#
# TWO GENERIC OPERATORS ON A NAMED STATE BLOCK. Neither mentions a mitochondrion, and that is the
# point: what they need is a set with a scalar block and a relation over it, which is a shape many
# biological objects have. They exist because the alternative -- `cell_chem_diffuse` -- is welded
# to a two-species `chem` block (`N_SPECIES = 2`) and to the cell adjacency of a vertex mesh, so
# there was no way to diffuse ONE scalar over an ordinary `edge_index`.
@register_operator("seed_state_random", family="seed", set="compartment", kind="seed")
class SeedStateRandom(Seed):
    """Fill a named state block with U(lo, hi), once, at x_0.

    set -> set: writes `block`, reads nothing.

    An initial condition with no structure is still an initial condition, and it has to be an
    operator rather than a `sets:` key for the reason `seed:` exists at all: `build` seeds
    POSITIONS from declarative placement rules, and every other block starts at zero. A network
    of identical zeros has no dynamics to show, so the first thing any relaxation on a graph
    needs is something to relax from.

    Reference: none -- an initial condition, not a mechanism. Plexus (this work).
    """
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["block"]
    MECHANISM_TAGS = ["initial_condition", "random_state"]
    PARAM_ROLES = {"block": "state_block", "lo": "lower_bound", "hi": "upper_bound"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "compartment")
        self.block = str(params["block"])
        self.lo = float(params.get("lo", 0.0))
        self.hi = float(params.get("hi", 1.0))
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if self.block not in lvl.state_schema:
            raise ValueError(
                f"seed_state_random: {lvl.name!r} has no state block {self.block!r} "
                f"(has: {', '.join(b.name for b in lvl.state_schema.blocks)}). Declare it under "
                f"`sets.{lvl.name}.state:`.")
        b0, b1 = lvl.state_schema[self.block]
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        v = torch.rand(lvl.n, b1 - b0, generator=g) * (self.hi - self.lo) + self.lo
        lvl.state[:, b0:b1] = v.to(lvl.state.device)
        print(f"[seed_state_random] {lvl.name}.{self.block} ~ U({self.lo:g}, {self.hi:g}) "
              f"over {lvl.n:,} elements", flush=True)
        return {}


@register_operator("state_diffuse", family="signalling", set="compartment", kind="lateral")
class StateDiffuse(Lateral):
    """Diffusion of a scalar along a set's own relation -- the graph Laplacian, nothing more.

    set -[edge_index]-> set: reads `block` and the relation, emits d(block)/dt.

        dv_i/dt = D sum_{j ~ i} (v_j - v_i) / deg(i)      normalise: true (the default)
        dv_i/dt = D sum_{j ~ i} (v_j - v_i)               normalise: false

    v is the value of `block` at element i, the sum runs over its neighbours in `edge_index`, and
    D is a rate in inverse time -- the graph carries no length, so there is no length in the
    coefficient either. Dividing by the degree gives the NORMALISED Laplacian, whose eigenvalues
    lie in [-2, 0], so an explicit step is stable at any degree; without it an element that
    acquires many neighbours can overshoot in one tick, which on a relation rebuilt every frame by
    proximity is a thing that actually happens.

    THE RELATION IS NOT THIS OPERATOR'S BUSINESS. It reads whatever `edge_index` the set carries,
    so what the neighbours ARE is decided by the rewire scheduled before it: `radius_graph` makes
    them "everything within r", and a different rewire would make them something else without this
    operator changing. That separation is why the mechanism and the relation are two operators.

    Returns its delta under an explicit `(set, block)` key rather than relying on `INTEGRAND`,
    because the block is a PARAMETER here -- one class serving any scalar -- and `INTEGRAND` is
    read off the class by the integration-order check.

    Reference: Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86 (diffusion); Chung, F.
    (1997). Spectral Graph Theory (the normalised Laplacian).
    """
    EMIT = "velocity"
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["block", "D"]
    INPUTS = ["set"]; OUTPUTS = ["set"]; READS = ["block"]; WRITES = ["block"]
    MECHANISM_TAGS = ["diffusion", "graph_laplacian", "signalling"]
    PARAM_ROLES = {"block": "state_block", "D": "diffusion_rate",
                   "normalise": "degree_normalised"}
    REFERENCE = ("Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; "
                 "Chung, F. (1997). Spectral Graph Theory.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "compartment")
        self.block = str(params["block"])
        self.D = float(params["D"])
        self.norm = bool(params.get("normalise", True))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        ei = getattr(lvl, "edge_index", None)
        b0, b1 = lvl.state_schema[self.block]
        v = lvl.state[:, b0:b1]
        if ei is None or ei.numel() == 0:
            return {(lvl.name, self.block): torch.zeros_like(v)}
        i, j = ei[0], ei[1]                                  # row 0 receives, row 1 sends
        occ = lvl.occ if getattr(lvl, "occ", None) is not None else torch.ones_like(v[:, 0])
        live = (occ[i] > 0) & (occ[j] > 0)
        i, j = i[live], j[live]
        d = torch.zeros_like(v)
        d.index_add_(0, i, v[j] - v[i])
        if self.norm:
            deg = torch.zeros(lvl.n, device=v.device, dtype=v.dtype)
            deg.index_add_(0, i, torch.ones_like(i, dtype=v.dtype))
            d = d / deg.clamp_min(1.0)[:, None]
        return {(lvl.name, self.block): self.D * d}


# --------------------------------------------------------------------------- motility
#
# HOW AN ADHERENT CELL MOVES: by gripping the substrate and pulling, not by being pushed.
#
#     seed_polarity        seed     a unit direction per cell, in the substrate plane
#     substrate_traction   lateral  a tangential force on the material that TOUCHES the floor
#     protrusion           lateral  an active push at the leading edge, pull at the rear
#
# WHY NOT A BODY FORCE ON THE WHOLE CELL. The cheap way to move a cell is `gravity` with a
# sideways vector, and it is wrong in a way that matters: every material point feels it equally,
# so the cell translates as a blob and its speed has no relationship to its shape, its contact
# area, or how well it adheres. A cell that spreads twice as far moves at the same speed, which
# makes the adhesion mechanics decorative.
#
# Both operators below tie the force to WHERE the material is -- near the floor for traction,
# along the polarity axis for protrusion -- so speed emerges from the mechanics instead of being
# set. They compose: traction is the grip, protrusion is the step.

@register_operator("seed_polarity", family="motility", set="cell", kind="seed")
class SeedPolarity(Seed):
    """A unit direction per element, lying IN the substrate plane, written to `block`.

    cell -> cell: writes `polarity`, reads nothing.

    In the plane and not isotropic in 3D, because a polarity with a vertical component asks the
    cell to crawl into the floor or off it -- the traction below would then drive material through
    the substrate, which the wall condition resists rather than converts, and the cell would
    grind rather than move. The direction a crawling cell has is a direction ALONG its substrate.

    `spread` in [0, 1] interpolates between one shared heading (0, a marching population) and an
    independent draw per cell (1, a scattering one).

    Reference: none -- an initial condition. Plexus (this work).
    """
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["polarity", "initial_condition", "motility"]
    PARAM_ROLES = {"block": "state_block", "axis": "substrate_normal_axis",
                   "spread": "heading_dispersion", "seed": "rng_seed"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.block = str(params.get("block", "polarity"))
        self.axis = int(params.get("axis", 1))          # the substrate's normal (up)
        self.spread = float(params.get("spread", 1.0))
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if self.block not in lvl.state_schema:
            raise ValueError(
                f"seed_polarity: {lvl.name!r} has no state block {self.block!r}. Declare it under "
                f"`sets.{lvl.name}.state:` -- a direction is state the cell carries, not a "
                f"parameter of the operator that reads it.")
        b0, b1 = lvl.state_schema[self.block]
        D = b1 - b0
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        ax = [i for i in range(D) if i != self.axis]
        th0 = float(torch.rand(1, generator=g)) * 2 * math.pi
        th = th0 + (torch.rand(lvl.n, generator=g) - 0.5) * (2 * math.pi * self.spread)
        v = torch.zeros(lvl.n, D)
        v[:, ax[0]] = torch.cos(th)
        v[:, ax[1 % len(ax)]] = torch.sin(th)
        lvl.state[:, b0:b1] = v.to(lvl.state.device)
        print(f"[seed_polarity] {lvl.n} heading(s) in the plane normal to axis {self.axis}, "
              f"spread {self.spread:g}", flush=True)
        return {}


@register_operator("substrate_traction", family="motility", set="particle", kind="lateral")
class SubstrateTraction(Lateral):
    """A tangential acceleration on the material within a contact layer of the substrate.

    particle -[containment]-> particle: reads `pos` and its cell's `polarity`; emits an
    acceleration the MPM substep consumes as a body force.

        a_i = f * w(h_i) * p_{c(i)},        w(h) = 1 - clamp(h / d, 0, 1)

    with h_i the height of point i above the substrate, d the contact-layer thickness in world
    units, f the traction per unit mass (world units per time squared), p the unit polarity of the
    CELL that owns the point, and c(i) the composition of the containment maps from the point up
    to its cell -- two hops here, and `H.lift_index` is what makes that one gather.

    `w` IS A RAMP AND NOT A STEP. A hard cutoff makes the traction discontinuous in the one
    coordinate the cell is actively changing as it spreads, so the total force jumps whenever a
    point crosses the threshold and the cell judders at the frequency of its own surface
    roughness. The ramp makes the total traction a smooth function of how much material is near
    the floor -- which IS the quantity the model is claiming matters.

    EMITS `mpm_acceleration`, so it is routed to the substep as `a_ext` exactly like gravity and
    is never integrated on the set directly.

    Reference: the traction-based picture of adherent motility -- e.g. Barnhart, E. L. et al.
    (2011). An adhesion-dependent switch between mechanisms that determine motile cell shape.
    PLoS Biol. 9:e1001059. The implementation is not theirs.
    """
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["f", "cell_set"]
    INPUTS = ["particle", "cell"]; OUTPUTS = ["particle"]
    READS = ["pos", "polarity"]; WRITES = []
    MAPS = ["parent"]
    MECHANISM_TAGS = ["motility", "traction", "adhesion", "substrate"]
    PARAM_ROLES = {"f": "traction_per_unit_mass", "contact": "contact_layer_thickness",
                   "gate": "phase_gate",
                   "floor": "substrate_height", "axis": "substrate_normal_axis",
                   "cell_set": "polarity_owner", "block": "polarity_block"}
    REFERENCE = ("Barnhart, E. L., Lee, K.-C., Keren, K., Mogilner, A. & Theriot, J. A. (2011). "
                 "PLoS Biol. 9:e1001059.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.f = float(params["f"])
        self.cell_set = str(params["cell_set"])
        self.block = str(params.get("block", "polarity"))
        self.axis = int(params.get("axis", 1))
        self.floor = float(params.get("floor", 0.0))
        self.contact = float(params.get("contact", 0.01))
        self.gate = params.get("gate")

    def forward(self, H, mask=None):
        p = H.level(self.at)
        X = p.get("pos")
        h = X[:, self.axis] - self.floor
        w = (1.0 - (h / max(self.contact, 1e-9)).clamp(0.0, 1.0)).clamp(min=0.0)
        cl = H.level(self.cell_set)
        b0, b1 = cl.state_schema[self.block]
        pol = cl.state[:, b0:b1]
        idx = H.lift_index(p.name, self.cell_set)      # point -> ... -> cell, one composed gather
        a = pol[idx] * (self.f * w)[:, None]
        gt = _gate(self, H, idx)
        if gt is not None:
            a = a * gt[:, None]
        if mask is not None:
            a = a * mask.float()[:, None]
        return {p.name: a}


@register_operator("protrusion", family="motility", set="particle", kind="lateral")
class Protrusion(Lateral):
    """An active push at the leading edge and a pull at the rear -- the step, not the grip.

    particle -[containment]-> particle: reads `pos`, its cell's centroid and its cell's
    `polarity`; emits an acceleration the MPM substep consumes as a body force.

        s_i = (x_i - x_c) . p_c / R,                     where along the cell i sits, in [-1, 1]
        a_i = f * ( tanh(s_i / width) ) * p_c            forward at the front, back at the rear

    x_c is the centroid of the cell that owns point i (kept current by `aggregate_centroid`), p_c
    its unit polarity, R the cell radius in world units, f the drive per unit mass and `width` the
    softness of the front/rear split -- small is two sharp lobes, large is nearly uniform.

    THE FORCE SUMS TO ZERO OVER A SYMMETRIC CELL, which is the point and the difference from a
    body force. `tanh` is odd, so a cell whose material is distributed symmetrically about its
    centroid receives no net push: it EXTENDS forward and retracts behind instead of accelerating.
    Motion then comes from that extension being anchored by `substrate_traction` -- the cell
    reaches, grips, and pulls itself over the new contact. Without traction, protrusion alone
    stretches a cell that goes nowhere, which is the correct behaviour of a cell on a frictionless
    surface and a useful control.

    Reference: the protrusion-adhesion-contraction picture of crawling -- e.g. Mogilner, A. (2009).
    Mathematics of cell motility. J. Math. Biol. 58:105-134. The implementation is not his.
    """
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["f", "cell_set", "radius"]
    INPUTS = ["particle", "cell"]; OUTPUTS = ["particle"]
    READS = ["pos", "polarity"]; WRITES = []
    MAPS = ["parent"]
    MECHANISM_TAGS = ["motility", "protrusion", "active_stress"]
    PARAM_ROLES = {"f": "drive_per_unit_mass", "radius": "cell_radius_world",
                   "gate": "phase_gate",
                   "width": "front_rear_softness", "cell_set": "polarity_owner"}
    REFERENCE = "Mogilner, A. (2009). Mathematics of cell motility. J. Math. Biol. 58:105-134."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.f = float(params["f"])
        self.cell_set = str(params["cell_set"])
        self.block = str(params.get("block", "polarity"))
        self.radius = float(params["radius"])
        self.width = float(params.get("width", 0.45))
        self.gate = params.get("gate")

    def forward(self, H, mask=None):
        p = H.level(self.at)
        X = p.get("pos")
        cl = H.level(self.cell_set)
        b0, b1 = cl.state_schema[self.block]
        pol = cl.state[:, b0:b1]
        cen = cl.get("pos")
        idx = H.lift_index(p.name, self.cell_set)
        pol_i, cen_i = pol[idx], cen[idx][:, : X.shape[1]]
        s = ((X - cen_i) * pol_i).sum(1) / max(self.radius, 1e-9)
        a = pol_i * (self.f * torch.tanh(s / max(self.width, 1e-6)))[:, None]
        gt = _gate(self, H, idx)
        if gt is not None:
            a = a * gt[:, None]
        if mask is not None:
            a = a * mask.float()[:, None]
        return {p.name: a}


# --------------------------------------------------------------------------- the crawl cycle
#
# A CLOCK PER CELL, AND TWO OPERATORS READING IT AT DIFFERENT PHASES.
#
#     phase_clock   lateral   dphi/dt = omega, per cell -- the frequency source, as STATE
#     `gate:` on protrusion and substrate_traction reads that phase
#
# WHY NET MOTION NEEDS TWO PHASES AND NOT ONE MECHANISM. `protrusion` is odd about the cell's
# centroid, so it extends the front and retracts the rear and sums to zero: driven by a symmetric
# cycle with a CONSTANT grip, the cell reaches out and is pulled straight back, and returns exactly
# where it started. That is not a tuning failure, it is reciprocity -- the same argument as the
# scallop theorem, and no amount of amplitude fixes it. What breaks it is gripping at one part of
# the cycle and releasing at another: extend while anchored, retract while free. So the adhesion
# has to be modulated OUT OF PHASE with the protrusion, which is why the clock is a separate thing
# both of them read rather than a parameter inside either.
#
# WHY THE PHASE IS PER-CELL STATE AND NOT THE GLOBAL `pacemaker`. `field_ops.pacemaker` publishes
# one periodic scalar per tick under `H.signals`, which is the right object when a whole tissue
# beats together and the wrong one here: every cell would extend and grip in lockstep, which is an
# artefact a reader has to be told to ignore. As a `phase` block on the cell set, each cell carries
# its own angle, the spread of angles is an initial condition, `omega` can vary from cell to cell,
# and the phase is recorded -- so "are they synchronised?" becomes a question about the data
# instead of about the code.
@register_operator("phase_clock", family="motility", set="cell", kind="lateral")
class PhaseClock(Lateral):
    """Advance a per-element phase at a fixed rate: the frequency source, as state.

    cell -> cell: reads `block`, emits d(block)/dt.

        dphi_i/dt = omega_i,        omega_i = omega * (1 + jitter * u_i),  u_i ~ U(-1, 1)

    omega is in radians per unit time, so the period is 2*pi/omega in the run's own time units.
    `jitter` disperses the RATES, which is what makes a population drift apart rather than merely
    start apart: two cells seeded at different phases but identical omega stay exactly that far
    apart forever, which is a rotation of a synchronised population and not a desynchronised one.

    The phase is not wrapped. A angle that grows without bound is what the readers want -- `sin`
    and `cos` are periodic anyway -- and wrapping would put a discontinuity into an integrated
    block, which the engine would then integrate across.

    Reference: none -- a linear phase is a modelling choice, not a published law. Plexus (this work).
    """
    EMIT = "velocity"
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["omega"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["phase"]; WRITES = ["phase"]
    MECHANISM_TAGS = ["clock", "oscillator", "motility"]
    PARAM_ROLES = {"omega": "angular_frequency", "jitter": "rate_dispersion",
                   "block": "phase_block"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.block = str(params.get("block", "phase"))
        self.omega = float(params["omega"])
        self.jitter = float(params.get("jitter", 0.0))
        self.seed = int(params.get("seed", 0))
        self._w = None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        b0, b1 = lvl.state_schema[self.block]
        if self._w is None:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            u = (torch.rand(lvl.n, b1 - b0, generator=g) * 2.0 - 1.0)
            self._w = (self.omega * (1.0 + self.jitter * u)).to(lvl.state.device)
        return {(lvl.name, self.block): self._w}


def _gate(op, H, idx):
    """The cycle's amplitude for each point, from its cell's phase -- 1 + cos, in [0, 1].

    `gate: {block: phase, offset: <radians>, floor: <0..1>}` on any operator that carries a
    per-cell direction. `offset` is what puts two operators at different points of the SAME cycle;
    `floor` keeps a residual so a gated force never switches fully off, which matters for the
    adhesion (a cell that releases completely does not slide back, it falls over).
    """
    cfg = getattr(op, "gate", None)
    if not cfg:
        return None
    cl = H.level(op.cell_set)
    b0, b1 = cl.state_schema[str(cfg.get("block", "phase"))]
    ph = cl.state[:, b0:b1][:, 0]
    fl = float(cfg.get("floor", 0.0))
    g = 0.5 * (1.0 + torch.cos(ph + float(cfg.get("offset", 0.0))))
    return (fl + (1.0 - fl) * g)[idx]


@register_operator("polar_active_stress", family="motility", set="particle", kind="lateral")
class PolarActiveStress(Lateral):
    """The cytoskeleton extending and retracting along the cell's polarity, as a STRESS.

    particle -[containment]-> particle: reads its cell's `polarity` and `phase`; writes a
    per-particle active stress that `mpm_scatter` adds to the elastic Kirchhoff stress.

        sigma_act = A cos(phi_c + offset) (n n^T - I/3)

    n is the cell's unit polarity, phi_c its phase, and A the amplitude in the run's stress units.
    Over one cycle the sign REVERSES: half of it is extensile along n and half contractile, which
    is the extend-then-retract the cycle is for. A body force cannot do this -- it can only push --
    which is why the active element had to become a stress.

    A STRESS RATHER THAN A FORCE, and the difference is not stylistic. A stress enters the momentum
    balance as a DIVERGENCE, so it is transmitted through the material and conserves momentum; a
    body force is applied pointwise and does neither. It also makes the amplitude interpretable:
    the strain it produces is roughly A / (lambda + 2 mu), so `amplitude_frac` states the target
    strain directly and `amplitude` is derived from the material the operator is acting on. The
    body-force version of this needed a drive 47x larger than the one in use before its
    deformation would have been visible at all, and there was no number in the spec that said so.

    DEVIATORIC, hence CONSTANT VOLUME. `n n^T - I/3` is traceless: it elongates along n and
    contracts across it, to first order without changing the volume. The plain `n n^T` form is
    also a pressure along n, so a cell driven by it would breathe as well as reach -- and a real
    cell reaching forward is not inflating, it is redistributing the material it has.

    ON THE CYTOSKELETON, not on the whole cell: the set that generates the force in a real cell is
    the one that carries it here, which is what putting every organelle in its own set bought.

    Reference: Simha, R. A. & Ramaswamy, S. (2002). Phys. Rev. Lett. 89:058101; Marchetti, M. C.
    et al. (2013). Rev. Mod. Phys. 85:1143-1189 (active stress in polar/nematic gels).
    """
    EMIT = None                       # a stress, consumed by the substep; no integrable delta
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["cell_set"]
    INPUTS = ["particle", "cell"]; OUTPUTS = ["particle"]
    READS = ["polarity", "phase"]; WRITES = []
    MAPS = ["parent"]
    MECHANISM_TAGS = ["active_stress", "motility", "protrusion", "cytoskeleton"]
    PARAM_ROLES = {"amplitude": "active_stress", "amplitude_frac": "target_strain",
                   "offset": "cycle_phase_offset", "cell_set": "polarity_owner",
                   "deviatoric": "volume_preserving"}
    REFERENCE = ("Simha, R. A. & Ramaswamy, S. (2002). Phys. Rev. Lett. 89:058101; "
                 "Marchetti, M. C. et al. (2013). Rev. Mod. Phys. 85:1143-1189.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.cell_set = str(params["cell_set"])
        self.block = str(params.get("block", "polarity"))
        self.phase_block = str(params.get("phase_block", "phase"))
        self.amplitude = params.get("amplitude")
        self.frac = params.get("amplitude_frac")
        self.offset = float(params.get("offset", 0.0))
        self.deviatoric = bool(params.get("deviatoric", True))
        if (self.amplitude is None) == (self.frac is None):
            raise ValueError("polar_active_stress: give `amplitude` (stress units) OR "
                             "`amplitude_frac` (of the set's own lambda + 2 mu), not both and "
                             "not neither -- the second is the one that states a target strain.")

    def forward(self, H, mask=None):
        p = H.level(self.at)
        X = p.get("pos")
        D = X.shape[1]
        cl = H.level(self.cell_set)
        b0, b1 = cl.state_schema[self.block]
        q0, q1 = cl.state_schema[self.phase_block]
        idx = H.lift_index(p.name, self.cell_set)
        n = cl.state[:, b0:b1][:, :D][idx]
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
        ph = cl.state[:, q0:q1][:, 0][idx]
        A = (torch.as_tensor(float(self.amplitude), device=X.device, dtype=X.dtype)
             if self.amplitude is not None
             else float(self.frac) * (p.la + 2.0 * p.mu))
        g = A * torch.cos(ph + self.offset)
        if mask is not None:
            g = g * mask.float()
        M = n[:, :, None] * n[:, None, :]
        if self.deviatoric:
            M = M - torch.eye(D, device=X.device, dtype=X.dtype)[None] / float(D)
        sig = g[:, None, None] * M
        # ALLOCATED ONCE AND WRITTEN IN PLACE. The substep is captured as a CUDA graph, which bakes
        # in the addresses it saw; a fresh tensor per tick would leave the replay reading the one
        # from the tick it was captured on.
        buf = getattr(p, "act_stress", None)
        if buf is None or buf.shape != sig.shape:
            p.register_buffer("act_stress", torch.zeros_like(sig))
            buf = p.act_stress
        buf.copy_(sig)
        return {}
