"""units -- the physical scale of a model, declared once and never inferred.

WHY THIS EXISTS. Until now a Plexus spec was entirely dimensionless: the world is a unit box, `dt` is a
number, a stiffness is a number, and nothing anywhere says what any of them are in the units a
biologist measures. That is fine for a mechanism and fatal for a claim. It has already cost real
mistakes in this codebase:

  * a basement membrane whose thickness was `2e-3` box units, which -- once the box was calibrated
    against cell size -- is ~2.4 um against the ~0.1 um a basement membrane actually is, i.e. 24x too
    thick, with every length defined in terms of it too big by the same factor;
  * a modulus written as `E = 400` and quoted against Candiello et al.'s 0.4-3 MPa, which is a
    comparison between a number and a pressure;
  * a turnover time `tau_bm = 40` whose being 4-14 h depended on an unstated 9-18 minutes per frame.

THE CONTRACT, and it is deliberately minimal: THREE base scales, because mechanics needs three.

    length_um   micrometres per simulation length unit   (the world box is 1 unit by convention)
    time_s      seconds per simulation time unit         (DEFAULT 1.0 -- `dt` is in SECONDS)
    force_nN    nanonewtons per simulation force unit    (optional; without it, only force RATIOS
                                                          are meaningful and the loader says so)

plus an optional `amount` for field concentrations (proteases, morphogens), which is a fourth base
unit and is not needed by anything mechanical.

EVERYTHING ELSE IS DERIVED and must never be declared separately, because a second declaration is a
second chance to disagree: area is length^2, a velocity is length/time, a stress is force/length^2, a
2D membrane modulus is force/length, an energy is force*length, a rate is 1/time.

    quantity                 dimension          from the three
    position, radius, l0     L                  length_um
    area                     L^2                length_um^2
    volume                   L^3                length_um^3
    velocity                 L/T                length_um / time_s
    rate, 1/tau              1/T                1 / time_s
    force                    F                  force_nN
    stress, Young's modulus  F/L^2              force_nN / length_um^2      -> Pa via 1e-3
    2D (membrane) modulus    F/L                force_nN / length_um        -> N/m via 1e-3
    energy                   F L                force_nN * length_um
    mobility 1/gamma         L/(F T)            length_um / (force_nN * time_s)
    areal mass density       M/L^2              (not derivable; declare `amount` if you need it)

WHAT IS DELIBERATELY NOT PROVIDED. There is no automatic conversion of state buffers and no unit
checking of operator ARITHMETIC. Both would be large and both would be a false comfort: the operators
compute in simulation units, exactly as now, and `Units` is a declaration attached to the run so that
what comes OUT can be reported in micrometres and seconds, and so that a spec which makes a physical
claim can be asked what it is claiming it in. A model with no `units:` block is dimensionless and no
statement about it may carry a unit -- which is the state every existing spec is in, and is why the
loader warns rather than fails.

WHAT WAS ADDED LATER, AND WHY THE SENTENCE ABOVE NEEDED ONE WORD OF QUALIFICATION. The arithmetic is
still unchecked. What is checked is the BOUNDARY: what a spec hands an operator, and what the plot
reports back out. Neither is arithmetic, both are one comparison at load, and both had failed in
production by the time this was written -- see `UNITS_REFACTOR.md`. `Dim` and `Quantity` below are
that vocabulary. They describe quantities and never touch a tensor, so nothing is wrapped and the
zero-copy path into warp is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Units:
    """The three base scales of a run, plus an optional amount. All conversions are DERIVED."""

    length_um: float = 1.0       # micrometres per simulation length unit
    time_s: float = 1.0          # seconds per simulation time unit (default: dt IS seconds)
    force_nN: float | None = None    # nanonewtons per simulation force unit; None = ratios only
    amount: str | None = None    # free-text label for a field's amount ("nM", "molecules", ...)
    declared: bool = False       # False when no `units:` block was given: the run is dimensionless

    # -- derived, and there is exactly one way to compute each ---------------------------------
    @property
    def area_um2(self):
        return self.length_um ** 2

    @property
    def volume_um3(self):
        return self.length_um ** 3

    @property
    def velocity_um_per_s(self):
        return self.length_um / self.time_s

    @property
    def rate_per_s(self):
        return 1.0 / self.time_s

    @property
    def stress_Pa(self):
        """force/length^2. nN/um^2 = 1e-9 N / 1e-12 m^2 = 1e3 Pa, hence the factor."""
        return None if self.force_nN is None else 1.0e3 * self.force_nN / self.length_um ** 2

    @property
    def tension_N_per_m(self):
        """force/length, the 2D modulus a membrane actually has. nN/um = 1e-9 N / 1e-6 m = 1e-3 N/m."""
        return None if self.force_nN is None else 1.0e-3 * self.force_nN / self.length_um

    @property
    def energy_aJ(self):
        """force*length. nN*um = 1e-9 N * 1e-6 m = 1e-15 J = 1000 aJ."""
        return None if self.force_nN is None else 1.0e3 * self.force_nN * self.length_um

    def seconds(self, sim_time):
        return sim_time * self.time_s

    def hours(self, sim_time):
        return sim_time * self.time_s / 3600.0

    def um(self, sim_length):
        return sim_length * self.length_um

    def describe(self):
        if not self.declared:
            return ("units: NONE DECLARED -- this run is dimensionless and no result from it may be "
                    "quoted with a unit")
        f = "ratios only" if self.force_nN is None else f"{self.force_nN:g} nN"
        s = "" if self.stress_Pa is None else f", 1 stress unit = {self.stress_Pa:.4g} Pa"
        return (f"units: 1 length unit = {self.length_um:g} um, 1 time unit = {self.time_s:g} s "
                f"({self.time_s / 3600.0:.4g} h), 1 force unit = {f}{s}"
                + (f", amount in {self.amount}" if self.amount else ""))


def parse(raw):
    """Build `Units` from a spec's `general.units:` block. Absent -> dimensionless, `declared=False`.

    Only the three base scales (and `amount`) are accepted. Declaring a derived scale is an ERROR
    rather than a convenience: `area_um2: 4` alongside `length_um: 3` is a contradiction the loader
    would otherwise have to choose between silently.
    """
    if raw is None:
        return Units(declared=False)
    if not isinstance(raw, dict):
        raise ValueError("general.units must be a mapping, e.g. {length_um: 1176, time_s: 600}")
    allowed = {"length_um", "time_s", "force_nN", "amount"}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(
            f"general.units has undeclarable entries {sorted(extra)}. Only {sorted(allowed)} are "
            f"base scales; area, velocity, stress, rate and energy are DERIVED from them (see "
            f"plexus/units.py) and declaring one is a second chance to disagree with the first.")
    for k in ("length_um", "time_s", "force_nN"):
        if k in raw and raw[k] is not None and float(raw[k]) <= 0:
            raise ValueError(f"general.units.{k} must be positive, got {raw[k]}")
    return Units(length_um=float(raw.get("length_um", 1.0)),
                 time_s=float(raw.get("time_s", 1.0)),
                 force_nN=(None if raw.get("force_nN") is None else float(raw["force_nN"])),
                 amount=(None if raw.get("amount") is None else str(raw["amount"])),
                 declared=True)


# =============================================================== the quantity vocabulary
#
# TWO TAGS, NOT ONE, AND THAT IS THE WHOLE DESIGN. Half of what has gone wrong in this tree is
# dimensional: `cell_grow.rate` meant "fraction of itself a cell adds per FRAME" while the engine
# integrated it per unit of simulation time, and the movie's volume panel appends a micrometre-cubed
# label without ever multiplying by `length_um ** 3`, so it is wrong by a factor of 1000 on every
# tissue run. A dimension catches both.
#
# The other half is not, and no dimensional system ever catches it. `cell_grow`'s size checkpoint
# computed `v_star - Vbirth`, where `v_star` is a WEDGE volume -- the sum of origin-referenced cones
# over a cell's mid-surface ring -- and `Vbirth` is the POLYHEDRON volume of the same cell, two caps
# and one wall per ring edge. Measured on the reference spheroid those read 2.5433 and 1.3508 FOR
# ONE CELL. Both are lengths cubed. The subtraction is dimensionally perfect and biologically
# meaningless, and it makes G1 run 60% long with nothing in the output to show for it. So a
# quantity carries a `Dim` AND an optional CONVENTION, and the convention is what catches that class.
#
# NO DEPENDENCY, AND NOT `astropy` OR `pint`. Those wrap values, and the values here are torch
# tensors that `warp.from_torch` takes zero-copy. This project has already paid once for handing
# warp a view it did not expect -- a strided column whose `index_add` reduced in a different order
# and moved `pos` by 1.09e-04 on the first frame. Nothing below ever touches a tensor.

from typing import NamedTuple, Optional          # noqa: E402  (grouped with its own section)

# SUPERSCRIPTS, IN THE CODE AND ON THE SCREEN. `L^3` is an ASCII transliteration of an exponent, and
# `um^3` was already called out in the renderer as "a choice, not a limitation": VTK's text renderer
# and every terminal this project runs in take the real characters. The PARSER accepts both
# spellings, so a spec may be written either way and a round trip through `str` holds.
_SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻"}
_UNSUP = {v: k for k, v in _SUP.items()}


def sup(n: int) -> str:
    """3 -> the superscript three; 1 -> the empty string, because a superscript one is noise."""
    return "" if n == 1 else "".join(_SUP[c] for c in str(n))


def _desup(s: str) -> str:
    """Superscript digits back to ASCII with a caret, so the parser reads a single spelling."""
    out: list = []
    for c in s:
        if c in _UNSUP:
            if out and out[-1] not in "^-0123456789":
                out.append("^")
            out.append(_UNSUP[c])
        else:
            out.append(c)
    return "".join(out)


class Dim(NamedTuple):
    """Integer exponents of the three base scales plus the optional amount.

    THE THREE ARE `Units`' OWN, above: `length_um`, `time_s`, `force_nN`, because "mechanics needs
    three". `A` is the fourth for a field's concentration, which nothing mechanical needs.

    A TUPLE OF FOUR INTEGERS RATHER THAN A DICT, because a dimension is compared far more often than
    it is built: equality is one tuple compare, and `NamedTuple` gives that plus hashability free.
    """
    L: int = 0
    T: int = 0
    F: int = 0
    A: int = 0

    def __mul__(self, other):
        return Dim(self.L + other.L, self.T + other.T, self.F + other.F, self.A + other.A)

    def __truediv__(self, other):
        return Dim(self.L - other.L, self.T - other.T, self.F - other.F, self.A - other.A)

    def __pow__(self, n: int):
        return Dim(self.L * n, self.T * n, self.F * n, self.A * n)

    def __str__(self) -> str:
        """`L³`, `1/T`, `F/L²`, `1` -- a spelling `parse_dim` accepts, so a round trip holds."""
        if self == DIMENSIONLESS:
            return "1"
        num = [s + sup(e) for s, e in zip("LTFA", self) if e > 0]
        den = [s + sup(-e) for s, e in zip("LTFA", self) if e < 0]
        top = "*".join(num) if num else "1"
        if not den:
            return top
        bot = "*".join(den)
        return f"{top}/{bot}" if len(den) == 1 else f"{top}/({bot})"


DIMENSIONLESS = Dim()
LENGTH = Dim(L=1)
TIME = Dim(T=1)
FORCE = Dim(F=1)
AMOUNT = Dim(A=1)

# THIS FILE'S OWN TABLE, TRANSCRIBED. The docstring above lists every derived quantity and how it is
# built from the three base scales; this is that table and nothing else, so an operator writes
# `"volume"` and cannot disagree with it. A NAME IS PREFERRED TO A FORMULA in a declaration for
# exactly that reason: `"L^3"` and `"volume"` mean the same thing, and only one of them stays right
# if the table is ever revised.
NAMED = {
    "dimensionless": DIMENSIONLESS,
    "fraction": DIMENSIONLESS,           # a ratio that happens to lie in [0, 1]
    "count": DIMENSIONLESS,              # a number of entities
    "angle": DIMENSIONLESS,              # radians or degrees: a ratio of two lengths either way
    "length": LENGTH,                    # position, radius, rest length, thickness
    "position": LENGTH, "radius": LENGTH, "thickness": LENGTH,
    "area": Dim(L=2),
    "volume": Dim(L=3),
    "time": TIME,
    "velocity": Dim(L=1, T=-1),
    "rate": Dim(T=-1),                   # 1/tau, a per-unit-time frequency
    "force": FORCE,
    "stress": Dim(L=-2, F=1),            # also Young's modulus -> Pa
    "modulus": Dim(L=-2, F=1),
    "membrane_modulus": Dim(L=-1, F=1),  # the 2D modulus Y2 = E*T -> N/m
    "tension": Dim(L=-1, F=1),           # a surface tension: energy per area IS force per length
    "line_tension": FORCE,               # energy per length IS a force
    "energy": Dim(L=1, F=1),
    "mobility": Dim(L=1, T=-1, F=-1),    # 1/gamma
    "amount": AMOUNT,
    "concentration": Dim(L=-3, A=1),
}

# THE ABSENCE OF A CLAIM, and deliberately not a `Dim`. `DIMENSIONLESS` is a claim -- this number is
# a pure ratio. `UNKNOWN` is "nobody has said what this is yet". A checker that read undeclared as
# dimensionless would warn on every one of the ~145 operators nobody has annotated, and everyone
# would learn to ignore it within a day. `UNKNOWN` compares compatible with everything, silently.
UNKNOWN = "unknown"


def parse_dim(s):
    """`"volume"`, `"L³"` or `"L^3"`, `"1/T"`, `"F/L²"`, `"L/(F*T)"`, `"1"` -> a `Dim`.

    A NAME FIRST, A FORMULA SECOND: `NAMED` is consulted before the parser, so this file's table is
    the authority and a formula is the escape hatch rather than the habit.

    Returns `UNKNOWN` rather than raising on anything unreadable. This feeds a WARN-ONLY checker, and
    a typo in a unit annotation must not stop a run -- it is the least important defect a spec can
    carry, and a checker whose own failures are fatal is worse than no checker.
    """
    if s is None or s is UNKNOWN:
        return UNKNOWN
    if isinstance(s, Dim):
        return s
    t = _desup(str(s).strip())
    if not t:
        return UNKNOWN
    if t.lower() in NAMED:
        return NAMED[t.lower()]
    try:
        num, _, den = t.partition("/")
        d = _parse_product(num)
        if den:
            d = d / _parse_product(den)
        return d
    except (ValueError, KeyError):
        return UNKNOWN


def _parse_product(s):
    """`L^3`, `F*L`, `(F*T)`, `1` -> a `Dim`. Raises on anything else; `parse_dim` catches."""
    t = s.strip().strip("()").strip()
    if t in ("", "1"):
        return DIMENSIONLESS
    out = DIMENSIONLESS
    for term in t.split("*"):
        base, _, exp = term.strip().partition("^")
        base = base.strip().upper()
        if base not in ("L", "T", "F", "A"):
            raise ValueError(f"unknown base scale {base!r}")
        out = out * (Dim(**{base: 1}) ** (int(exp) if exp.strip() else 1))
    return out


class Quantity(NamedTuple):
    """A dimension plus an optional CONVENTION: what it is, and which way it was measured.

    THE CONVENTION IS A BARE STRING AND NOT AN ENUM, because the set of them is not knowable in
    advance and a closed enum would make adding one a change to this file. Two exist today:

        volume[wedge]        `face_geometry_3d`: origin-referenced cones over the mid-surface ring
        volume[polyhedron]   `apicobasal_geometry_3d`: divergence theorem over two caps and one
                             wall per ring edge

    `None` means no convention distinction exists for this quantity -- a length is a length. That is
    NOT the same as a convention nobody has declared yet, which is what `UNKNOWN` says.
    """
    dim: object                          # a `Dim`, or `UNKNOWN`
    convention: Optional[str] = None

    def __str__(self) -> str:
        d = "unknown" if self.dim is UNKNOWN else str(self.dim)
        return f"{d}[{self.convention}]" if self.convention else d


def quantity(s):
    """`"volume[polyhedron]"` -> `Quantity(Dim(L=3), "polyhedron")`. The spec-facing entry point.

    NOT NAMED `parse`, because `parse` in this module already builds the run's `Units` from a
    `general.units:` block and has done since that contract was written. Two functions called
    `parse` in one file, one taking a mapping of scales and one taking a quantity string, is the
    kind of collision that is obvious for a week and invisible afterwards.
    """
    if s is None:
        return Quantity(UNKNOWN)
    if isinstance(s, Quantity):
        return s
    t = str(s).strip()
    conv = None
    if t.endswith("]") and "[" in t:
        t, _, rest = t.partition("[")
        conv = rest[:-1].strip() or None
    return Quantity(parse_dim(t), conv)


def compatible(a, b):
    """Do these two describe the same kind of number?

    THE TWO TESTS ARE INDEPENDENT AND BOTH ARE PERMISSIVE ON ABSENCE:

      dimension    equal, or either is `UNKNOWN`. An undeclared quantity accuses nobody.
      convention   equal, or either is `None`. `None` means "no distinction exists here", so a plain
                   `volume` is compatible with `volume[wedge]`: whoever declared nothing has made no
                   claim to contradict. Only two DIFFERENT STATED conventions are a finding, which
                   is exactly the `v_star - Vbirth` case and nothing wider.
    """
    a, b = quantity(a), quantity(b)
    if a.dim is not UNKNOWN and b.dim is not UNKNOWN and a.dim != b.dim:
        return False
    if a.convention and b.convention and a.convention != b.convention:
        return False
    return True


def why_incompatible(a, b):
    """One short clause naming the disagreement, for the checker's single line. `None` if ok."""
    a, b = quantity(a), quantity(b)
    if a.dim is not UNKNOWN and b.dim is not UNKNOWN and a.dim != b.dim:
        return f"{a.dim} vs {b.dim}"
    if a.convention and b.convention and a.convention != b.convention:
        return f"{a.convention} vs {b.convention} (same dimension, different measurement)"
    return None


def to_physical(value, dim, units):
    """Simulation units -> physical units. THE ONLY FUNCTION IN THE TREE ALLOWED TO CONVERT.

        physical = value * length_um**L * time_s**T * force_nN**F

    `units` is this module's `Units`. Returns `None` -- "this cannot be quoted" -- when the run is
    undeclared, or when a base scale the dimension needs was not given. That is the docstring's own
    rule at the top of this file: without `force_nN`, "only force RATIOS are meaningful". A caller
    handed `None` must print the bare simulation number with no unit rather than invent a scale.

    NOTHING ELSE CONVERTS. State buffers are not converted and operators keep computing in
    simulation units; this exists for the reporting boundary -- scale bar, curve axes, panel
    readouts, colour legend -- which is where a simulation number becomes a claim about a tissue.
    """
    d = dim if isinstance(dim, Dim) else parse_dim(dim)
    if d is UNKNOWN or value is None or units is None or not getattr(units, "declared", False):
        return None
    if d.A != 0:                         # `amount` is a LABEL, not a number: nothing to scale by
        return None
    scale = 1.0
    for exp, s in ((d.L, units.length_um), (d.T, units.time_s), (d.F, units.force_nN)):
        if exp == 0:
            continue
        if s is None:
            return None                  # an undeclared scale is not a scale of 1
        scale *= float(s) ** exp
    return value * scale


def unit_label(dim, units):
    """`µm`, `µm²`, `µm³`, `s`, `µm/s`, `nN` ... or `None` when the run cannot quote one.

    Built from the same declaration `to_physical` uses, so a LABEL CAN NEVER APPEAR OVER A NUMBER
    THAT WAS NOT CONVERTED -- which is the defect this campaign found on screen: the movie's volume
    panel printed `µm³` beside a raw simulation value, wrong by `length_um ** 3` and so by 1000 on
    every tissue run in the tree.
    """
    d = dim if isinstance(dim, Dim) else parse_dim(dim)
    if d is UNKNOWN or d == DIMENSIONLESS or units is None or not getattr(units, "declared", False):
        return None
    num, den = [], []
    for exp, s, sym in ((d.L, units.length_um, "µm"), (d.T, units.time_s, "s"),
                        (d.F, units.force_nN, "nN")):
        if exp == 0:
            continue
        if s is None:
            return None                  # cannot label what cannot be converted
        (num if exp > 0 else den).append(sym + sup(abs(exp)))
    if not num and not den:
        return None
    top = "·".join(num) if num else "1"
    return top if not den else top + "/" + "·".join(den)
