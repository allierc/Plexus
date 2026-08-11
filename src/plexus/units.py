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
checking of operator arithmetic. Both would be large and both would be a false comfort: the operators
compute in simulation units, exactly as now, and `Units` is a declaration attached to the run so that
what comes OUT can be reported in micrometres and seconds, and so that a spec which makes a physical
claim can be asked what it is claiming it in. A model with no `units:` block is dimensionless and no
statement about it may carry a unit -- which is the state every existing spec is in, and is why the
loader warns rather than fails.
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
