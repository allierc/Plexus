"""The unit vocabulary, pinned: the two tags, the one conversion, and the silence of the undeclared.

WHY THESE TESTS AND NOT OTHERS. `plexus.units` describes quantities and never touches a tensor, so
there is nothing here about the engine. What has to hold is the four things the campaign was written
for -- see `UNITS_REFACTOR.md`:

  1. a dimension catches `rate` meaning per-frame where per-unit-time was meant;
  2. a CONVENTION catches wedge-versus-polyhedron, which is dimensionally perfect and was the defect
     that made G1 run 60% long;
  3. `to_physical` converts, and it is the only thing that does -- the movie's volume panel reading
     0.2 where 200 was true is a missing multiplication by `length_um ** 3`;
  4. an undeclared quantity accuses nobody, because a checker that warns on absence gets switched
     off and then catches nothing at all.
"""
from __future__ import annotations

import pytest

from plexus.units import (DIMENSIONLESS, UNKNOWN, Units, compatible, parse_dim,
                          quantity, to_physical, unit_label, why_incompatible)

# THE RUN'S DECLARATION, as `general.units:` would build it. `declared=True` is the whole
# difference between a number that may carry a unit and one that may not.
UM10 = Units(length_um=10.0, time_s=600.0, declared=True)
NONE = Units()                                   # no `units:` block: dimensionless


# ----------------------------------------------------------------- the algebra
def test_dimensions_compose_by_adding_exponents():
    assert parse_dim("length") ** 3 == parse_dim("volume")
    assert parse_dim("length") / parse_dim("time") == parse_dim("velocity")
    assert parse_dim("force") / parse_dim("area") == parse_dim("stress")
    # energy per unit AREA is a force per unit LENGTH -- the two spellings of a surface tension,
    # and `kappa_s` is one, so the registry must not have them disagree.
    assert parse_dim("energy") / parse_dim("area") == parse_dim("tension")


@pytest.mark.parametrize("text", ["volume", "L^3", "L³", "1/T", "F/L^2", "F/L²",
                                  "L/(F*T)", "1", "stress", "L^-2"])
def test_every_spelling_round_trips_through_str(text):
    """BOTH SPELLINGS IN, ONE SPELLING OUT. A spec may write `L^3` or `L³`; `str` emits the
    superscript and the parser must read its own output back, or a declaration copied out of a
    printed table would stop meaning what it meant."""
    d = parse_dim(text)
    assert d is not UNKNOWN
    assert parse_dim(str(d)) == d


def test_an_unreadable_declaration_is_unknown_and_does_not_raise():
    """A BROKEN ANNOTATION MUST NOT STOP A RUN. This vocabulary feeds a warn-only checker, and a
    typo in a unit string is the least important defect a spec can carry."""
    for bad in ("Q^2", "volume[", "L^^3", "", None, "kg/m^3"):
        assert parse_dim(bad) is UNKNOWN


# ----------------------------------------------------------------- the two tags
def test_the_convention_catches_what_the_dimension_cannot():
    """THE `v_star - Vbirth` DEFECT, AS A TEST. Both are volumes; one is the wedge (origin-referenced
    cones over the mid-surface ring) and one the polyhedron (two caps, one wall per ring edge), and
    on the reference spheroid they read 2.5433 and 1.3508 for the SAME cell."""
    wedge, poly = quantity("volume[wedge]"), quantity("volume[polyhedron]")
    assert wedge.dim == poly.dim                       # dimensionally identical
    assert not compatible(wedge, poly)                 # and still not the same quantity
    assert "wedge" in why_incompatible(wedge, poly)


def test_a_dimension_mismatch_is_named_rather_than_merely_refused():
    assert not compatible("rate", "time")
    assert why_incompatible("rate", "time") == "1/T vs T"


def test_unknown_and_absent_conventions_accuse_nobody():
    """POINT 4. `UNKNOWN` is the absence of a claim, `DIMENSIONLESS` is a claim, and an untagged
    quantity is compatible with a tagged one -- only two DIFFERENT stated conventions are a finding."""
    assert compatible(UNKNOWN, "volume")
    assert compatible("volume", "volume[wedge]")       # nothing declared, nothing contradicted
    assert why_incompatible("volume", "volume[wedge]") is None
    assert parse_dim("dimensionless") is not UNKNOWN
    assert parse_dim("dimensionless") == DIMENSIONLESS


# ----------------------------------------------------------------- the boundary
def test_to_physical_is_the_missing_multiplication():
    """THE BUG THAT IS ON SCREEN TODAY. The movie's volume panel prints a simulation-unit number
    beside a micrometre-cubed label; at `length_um: 10` the true figure is 1000x larger."""
    assert to_physical(0.2, "volume", UM10) == pytest.approx(200.0)
    assert to_physical(1.0, "length", UM10) == pytest.approx(10.0)
    assert to_physical(2.0, "velocity", UM10) == pytest.approx(2.0 * 10.0 / 600.0)


def test_an_undeclared_scale_yields_no_number_and_no_label():
    """THE PAPER'S OWN RULE -- "without it only force RATIOS are meaningful". An undeclared scale is
    not a scale of 1: a caller that gets `None` must print the bare simulation number."""
    assert to_physical(3.0, "stress", UM10) is None          # force_nN was never declared
    assert unit_label("stress", UM10) is None
    # AND A RUN WITH NO `units:` BLOCK AT ALL quotes nothing, which is the honest state of
    # every spec written before that contract existed.
    assert to_physical(3.0, "volume", NONE) is None
    assert unit_label("volume", NONE) is None


def test_labels_use_real_superscripts_and_match_what_was_converted():
    assert unit_label("volume", UM10) == "µm³"
    assert unit_label("area", UM10) == "µm²"
    assert unit_label("length", UM10) == "µm"
    assert unit_label("rate", UM10) == "1/s"
    # a pure ratio carries no unit however much is declared, which is why `fraction` exists
    assert unit_label("fraction", UM10) is None
