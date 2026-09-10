"""The unit vocabulary, pinned: the two tags, the one conversion, and the silence of the undeclared.

WHY THESE TESTS AND NOT OTHERS. `plexus.units` describes quantities and never touches a tensor, so
there is nothing here about the engine. What has to hold is the four things the campaign was written
for -- see `notes/campaigns/UNITS_REFACTOR.md`:

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


# ----------------------------------------------------------------- the checker
def test_the_checker_never_raises_and_never_stops_a_load():
    """THE ONE PROPERTY THAT MATTERS MOST. A units annotation is the least load-bearing thing in
    any spec, and a typo in one must never cost a 600-frame run. `check` catches its own
    exceptions, so even a spec object it cannot understand costs one printed line."""
    from plexus.units_check import check

    class Broken:                                   # not a Spec at all
        pass

    said = []
    out = check(Broken(), {"nonsense": object}, emit=said.append)
    assert isinstance(out, list)                    # returned, not raised
    assert said                                     # and it said something


def test_the_checker_names_a_convention_disagreement_between_two_operators():
    """WEDGE AGAINST POLYHEDRON, WHICH IS THE DEFECT THE CONVENTION TAG EXISTS FOR. Two operators
    that believe different things about the same block is the same error whether or not the spec
    ever declared one."""
    from plexus.units_check import check

    class A:
        BLOCK_UNITS = {"V0f": "volume[wedge]"}

    class B:
        BLOCK_UNITS = {"V0f": "volume[polyhedron]"}

    class Sim:
        units, dt, sets, operators = Units(), 1.0, {}, []

    said = []
    out = check(Sim(), {"grower": A, "energy": B}, emit=said.append)
    assert any("V0f" in w and "wedge" in w and "polyhedron" in w for w in out), out


def test_a_block_the_spec_declares_is_checked_against_what_the_operator_believes():
    from plexus.units_check import check

    class Energy:
        BLOCK_UNITS = {"V0f": "volume[polyhedron]"}

    class Sim:
        units, dt, operators = Units(), 1.0, []
        sets = {"cell": {"state": {"V0f": {"width": 1, "unit": "volume[wedge]"}}}}

    out = check(Sim(), {"energy": Energy}, emit=lambda *_: None)
    assert any("sets.cell.V0f" in w for w in out), out


def test_an_unparseable_annotation_is_reported_because_absence_is_not():
    """A TYPO IS NOT AN ABSENCE. `parse_dim` swallows a malformed string into UNKNOWN so a run is
    never stopped, which is exactly what would make a misspelt unit indistinguishable from a
    missing one -- so the checker says which it was."""
    from plexus.units_check import check

    class Sim:
        units, dt, operators = Units(), 1.0, []
        sets = {"cell": {"state": {"V0f": {"width": 1, "unit": "kg/m^3"}}}}

    out = check(Sim(), {}, emit=lambda *_: None)
    assert any("does not parse" in w for w in out), out


def test_an_undeclared_block_is_silent():
    """POINT 4 AGAIN, AT THE CHECKER. 145 of 177 operator classes declare nothing, by design; a
    checker that named them would be switched off on its first run."""
    from plexus.units_check import check

    class Sim:
        units, dt, operators = Units(), 1.0, []
        sets = {"cell": {"state": {"V0f": {"width": 1}, "chem": 2}}}

    assert check(Sim(), {}, emit=lambda *_: None) == []
