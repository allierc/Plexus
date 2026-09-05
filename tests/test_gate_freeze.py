"""R1(d) 3 and 4: a gate can be denominated in volume and tension, and a moved threshold is caught.

4 IS THE ONE THAT MATTERS. `run_gates.py`'s header has said since June that `--freeze-reference`
records a sha1 of the `_gate:` block and that "every later grading re-hashes and refuses to grade if
the block has moved, naming the rows that changed", closing with: the paper's rule -- *a threshold
chosen after seeing the number is not a threshold* -- "is otherwise an honour system". `gate_sha1`
appeared three times in that file: computed, written, printed. NEVER COMPARED. The honour system the
header says we escaped was the one we were on, for every gate in the tree.

THE TEST THAT COUNTS IS THE TAMPERING ONE. A drift check that only ever returns None on a clean tree
is indistinguishable from a function that returns None. So each case here MOVES something a person
would plausibly move after seeing a result -- a threshold, a tier, a basis -- and asserts the check
names the row AND the field.

3 IS WIRING, NOT ARITHMETIC. `plexus.units` already derives `volume_um3` and `tension_N_per_m`;
`_convert` simply did not offer them, and raised INSIDE the grading try, so a row denominated in
um^3 or mN/m scored INFRA_FAIL rather than passing or failing. That silently disables AB-M6, the one
non-circular measurement row in the apico-basal table.
"""
from __future__ import annotations

import copy

import pytest

import run_gates as RG


@pytest.fixture(scope="module")
def gate():
    g = RG.gates()
    if "00_spheroid" not in g:
        pytest.skip("gate 00 is not on disk")
    return g["00_spheroid"][1]


def _patched(cfg, name, **fields):
    t = copy.deepcopy(cfg)
    for m in t["_gate"]["measures"]:
        if m["name"] == name:
            m.update(fields)
    return t


# --------------------------------------------------------------------------------------------- #
#  4. the read-back
# --------------------------------------------------------------------------------------------- #
def test_every_frozen_gate_is_clean_today():
    """It must land GREEN when switched on, or nobody will switch it on.

    All three references were hand-checked before this was written and still matched; if one of them
    drifts later, that is the check working, not this test being wrong.
    """
    for gid, (_path, cfg) in RG.gates().items():
        assert RG._frozen_drift(gid, cfg) is None, f"{gid} has drifted from its frozen block"


def test_a_moved_threshold_is_caught_and_named(gate):
    """The case the whole mechanism exists for: run it, see 0.83, edit the row to le: 0.9."""
    d = RG._frozen_drift("00_spheroid", _patched(gate, "reservoir_headroom", **{"assert": {"le": 0.55}}))
    assert d is not None, "a moved threshold was not caught"
    assert d["frozen"] != d["now"]
    assert "reservoir_headroom.assert" in d["rows"], d["rows"]


def test_a_relabelled_tier_is_caught(gate):
    """Promoting a bookkeeping row to `measurement` is a claim about evidence, not a typo."""
    d = RG._frozen_drift("00_spheroid", _patched(gate, "reservoir_headroom", tier="measurement"))
    assert "reservoir_headroom.tier" in d["rows"]


def test_a_relabelled_basis_is_caught(gate):
    """`reference` -> `literature` turns a regression pin into a claim about cells. It is the
    relabel the roll-up's honest-split column exists to prevent, so it must not be silent."""
    d = RG._frozen_drift("00_spheroid", _patched(gate, "reservoir_headroom", basis="literature"))
    assert "reservoir_headroom.basis" in d["rows"]


def test_a_deleted_row_is_caught(gate):
    """Deleting the row that fails is the cheapest way to make a gate pass."""
    t = copy.deepcopy(gate)
    t["_gate"]["measures"] = [m for m in t["_gate"]["measures"] if m["name"] != "no_nan"]
    assert "no_nan (removed)" in RG._frozen_drift("00_spheroid", t)["rows"]


def test_a_new_row_is_caught(gate):
    """Adding a row after the freeze is not tampering, but it is still a changed block: the freeze
    is the whole table, so it has to be re-taken deliberately."""
    t = copy.deepcopy(gate)
    t["_gate"]["measures"].append(dict(name="_added_later", tier="bookkeeping", basis="identity",
                                       fn="cell_count", reduce="last", unit="cells",
                                       why="x", **{"assert": {"eq": 1}}))
    assert "_added_later (new)" in RG._frozen_drift("00_spheroid", t)["rows"]


def test_an_unfrozen_gate_is_not_refused(gate):
    """A gate with no reference has nothing to be inconsistent with, and must still grade."""
    assert RG._frozen_drift("a_gate_never_frozen", gate) is None


# --------------------------------------------------------------------------------------------- #
#  3. the two unit kinds
# --------------------------------------------------------------------------------------------- #
def test_volume_converts_as_length_cubed():
    """L^3, derived from `plexus.units` rather than recomputed here."""
    cfg = {"general": {"units": {"length_um": 10.0, "time_s": 600.0}}}
    RG.GM.PHYSICAL["_probe_volume"] = ("volume", "um^3")
    try:
        assert RG._convert(2.0, "_probe_volume", cfg) == pytest.approx(2.0 * 10.0 ** 3)
    finally:
        del RG.GM.PHYSICAL["_probe_volume"]


def test_tension_needs_a_force_scale_and_says_so():
    """A tension is force/length: without `force_nN` the number has no N/m to be quoted in, and the
    row must RAISE rather than invent one. That is the rule the units block exists to enforce --
    it is how a membrane once came to be 24x too thick and a modulus came to be quoted as a
    pressure."""
    RG.GM.PHYSICAL["_probe_tension"] = ("tension", "mN/m")
    try:
        with pytest.raises(KeyError, match="force_nN"):
            RG._convert(1.0, "_probe_tension", {"general": {"units": {"length_um": 10.0}}})
        cfg = {"general": {"units": {"length_um": 10.0, "time_s": 1.0, "force_nN": 5.0}}}
        # nN/um = 1e-3 N/m, then mN/m: 1e3 * 1e-3 * force_nN / length_um
        assert RG._convert(1.0, "_probe_tension", cfg) == pytest.approx(5.0 / 10.0)
    finally:
        del RG.GM.PHYSICAL["_probe_tension"]
