"""Stage-0 tests: the spec schema, the option vocabulary, and the tier-1 gates that are cheap.

These are the gates the reference calls verification: they can be run before any biology is
claimed, they cost seconds, and they catch the errors that are otherwise invisible.
"""

from __future__ import annotations

import os
import sys

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.dirname(_HERE)
_PLEXUS_SRC = os.path.abspath(os.path.join(_PROTO, "..", "..", "src"))
for p in (_PROTO, _PLEXUS_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import engine
import gates as gates_mod
import spec_schema

CONFIG = os.path.join(_PROTO, "config", "toy_small.yaml")


# --- the schema ---------------------------------------------------------------------------- #

def test_reference_config_loads():
    fit = spec_schema.load(CONFIG)
    assert fit.name == "toy_small"
    assert fit.units.declared
    assert fit.model.encoder_decoder in spec_schema.ENCODER_DECODER


def test_yaml_bare_off_is_not_a_boolean():
    """`encoder_decoder: off` is YAML 1.1 False. The schema must accept it as the string."""
    raw = yaml.safe_load(open(CONFIG))
    assert raw["model"]["encoder_decoder"] is False, "pyyaml stopped coercing bare off"
    assert spec_schema.load(CONFIG).model.encoder_decoder == "off"


@pytest.mark.parametrize("section,key,bad", [
    ("model", "message", "graph_cast"),
    ("model", "embedding", "learned"),
    ("model", "observation", "gcamp"),
    ("data", "source", "hdf5"),
    ("training", "target", "derivative"),
])
def test_typos_in_the_vocabulary_fail_loudly(tmp_path, section, key, bad):
    raw = yaml.safe_load(open(CONFIG))
    raw[section][key] = bad
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=f"{section}.{key}"):
        spec_schema.load(str(p))


def test_units_block_is_required(tmp_path):
    raw = yaml.safe_load(open(CONFIG))
    del raw["general"]["units"]
    p = tmp_path / "nounits.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="units"):
        spec_schema.load(str(p))


def test_derived_units_may_not_be_declared(tmp_path):
    """plexus/units.py: everything but the three base scales is DERIVED, and a second declaration
    is a second chance to disagree."""
    raw = yaml.safe_load(open(CONFIG))
    raw["general"]["units"]["velocity_um_per_s"] = 3.0
    p = tmp_path / "extraunits.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="DERIVED"):
        spec_schema.load(str(p))


def test_encoder_decoder_on_requires_a_declared_mesh(tmp_path):
    raw = yaml.safe_load(open(CONFIG))
    raw["model"]["encoder_decoder"] = "on"
    raw["model"].pop("mesh_resolution", None)
    p = tmp_path / "nomesh.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="mesh_resolution"):
        spec_schema.load(str(p))


def test_simple_message_refuses_multiple_passes(tmp_path):
    """The simple form carries no edge state, so repeating it is a different model than it says."""
    raw = yaml.safe_load(open(CONFIG))
    raw["model"]["message"] = "simple"
    raw["model"]["n_passes"] = 4
    p = tmp_path / "simple4.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="edge state"):
        spec_schema.load(str(p))


# --- the gate table ------------------------------------------------------------------------ #

def test_every_gate_has_a_tier_a_stage_and_a_unit():
    for g in gates_mod.build_table().values():
        assert g.tier in gates_mod.TIERS, g.gid
        assert isinstance(g.stage, int) and g.stage >= 0, g.gid
        assert g.unit, f"{g.gid} has no unit; a threshold must say what it is OF"


def test_measurement_thresholds_are_not_in_mesh_units():
    """The rule the units block yields: a gate's threshold belongs in the unit of the phenomenon,
    not of the mesh. A threshold in grid cells is easiest to pass and reads as an endorsement."""
    mesh = {"grid cells", "cells", "voxels", "pixels", "steps"}
    bad = [g.gid for g in gates_mod.build_table().values()
           if g.tier == "measurement" and g.unit.lower() in mesh]
    assert not bad, f"measurement gates denominated in mesh units: {bad}"


def test_gate_ids_are_unique_and_ordered():
    table = gates_mod.build_table()
    ids = sorted(table, key=gates_mod._order)
    assert len(ids) == len(set(ids))
    assert ids[0] == "G1" and ids[1] == "G1b"


def test_outcome_is_skip_until_measured():
    g = gates_mod.build_table()["G9"]
    assert g.outcome == gates_mod.SKIP
    assert g.record(0.95).outcome == gates_mod.PASS
    assert g.record(0.5).outcome == gates_mod.FAIL
    assert g.record(None).outcome == gates_mod.SKIP


# --- the tier-1 checks themselves ------------------------------------------------------------ #

def test_G1_all_24_option_combinations_parse():
    assert len(engine._all_option_combinations()) == 24
    measured, note = engine.gate_G1_parse(CONFIG)
    assert measured == 24, note


def test_G2_no_dataset_identity_in_code():
    measured, note = engine.gate_G2_no_hardcoding(CONFIG)
    assert measured == 0, note


def test_G7_units_declared():
    measured, note = engine.gate_G7_units(CONFIG)
    assert measured == 1.0, note


def test_gate_artifacts_are_written(tmp_path):
    n_fail = engine.run_gates(CONFIG, str(tmp_path))
    assert n_fail == 0
    assert (tmp_path / "gates.csv").exists()
    tex = (tmp_path / "gates_table.tex").read_text()
    assert r"\tblGates" in tex and r"\tierProportion" in tex
    assert "<" not in tex.replace("$<$", ""), "raw < in LaTeX text mode renders as inverted punctuation"
