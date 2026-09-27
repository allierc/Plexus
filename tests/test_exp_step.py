"""tools/exp_step.py: the coarse level and the step rules, on planted specs (2026-09-27)."""
import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp  # noqa: E402
import exp_step as S  # noqa: E402

BASE = {"general": {"name": "b", "seed": 1, "n_frames": 100},
        "sets": {"cell": {"n": 10}},
        "seed": [{"op": "seed_mesh", "seed": 1, "hole": {"axis": [0, 0, 1], "half_angle_deg": 30}}],
        "operators": [{"op": "cell_grow", "gain": 12.0, "rate": 0.002},
                      {"op": "cell_mechanics", "implementation": "apicobasal_contact", "e_cm": 0.15}]}


def test_coarse_ignores_numbers_and_sees_structure():
    a = S.coarse(BASE)
    b = json.loads(json.dumps(BASE))
    b["operators"][0]["gain"] = 3.0
    b["general"]["seed"] = 7
    assert S.coarse(b) == a                                            # numbers and seeds: fine level
    c = json.loads(json.dumps(BASE))
    c["operators"][1]["implementation"] = "apicobasal_region"
    assert S.coarse(c) != a                                            # a variant: coarse
    d = json.loads(json.dumps(BASE))
    d["operators"][1]["k_height"] = 30.0
    assert S.coarse(d) != a                                            # a new key: coarse
    assert "operators[cell_mechanics].e_cm" in S._flat(BASE)           # the differs_by address


@pytest.fixture
def expdir(tmp_path, monkeypatch):
    (tmp_path / "config" / "tissue").mkdir(parents=True)
    (tmp_path / "experiments" / "exp98_t").mkdir(parents=True)
    yaml.safe_dump(BASE, open(tmp_path / "config" / "tissue" / "start.yaml", "w"))
    fm = {"number": 98, "name": "t", "mode": "steps", "gates": "g.yaml", "start": "tissue/start", "main_arm": "hole",
          "step_arms": {"hole": {}, "intact": {"differs_by": {"seed[seed_mesh].hole": "__delete__"}},
                        "high": {"same_as": "hole"}}}
    open(tmp_path / "experiments" / "exp98_t.md", "w").write("---\n" + yaml.safe_dump(fm) + "---\n")
    monkeypatch.setattr(exp, "EXP", str(tmp_path / "experiments"))
    monkeypatch.setattr(S, "ROOT", str(tmp_path))
    return tmp_path


def _landed(d, **kw):
    rec = dict(step=1, state="landed", valid=True, winner_spec="tissue/start", arms=None, **kw)
    with open(d / "experiments" / "exp98_t" / "steps.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def test_first_step_sweeps_the_start_and_expands_arms_and_seeds(expdir):
    P = S.plan(98, {"intent": "baseline", "sweep": {"operators[cell_grow].gain": [1, 3, 10, 30]}})
    assert len(P["cands"]) == 4 and len(P["jobs"]) == 8                # 4 candidates x (hole, intact); `high` is an alias
    intact = [j for j in P["jobs"] if j["arm"] == "intact"][0]["spec"]
    assert "hole" not in intact["seed"][0]                             # __delete__
    assert P["jobs"][0]["spec"]["operators"][0]["gain"] == 1


def test_fewer_than_four_candidates_is_refused(expdir):
    with pytest.raises(SystemExit, match="candidate"):
        S.plan(98, {"sweep": {"operators[cell_grow].gain": [1, 3, 10]}})


def test_a_small_numeric_step_after_a_winner_is_refused(expdir):
    _landed(expdir)
    with pytest.raises(SystemExit, match="not a step"):
        S.plan(98, {"sweep": {"operators[cell_grow].gain": [10, 12, 14, 16]}})
    P = S.plan(98, {"sweep": {"operators[cell_grow].gain": [0.1, 0.3, 1, 3]}})     # >= 10x from 12: a regime step
    assert not P["structural"] and P["jump"] >= 10


def test_a_structural_step_is_accepted(expdir):
    _landed(expdir)
    s = json.loads(json.dumps(BASE))
    s["operators"][1]["k_height"] = 30.0
    yaml.safe_dump(s, open(expdir / "config" / "tissue" / "tall.yaml", "w"))
    P = S.plan(98, {"base": "tissue/tall", "sweep": {"operators[cell_mechanics].k_height": [3, 10, 30, 100]}})
    assert P["structural"] and any("k_height" in x for x in P["coarse_diff"])


def test_sweep_values_must_be_numbers_and_confirm_needs_seeds(expdir):
    with pytest.raises(SystemExit, match="numbers"):
        S.plan(98, {"sweep": {"operators[cell_mechanics].implementation": ["a", "b", "c", "d"]}})
    with pytest.raises(SystemExit, match="seeds"):
        S.plan(98, {"kind": "confirm", "seeds": [1, 2]})
    P = S.plan(98, {"kind": "confirm", "seeds": [1, 2, 3]})
    assert {j["seed"] for j in P["jobs"]} == {1, 2, 3}
    assert all(j["spec"]["seed"][0]["seed"] == j["seed"] for j in P["jobs"])     # every leaf `seed` takes it


def test_outcome_rule():
    a = {"band": "good", "cell.count": 2.0}
    assert not S._outcome_differs(a, {"band": "good", "cell.count": 2.2})
    assert S._outcome_differs(a, {"band": "good", "cell.count": 3.0})
    assert S._outcome_differs(a, {"band": "explosion / chaotic", "cell.count": 2.0})


def test_the_question_must_be_answered(expdir):
    md = expdir / "experiments" / "exp98_t.md"
    fm = yaml.safe_load(md.read_text().split("---")[1])
    fm["question"] = "How does the bud emerge in the last winner, and how does that differ from Wang 2021's mechanism?"
    md.write_text("---\n" + yaml.safe_dump(fm) + "---\n")
    sw = {"operators[cell_grow].gain": [1, 3, 10, 30]}
    with pytest.raises(SystemExit, match="answer the experiment's question"):
        S.plan(98, {"sweep": sw, "answer": "growth escapes at the hole"})
    a = ("the inner mass grows on a clock and the membrane holds the layer everywhere except at the hole, so the "
         "growth escapes there; Wang's surface layer instead outgrows its interior by re-insertion and folds")
    assert S.plan(98, {"sweep": sw, "answer": a})["answer"] == a
