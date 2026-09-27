"""The shared scorer's own mechanics (tools/exp_gate_score.py), kept out of any one experiment's tests."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]


def test_cache_is_refreshed_when_the_run_relands_or_the_ruler_changes(tmp_path, monkeypatch):
    """A cached value is reused only while it is younger than the run's trajectory and the ruler's last
    change (2026-09-27): a run re-landed under the same name, or a changed ruler, is measured again."""
    import json
    import time
    import exp_gate_score as S
    import exp_measures
    import exp_measures.common as C
    run_d = tmp_path / "tissue" / "expXX_a_s1"
    run_d.mkdir(parents=True)
    tj = run_d / "trajectory.npz"
    tj.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(S, "EXP", str(tmp_path))
    monkeypatch.setattr(S, "run_exists", lambda r: True)
    monkeypatch.setattr(C, "run_dir", lambda r: str(run_d))
    monkeypatch.setattr(exp_measures, "run_measure", lambda m, r, **kw: calls.append(m) or {"v": len(calls)})
    monkeypatch.setattr(C, "CHANGED", {})
    G = {"runs": {"a": "tissue/expXX_a_s{seed}"}, "seeds": [1], "measures": [{"measure": "m.x"}]}
    old = time.time() - 3600
    os.utime(tj, (old, old))
    assert S.measure_all(G, 99)[("a", 1)]["m.x.v"] == 1
    assert S.measure_all(G, 99)[("a", 1)]["m.x.v"] == 1 and len(calls) == 1      # cached
    os.utime(tj, None)                                                            # the run re-landed
    f = tmp_path / "specs" / "exp99" / "measures.jsonl"
    lines = [json.loads(l) for l in open(f)]
    lines[-1]["at"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(old))
    f.write_text("".join(json.dumps(l) + "\n" for l in lines))
    assert S.measure_all(G, 99)[("a", 1)]["m.x.v"] == 2 and len(calls) == 2
    monkeypatch.setattr(C, "CHANGED", {"m.x": time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 120))})
    assert S.measure_all(G, 99)[("a", 1)]["m.x.v"] == 3                           # the ruler changed


def test_min_over_listed_arms_is_void_when_one_arm_has_no_run():
    """exp14: a listed arm whose runs all died must not be skipped by the minimum (2026-09-27)."""
    import exp_gate_score as S
    M = {("wt", 1): {"a": 5.0}, ("ctrl", 1): {"a": 4.0}}
    v, note = S.value({"min_over_arms": {"key": "a", "arms": ["wt", "mutant"]}}, M)
    assert v is None and "mutant" in note
    assert S.value({"min_over_arms": {"key": "a", "arms": ["wt", "ctrl"]}}, M)[0] == 4.0
    assert S.value({"min_over_arms": {"key": "a"}}, M)[0] == 4.0          # no list: every run on disk, as before
