"""run_record -- the object every discovery loop passes around, and its immutable archive.

The scientific rule of this whole system is:  simulation -> RunRecord -> knowledge  (never
simulation -> knowledge). A RunRecord is the raw, reproducible evidence of one simulation; knowledge
is *distilled* from many RunRecords and lives elsewhere (a revisable ledger).

Guardrails (external review), enforced structurally here:
  1. A RunRecord's RAW FACTS are immutable after creation. A new metric never rewrites them; it is
     appended as a VERSIONED analysis -> record.analyses["metric_v2"] alongside ["metric_v1"].
  2. COMPOSITION identity (a structural hash of the graph C -- operators + typed connections, NOT
     parameter values) is separate from RUN identity (a run also fixes theta, seed, backend, IC).
     Same comp_hash can have many runs.
  3. The ARCHIVE of RunRecords is the source of truth (the evidence). The knowledge ledger is NOT
     stored here; it is a distilled interpretation that may be revised as evidence accumulates.
"""
from __future__ import annotations
import os, json, hashlib
import numpy as np

SCHEMA_VERSION = 1


def _canon(obj):
    """Round floats so hashes/ids are stable across trivially-different runs."""
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _canon(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canon(x) for x in obj]
    return obj


def comp_hash(composition) -> str:
    """STRUCTURAL hash of a composition: operators + typed connections ONLY (no parameter values, no
    seed). This is composition identity -- deliberately independent of theta/run identity."""
    canon = composition.structure() if hasattr(composition, "structure") else composition
    return "C" + hashlib.sha1(json.dumps(_canon(canon), sort_keys=True).encode()).hexdigest()[:11]


def run_id(comp_h, params, seed, backend, ic) -> str:
    key = json.dumps([comp_h, _canon(params or {}), seed, backend, ic], sort_keys=True)
    return "R" + hashlib.sha1(key.encode()).hexdigest()[:15]


class RunRecord:
    """Immutable raw facts of one simulation + an append-only, versioned analyses map."""

    _RAW = ("run_id", "parent_id", "edit", "comp_hash", "composition", "params",
            "seed", "backend", "ic", "trajectory_ref")

    def __init__(self, composition, params, seed=0, backend="phase_field", ic="real_t0",
                 parent_id=None, edit=None, trajectory_ref=None):
        ch = comp_hash(composition)
        object.__setattr__(self, "comp_hash", ch)
        object.__setattr__(self, "run_id", run_id(ch, params, seed, backend, ic))
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "edit", edit)                 # the single mutation from parent (causality)
        object.__setattr__(self, "composition",
                           composition.structure() if hasattr(composition, "structure") else composition)
        object.__setattr__(self, "params", dict(params or {}))
        object.__setattr__(self, "seed", int(seed))
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "ic", ic)
        object.__setattr__(self, "trajectory_ref", trajectory_ref)
        object.__setattr__(self, "_analyses", {})              # metric_version -> analysis dict (append-only)
        object.__setattr__(self, "_frozen", True)

    # --- immutability of raw facts -------------------------------------------------
    def __setattr__(self, k, v):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"RunRecord is immutable; raw fact '{k}' cannot be changed. "
                                 f"Use add_analysis() to append a versioned measurement.")
        object.__setattr__(self, k, v)

    # --- append-only, versioned analyses -------------------------------------------
    def add_analysis(self, metric_version: str, result: dict):
        """Attach a measurement computed under `metric_version`. Never overwrites an existing version
        (raw facts + prior analyses are permanent); returns self for chaining."""
        if metric_version in self._analyses:
            raise ValueError(f"analysis '{metric_version}' already recorded for {self.run_id}; "
                             f"analyses are append-only")
        self._analyses[metric_version] = dict(result)
        return self

    @property
    def analyses(self) -> dict:
        return dict(self._analyses)                            # read-only view

    def analysis(self, metric_version: str) -> dict | None:
        return self._analyses.get(metric_version)

    def verdict(self, metric_version: str | None = None):
        a = self._analyses.get(metric_version) if metric_version else \
            (next(iter(reversed(self._analyses.values())), None))
        return (a or {}).get("verdict")

    # --- (de)serialise -------------------------------------------------------------
    def to_dict(self):
        d = {k: getattr(self, k) for k in self._RAW}
        d["schema"] = SCHEMA_VERSION
        return d

    @classmethod
    def _from_raw(cls, d):
        obj = cls.__new__(cls)
        for k in cls._RAW:
            object.__setattr__(obj, k, d.get(k))
        object.__setattr__(obj, "_analyses", {})
        object.__setattr__(obj, "_frozen", True)
        return obj

    def __repr__(self):
        return f"RunRecord({self.run_id} comp={self.comp_hash} seed={self.seed} edit={self.edit})"


class RunArchive:
    """Append-only, on-disk store of RunRecords = the SOURCE OF TRUTH. Raw facts are written once per
    run_id and never rewritten; versioned analyses and trajectories are appended. (The knowledge
    ledger is derived from this and lives separately.)"""

    def __init__(self, root):
        self.root = root
        self.facts = os.path.join(root, "records.jsonl")       # one line per run_id (write-once)
        self.ana = os.path.join(root, "analyses.jsonl")        # append-only versioned analyses
        self.traj = os.path.join(root, "trajectories")
        os.makedirs(self.traj, exist_ok=True)
        self._seen = set()
        if os.path.exists(self.facts):
            for l in open(self.facts):
                if l.strip():
                    self._seen.add(json.loads(l)["run_id"])

    def save_trajectory(self, run_id_, arr) -> str:
        ref = os.path.join(self.traj, f"{run_id_}.npy")
        if not os.path.exists(ref):                            # write-once
            np.save(ref, np.asarray(arr, np.float32))
        return ref

    def add(self, record: RunRecord):
        """Persist raw facts (idempotent: never rewrites an existing run) + any attached analyses."""
        if record.run_id not in self._seen:
            with open(self.facts, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
            self._seen.add(record.run_id)
        for mv, res in record.analyses.items():
            self.add_analysis(record.run_id, mv, res)
        return record.run_id

    def add_analysis(self, run_id_, metric_version, result):
        with open(self.ana, "a") as f:
            f.write(json.dumps({"run_id": run_id_, "metric_version": metric_version,
                                "result": _canon(result)}) + "\n")

    def all(self):
        """Reconstruct records (facts + merged analyses; first write of a (run,version) wins)."""
        recs = {}
        if os.path.exists(self.facts):
            for l in open(self.facts):
                if l.strip():
                    d = json.loads(l); recs[d["run_id"]] = RunRecord._from_raw(d)
        if os.path.exists(self.ana):
            for l in open(self.ana):
                if not l.strip():
                    continue
                e = json.loads(l); r = recs.get(e["run_id"])
                if r is not None and e["metric_version"] not in r._analyses:
                    r._analyses[e["metric_version"]] = e["result"]
        return list(recs.values())

    def load_trajectory(self, record: RunRecord):
        return np.load(record.trajectory_ref) if record.trajectory_ref and \
            os.path.exists(record.trajectory_ref) else None


if __name__ == "__main__":                                     # smoke test of the guardrails
    demo = {"operators": [{"op": "interface_relax"}, {"op": "tissue_grow"}], "connections": []}
    r = RunRecord(demo, params={"kappa": 1.2}, seed=0)
    r.add_analysis("metric_v1", {"duct": 0.9, "verdict": "Open"})
    r.add_analysis("metric_v2", {"duct": 0.4, "verdict": "Refuted"})     # metric changed -> new version
    assert r.analysis("metric_v1")["duct"] == 0.9                        # v1 raw fact untouched
    try:
        r.seed = 5; assert False
    except AttributeError:
        pass                                                            # raw facts immutable
    try:
        r.add_analysis("metric_v1", {})                                 # no overwrite
        assert False
    except ValueError:
        pass
    print("run_id", r.run_id, "comp_hash", r.comp_hash, "analyses", list(r.analyses))
    print("guardrails OK")
