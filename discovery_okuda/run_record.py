"""run_record -- the evidence object of the Okuda discovery campaign, and its immutable archive.

The scientific rule of the whole system, enforced structurally here:

    simulation  ->  RunRecord  ->  knowledge          (never simulation -> knowledge)

A RunRecord is the raw, reproducible evidence of ONE simulation. Knowledge is *distilled* from
many RunRecords and lives elsewhere (a revisable ledger).

Ported from prototype/SMG2_budding/discovery/run_record.py with four campaign-critical fixes
(see discovery/plexus2_discovery.pdf Sec. "Pre-flight"):

  D7  FULL-TRAJECTORY PERSISTENCE.  The SMG2 archive stored only `traj[-1]`, so every re-scoring
      was restricted to end-state observables.  Every decisive observable in the Okuda record is
      TEMPORAL (a protrusion that peaks mid-run then degrades; a pattern that floods only once
      growth starts; divisions arriving in synchronised waves) -- and the recording-stride
      artefact that cost us days is by construction a time-series defect.  We therefore persist
      the whole recorded trajectory AND a per-frame metric table, so a new observable can
      re-score the archive without re-simulating.

  D4  ACTED-LEDGER.  Every scheduled operator reports whether it actually did anything.  A run in
      which a scheduled operator never acted is an ERROR, not a result -- otherwise the campaign
      records "this mechanism cannot make tubes" when the mechanism never ran.

  D8  EARNED IMPOSSIBILITY.  `Claim` carries the four things an impossibility verdict needs:
      all-acted, a minimum sample count with an interval, a recorded falsification attempt, and
      an explicit scope.  Absent any of the four the verdict is Open.

  D9  INTERVALS, NOT POINT RATES.  `wilson()` -- a 3/4 point estimate is not evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np

SCHEMA_VERSION = 2


# --------------------------------------------------------------------------- canonicalisation
def _canon(obj):
    """Round floats so hashes / ids are stable across trivially-different runs."""
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, np.floating):
        return round(float(obj), 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _canon(obj.tolist())
    if isinstance(obj, dict):
        return {k: _canon(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canon(x) for x in obj]
    return obj


def comp_hash(composition) -> str:
    """STRUCTURAL hash of a composition: operators + typed connections ONLY.

    No parameter values, no seed.  This is the line that makes the campaign honest: a change of
    numbers PROVABLY cannot register as a new hypothesis.  Retuning theta is a different run of
    the SAME composition and is Loop II's business.
    """
    canon = composition.structure() if hasattr(composition, "structure") else composition
    return "C" + hashlib.sha1(json.dumps(_canon(canon), sort_keys=True).encode()).hexdigest()[:11]


def run_id(comp_h, params, seed, backend, ic) -> str:
    key = json.dumps([comp_h, _canon(params or {}), seed, backend, ic], sort_keys=True)
    return "R" + hashlib.sha1(key.encode()).hexdigest()[:15]


# --------------------------------------------------------------------------- statistics (D9)
def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for k successes in n trials.

    Returns (lo, hi).  Used everywhere a "rate" would otherwise be reported as a bare fraction.
    With n=4 the interval is enormous -- which is the point: it makes the thinness of the
    evidence visible instead of hiding it behind `rate = 0.75`.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- the evidence object
class RunRecord:
    """Immutable raw facts of one simulation + an append-only, versioned analyses map."""

    _RAW = ("run_id", "parent_id", "edit", "comp_hash", "composition", "params",
            "seed", "backend", "ic", "trajectory_ref", "acted", "wall_s", "campaign")

    def __init__(self, composition, params, seed=0, backend="tyssue_avm_3d", ic="vesicle",
                 parent_id=None, edit=None, trajectory_ref=None, acted=None, wall_s=None,
                 campaign=None):
        ch = comp_hash(composition)
        o = object.__setattr__
        o(self, "comp_hash", ch)
        o(self, "run_id", run_id(ch, params, seed, backend, ic))
        o(self, "parent_id", parent_id)
        o(self, "edit", edit)                       # the SINGLE mutation from parent (causality)
        o(self, "composition",
          composition.structure() if hasattr(composition, "structure") else composition)
        o(self, "params", dict(params or {}))
        o(self, "seed", int(seed))
        o(self, "backend", backend)
        o(self, "ic", ic)
        o(self, "trajectory_ref", trajectory_ref)
        # D4: {op_name: n_times_it_actually_did_something}. None == not instrumented (a defect).
        o(self, "acted", dict(acted) if acted else None)
        o(self, "wall_s", wall_s)
        o(self, "campaign", campaign)
        o(self, "_analyses", {})
        o(self, "_frozen", True)

    # --- immutability of raw facts ----------------------------------------------------------
    def __setattr__(self, k, v):
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"RunRecord is immutable; raw fact {k!r} cannot be changed. "
                f"Use add_analysis() to append a versioned measurement.")
        object.__setattr__(self, k, v)

    def set_trajectory_ref(self, ref):
        """Storage pointer, settable exactly once (known only after the archive writes)."""
        if self.trajectory_ref is not None:
            raise ValueError(f"trajectory_ref already set for {self.run_id}")
        object.__setattr__(self, "trajectory_ref", ref)
        return self

    def set_acted(self, acted: dict):
        """The acted-ledger, settable exactly once (known only after the run)."""
        if self.acted is not None:
            raise ValueError(f"acted already set for {self.run_id}")
        object.__setattr__(self, "acted", dict(acted))
        return self

    # --- D4: a scheduled operator that never acted invalidates the run ------------------------
    def inert_operators(self):
        """Scheduled operators that never did anything. Non-empty => the run is NOT evidence.

        This is the guard against the most dangerous bug in an automatic search: a silently
        no-op operator still returns metrics, so the loop records 'this mechanism cannot make
        tubes' when the mechanism never ran.
        """
        if self.acted is None:
            return None                                     # not instrumented -- treat as unknown
        scheduled = [o["op"] for o in (self.composition or {}).get("operators", [])]
        return sorted({op for op in scheduled if self.acted.get(op, 0) == 0})

    @property
    def is_valid_evidence(self) -> bool:
        inert = self.inert_operators()
        return inert is not None and len(inert) == 0

    # --- append-only, versioned analyses -----------------------------------------------------
    def add_analysis(self, metric_version: str, result: dict):
        if metric_version in self._analyses:
            raise ValueError(f"analysis {metric_version!r} already recorded for {self.run_id}; "
                             f"analyses are append-only")
        self._analyses[metric_version] = dict(result)
        return self

    @property
    def analyses(self) -> dict:
        return dict(self._analyses)

    def analysis(self, metric_version: str):
        return self._analyses.get(metric_version)

    # --- (de)serialise -----------------------------------------------------------------------
    def to_dict(self):
        d = {k: getattr(self, k) for k in self._RAW}
        d["schema"] = SCHEMA_VERSION
        return _canon(d)

    @classmethod
    def _from_raw(cls, d):
        obj = cls.__new__(cls)
        for k in cls._RAW:
            object.__setattr__(obj, k, d.get(k))
        object.__setattr__(obj, "_analyses", {})
        object.__setattr__(obj, "_frozen", True)
        return obj

    def __repr__(self):
        return (f"RunRecord({self.run_id} comp={self.comp_hash} seed={self.seed} "
                f"edit={self.edit} valid={self.is_valid_evidence})")


# --------------------------------------------------------------------------- the archive
class RunArchive:
    """Append-only on-disk store = the SOURCE OF TRUTH.

    Layout:
        records.jsonl      one line per run_id (write-once raw facts)
        analyses.jsonl     append-only versioned analyses
        traj/<run_id>.npz  D7: the FULL recorded trajectory + per-frame metric table
    """

    def __init__(self, root):
        self.root = root
        self.facts = os.path.join(root, "records.jsonl")
        self.ana = os.path.join(root, "analyses.jsonl")
        self.traj = os.path.join(root, "traj")
        os.makedirs(self.traj, exist_ok=True)
        self._seen = set()
        # ONE BAD LINE MUST NOT KILL EVERY JOB. This was `json.loads(line)["run_id"]` inside the loop,
        # and when eleven rows of a DIFFERENT schema were appended to this file -- the campaign's round
        # records, by a reset that archived them to the wrong place -- every run in the next round died
        # with KeyError: 'run_id'. It died AFTER several minutes of simulation, because the archive is
        # opened at the write step, so eight of eleven jobs burned their GPU time before failing. A
        # reader that crashes on a line it does not recognise takes the whole campaign with it.
        skipped = 0
        if os.path.exists(self.facts):
            for line in open(self.facts):
                if not line.strip():
                    continue
                try:
                    rid = json.loads(line).get("run_id")
                except Exception:
                    skipped += 1
                    continue
                if rid is None:
                    skipped += 1
                    continue
                self._seen.add(rid)
            if skipped:
                print(f"[archive] skipped {skipped} line(s) in {self.facts} with no run_id -- "
                      f"another writer is using this file")

    # --- D7: persist the WHOLE trajectory, not traj[-1] ---------------------------------------
    def save_trajectory(self, run_id_: str, frames, frame_metrics=None, meta=None) -> str:
        """Write-once. `frames` is the full recorded sequence; `frame_metrics` is a per-frame
        table {name: [v_0 .. v_T]}. Both are needed for any temporal re-scoring.

        Frames may be ragged (cell count grows), so we store them as an object array plus an
        explicit index rather than forcing a rectangular tensor.
        """
        ref = os.path.join(self.traj, f"{run_id_}.npz")
        if os.path.exists(ref):
            return ref
        payload = {}
        arrs = [np.asarray(f, np.float32) for f in frames]
        payload["n_frames"] = np.int64(len(arrs))
        for i, a in enumerate(arrs):
            payload[f"f{i}"] = a
        for k, v in (frame_metrics or {}).items():
            payload[f"m_{k}"] = np.asarray(v, np.float32)
        if meta:
            payload["meta_json"] = np.array(json.dumps(_canon(meta)))
        np.savez_compressed(ref, **payload)
        return ref

    def load_trajectory(self, record: RunRecord):
        """Return (frames, frame_metrics, meta) -- the whole time course."""
        ref = record.trajectory_ref
        if not ref or not os.path.exists(ref):
            return None, {}, {}
        z = np.load(ref, allow_pickle=False)
        n = int(z["n_frames"]) if "n_frames" in z else 0
        frames = [z[f"f{i}"] for i in range(n)]
        fm = {k[2:]: z[k] for k in z.files if k.startswith("m_")}
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        return frames, fm, meta

    def final_frame(self, record: RunRecord):
        frames, _, _ = self.load_trajectory(record)
        return frames[-1] if frames else None

    # --- records -------------------------------------------------------------------------------
    def add(self, record: RunRecord):
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
                                "result": _canon(result), "t": time.time()}) + "\n")

    def all(self):
        recs = {}
        if os.path.exists(self.facts):
            for line in open(self.facts):
                if line.strip():
                    d = json.loads(line)
                    recs[d["run_id"]] = RunRecord._from_raw(d)
        if os.path.exists(self.ana):
            for line in open(self.ana):
                if not line.strip():
                    continue
                e = json.loads(line)
                r = recs.get(e["run_id"])
                if r is not None and e["metric_version"] not in r._analyses:
                    r._analyses[e["metric_version"]] = e["result"]
        return list(recs.values())

    def by_comp(self):
        """{comp_hash: [RunRecord, ...]} -- evidence grouped by hypothesis."""
        out = {}
        for r in self.all():
            out.setdefault(r.comp_hash, []).append(r)
        return out


# --------------------------------------------------------------------------- D8: earned claims
VERDICTS = ("Established", "Refuted", "Structural", "Open")


class Claim:
    """A ledger entry. The point of this class is that `Structural` (an impossibility claim --
    the strongest thing the campaign can say) CANNOT be constructed without its four warrants.

    The SMG2 implementation produced impossibility verdicts from a low score plus an if/elif
    chain that guessed a reason from which operators were absent. That is a template, not
    evidence, and combined with the silent-no-op bug it is machinery for manufacturing
    confident false impossibility. Here the constructor refuses.
    """

    MIN_SAMPLES = 12                 # campaign parameter; recorded on every claim

    def __init__(self, comp_h, verdict, *, k, n, all_acted, falsification=None, scope=None,
                 reason="", evidence=(), thresholds=None):
        if verdict not in VERDICTS:
            raise ValueError(f"verdict {verdict!r} not in {VERDICTS}")
        lo, hi = wilson(k, n)
        if verdict == "Structural":
            missing = []
            if not all_acted:
                missing.append("all_acted (some scheduled operator never acted)")
            if n < self.MIN_SAMPLES:
                missing.append(f"n>={self.MIN_SAMPLES} (have {n})")
            if not falsification:
                missing.append("a recorded falsification attempt")
            if not scope:
                missing.append("an explicit scope")
            if missing:
                raise ValueError(
                    "Structural (impossibility) claim refused -- missing: " + "; ".join(missing) +
                    ". Demote to Open.")
        self.comp_hash = comp_h
        self.verdict = verdict
        self.k, self.n = int(k), int(n)
        self.rate = (k / n) if n else 0.0
        self.ci = (round(lo, 3), round(hi, 3))
        self.all_acted = bool(all_acted)
        self.falsification = falsification
        self.scope = scope
        self.reason = reason
        self.evidence = list(evidence)
        self.thresholds = dict(thresholds or {})

    def to_dict(self):
        return _canon({k: v for k, v in self.__dict__.items()})

    def __repr__(self):
        return (f"Claim({self.verdict} {self.comp_hash} {self.k}/{self.n} "
                f"CI{self.ci} acted={self.all_acted})")


# --------------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    demo = {"operators": [{"id": "a", "op": "cell_mechanics"}, {"id": "b", "op": "cell_chem_diffuse"}],
            "connections": []}
    r = RunRecord(demo, params={"kappa": 1.2}, seed=0)
    r.set_acted({"cell_mechanics": 350, "cell_chem_diffuse": 0})          # cell_chem_diffuse never acted
    assert r.inert_operators() == ["cell_chem_diffuse"]
    assert not r.is_valid_evidence, "a run with an inert operator is not evidence"

    r2 = RunRecord(demo, params={"kappa": 1.2}, seed=1)
    r2.set_acted({"cell_mechanics": 350, "cell_chem_diffuse": 350})
    assert r2.is_valid_evidence

    r2.add_analysis("metric_v1", {"aspect": 7.5, "Q": 0.05})
    r2.add_analysis("metric_v2", {"aspect": 7.5, "Q": 0.05, "p_ratio": 3.1})
    try:
        r2.add_analysis("metric_v1", {}); assert False
    except ValueError:
        pass
    try:
        r2.seed = 5; assert False
    except AttributeError:
        pass

    lo, hi = wilson(3, 4)
    assert hi - lo > 0.5, "4 samples must produce a visibly enormous interval"
    print(f"wilson(3,4) = [{lo:.2f}, {hi:.2f}]   <- why a 3/4 'rate' is not evidence")
    print(f"wilson(9,12) = {tuple(round(x,2) for x in wilson(9,12))}")

    try:
        Claim("Cxxx", "Structural", k=0, n=4, all_acted=True)
        assert False, "impossibility must be refused without its warrants"
    except ValueError as e:
        print("refused as designed:", str(e)[:96], "...")

    c = Claim("Cxxx", "Structural", k=0, n=16, all_acted=True,
              falsification={"searched": 40, "best_aspect": 1.02},
              scope={"substrate": "mid-surface AVM", "seeds": 16},
              reason="no composition without an extrusion node exceeded aspect 1.1")
    print("earned:", c)

    # D7 round-trip: the whole time course survives
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        A = RunArchive(d)
        frames = [np.random.rand(10 + i, 3) for i in range(6)]         # ragged: cells grow
        ref = A.save_trajectory(r2.run_id, frames, {"aspect": [1, 2, 3, 4, 5, 6]})
        r2.set_trajectory_ref(ref); A.add(r2)
        back = A.all()[0]
        f, fm, _ = A.load_trajectory(back)
        assert len(f) == 6 and f[3].shape == (13, 3)
        assert list(fm["aspect"]) == [1, 2, 3, 4, 5, 6]
        print(f"D7 OK: {len(f)} frames + per-frame metrics recovered (SMG2 stored 1)")
    print("\nrun_record guardrails OK")
