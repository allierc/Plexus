# RESUME — start here in a fresh session

**Rewritten 2026-07-31, 11:00.** Supersedes everything below the line.

## Read this first, in this order

1. `plexus2_discovery.pdf` — **the document.** Plain English, partitioned per phase. This is what
   Cedric reads and what you report into. 13 pp.
2. `plexus2_technical.pdf` — the old 23-page build spec, preserved. Reference only.
3. `SESSION_LOG.md` — my technical log. Hourly while working.

## The working agreement (agreed with Cedric 2026-07-31)

- Two tracks: **A** build the search method, **B** reproduce Okuda as the proof it works.
- **Report into `plexus2_discovery.pdf` at every phase boundary, then STOP and wait.** Do not roll
  on. This is the thing that makes it not vibe-coding.
- Plain English in that document. No jargon. Four defined terms, no more.
- Hourly `SESSION_LOG.md` while actively working.

## State on 2026-07-31 11:00

- **THE BELIEF REGISTER IS EMPTY.** Every measurement to date came through a broken ruler. The
  overnight "shaping and integrity are mutually exclusive" claim is **withdrawn** — all 32 runs
  ended at exactly 1778 cells because that is the vertex buffer ceiling, not biology.
- **THE COMPUTE CLUSTER IS OFF.** Cedric will say when it is back. Only `cluster.py` and
  `round.py` need it. Phase 0 and Phase 1 are entirely local (2x RTX A6000, 64 cores).
- **Phase 0 is in progress**, six items: 2 done (centroid, frame-pairing guard), 4 committed but
  **UNVERIFIED** at `a22975a` (Q, camera, cell counts, stopwatch), and 3 more not started
  (remove the silent parameter caps; the geometry-weighted diffusion implementation; separate the
  three mesh-failure modes).
- **Anything in `a22975a` is unverified.** Re-derive each defect before trusting it.
- Done and verified today: `time_analysis.py` (trajectory classification + evidence horizon,
  self-tested), `metrics.npz` written alongside `mechanics.npz` with 24 runs backfilled.

## The next thing

Finish Phase 0. Then Phase 1 — re-run the 32-run study at a 10x reservoir, locally. That decides
whether the withdrawn claim survives. Test whether the engine runs on `cuda` first: 10x reservoir
means ~18,000 cells and the study currently hardcodes `device="cpu"`.

---

<details><summary>Superseded — the 2026-07-30 resume, kept for reference</summary>

# RESUME — start here in a fresh session

**Written 2026-07-30, end of session.** Read this file first, then `SESSION_LOG.md` (newest hour
at the bottom). Everything is committed and pushed; nothing is running.

---

## 0. One paragraph of context

We are building an autonomous mechanism-search campaign on the Okuda tubulation problem, to run
unattended on the L4 partition for weeks. The product is **a causal lever-map of the mechanism
space** — not one answer. Specific questions ("which composition makes a tube", "does the
(χ,γ) phase diagram reproduce") are *queries against that map*. The design document is
`discovery/plexus2_discovery.pdf` (21 pp) and it is the source of truth for intent.

---

## 1. First commands

```bash
cd /workspace/Plexus/discovery
export PY=/workspace/.conda_envs/neural-graph-linux/bin/python
export PYTHONPATH=/workspace/Plexus/src

$PY validate_space.py            # must print 59/59
$PY critic.py                    # 12 type-guard rules, all self-tests
$PY cluster.py --status          # should be empty
$PY -c "import sys;sys.path.insert(0,'.');import cluster;cluster.preflight()"   # must PASS
$PY -c "import sys;sys.path.insert(0,'agents');from metrologist import Certification
print(Certification('_metrology').may_admit())"                                  # must be (True, ...)
```

If any of those fail, fix that before anything else — they are the gates.

---

## 2. THE NEXT THING TO DO

**Done 2026-07-30 (hours 12–14): the attended composition round ran.** It found eight defects,
starting with the fact that `--mode composition` could never have completed a round. Round 2 is
in the ledger. See SESSION_LOG hours 12–14 and `_turing_vertex/FINDINGS.md`.

**Next: round 3 will trigger ESCALATION on its own.** The Supervisor now resumes at round 2 with
`dry 1` and its single proximity cluster **frozen** (0 active), so `terminal()` returns
`ESCALATE: all clusters frozen` — and `round.py` now *reads* that verdict, which it never did
before. Expect it to open stage gate 3 and admit the stage-3 operators.

```bash
$PY round.py --mode composition --batch 6           # watch the escalation branch fire
```

Two open scientific threads, both from round 2, both needing a WITHIN-RUN time series rather
than a cross-run comparison (I made that mistake twice last night — see the retractions):

1. **What is `protr_peak = 4.03` on the control?** Three Analysts called it a tube; the Watcher
   vetoed it; the montage shows a small body with a short stub. Both knockouts collapse it
   (→1.39, →1.03), so both operators are necessary for *whatever it is*.
2. **Is the activator being carried off the tip?** `ta_tip_act_final` is 0.27 with growth on and
   0.80 with growth removed — but the second has no protrusion, so "tip" is not the same place.
   Confounded. Needs `ta_tip_act` over time in one run.

---

## 3. What works, verified

| | |
|---|---|
| composition space → runnable spec | 59/59 incl. parameter fidelity (V9) and clock re-anchoring (V10) |
| Critic | 12 enumerable rules; caught 3 silent no-ops including one in my own reference recipe |
| L4 driver | jobs submit, run, tracked **by job ID** (names and empty queues both lie) |
| Loop II (`--mode theta`) | ran the vcap sweep end to end |
| captioning | per wave, on the devcontainer, one model load |
| Metrologist | defects + retractions + admission gate, all exercised |

---

## 4. What is NOT done — do not assume otherwise

1. ~~The LLM Proposer has never run.~~ **Ran.** Works; wrote a real 6-slot proposal, took prompt
   corrections on the second call.
2. ~~The escalation path is unbuilt.~~ **Built** (`escalation.py`, self-tested): three-action
   decision table, `agents/llm_agents.request_operator`, wired into `round.py`, and it has one
   real entry — **OR001** in `campaign/operator_backlog.md`. Not yet fired *automatically*; round
   3 should do that.
3. ~~No progress reel.~~ **`reel.py`** builds round montages and study reels with burned-in
   labels. ⚠ Tiles are **not on a common spatial scale** (each movie used its own camera box) —
   the label says so; read shape, never relative size. A common-`Lbox` re-render is still to do,
   and so is the (χ,γ) phase-diagram panel.
4. **The Proposer call still blocks the round** — 6 GPUs idle for ~7 minutes. Should overlap N+1
   with N. **Still the main remaining inefficiency.**
5. Analyst / Watcher / Judge / Interpreter / Meta-review **have now all been invoked** on round 2.
   Evolution still has not.
6. **Analyst calls bypass the budget ledger.** `A.analyse` is called without the `ledger`, so the
   25 min per-round LLM ceiling is not enforced on the 15 analyst calls. Not yet fixed.

---

## 5. Open science

- **`D1d` — partially answered.** The clock re-anchoring restores cell count (2927 vs ~2700) but
  not the tube. The vcap sweep found: **`vcap=1.5`, the archived working point, is the WORST
  value swept**; the response is **non-monotonic** (2.19, 4.03, 1.73, 2.24, 3.22); `vcap=3.0`
  gives the best sustained protrusion (peak 3.22, final 2.81). My mechanism story was refuted.
- **R44's impossibility claim** ("emergent RD cannot sustain a tube") predates the precondition
  checks and re-enters as **Open**. Do not inherit it.
- The **(χ,γ) phase diagram** from Okuda — thin tube (0.01,100), thick tube (0.1,1), undulation
  (0.1,100), branching (0.01,0.01) — is the qualitative reproduction target and the talk figure.
  `agents/grounder.py::phase_diagram()`.

---

## 6. Hard-won rules — do not rediscover these

- **The instrument lies before the physics does.** Every error this session was a *comparison*,
  not a computation: my metric vs their metric, `ta_protr` vs `protr`, a scored tube-length vs a
  tube. Verify the instrument before trusting the measurement — including your own analysis.
- **Never swallow an exception around an artefact.** Three silent no-ops were found by removing a
  `try/except` or by enumerating a guard. A blank panel that looks deliberate is worse than a
  crash.
- **`bjobs` lies twice**: it hides finished jobs (empty ≠ done), and `-a` returns historical jobs
  (a name cannot distinguish an old EXIT from a new PEND). **Track job IDs.**
- **THE SPLIT**: local = intelligence (all LLM/VLM/PDF), cluster = jobs (engine/render/mechanics).
  The partition has no `transformers` and no `fitz`. `cluster.preflight()` enforces it.
- **A parameter CAN carry a hypothesis.** Composition identity excludes θ so a retune is not a new
  *mechanism* — but "I expect X if I raise vcap to Y" is a real, falsifiable prediction and must
  count toward the surprise rate.
- **Seeds are forced by the pipeline**; slot 0 is always the unchanged control.

---

## 7. Environment

```
$PY          /workspace/.conda_envs/neural-graph-linux/bin/python   (default python3 has no torch)
PYTHONPATH   /workspace/Plexus/src
cluster      allierc@login1, queue gpu_l4, -gpu num=1 mandatory, env `connectome-gnn`
paths        /workspace  <->  /groups/saalfeld/home/allierc/Graph   (same NFS export)
configs      config/okuda/   ·   job artefacts  log/okuda/<name>/   ·   evidence  discovery/_archive/
git          push may need --no-verify (git-lfs absent in the devcontainer)
```

---

## 8. Cadence Cedric asked for

Commit and push regularly; update `plexus2_discovery.pdf` when the design changes; append to
`SESSION_LOG.md` **every hour** with a `### ⏱ SUMMARY` block (done / found / decided / next /
blocked). Findings are numbered continuously — the last was **F20**.

</details>
