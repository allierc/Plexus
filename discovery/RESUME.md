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

**Run one attended composition round.** The LLM Proposer has been written, parses, and is wired
into `round.py` — but **it has never actually been called**. Every prompt-level failure mode
(bad JSON, the agent ignoring the control slot, Reflection rejecting everything) is still
unobserved.

```bash
$PY round.py --mode composition --batch 6 --dry     # inspect the proposal FIRST
$PY round.py --mode composition --batch 6           # then for real, and watch it
```

Read `campaign/proposal.json`, `campaign/analysis.md`, `campaign/knowledge.md` afterwards.
Expect it to break. That is the point of doing it attended.

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

1. **The LLM Proposer has never run.** (See §2.)
2. **The escalation path is unbuilt.** `Supervisor.escalate()` opens a stage gate or files an
   operator request, but nothing consumes an operator request. The lever-map reframing made this
   the *main* path for a multi-week campaign, not an edge case. **This is the biggest gap.**
3. **No progress reel.** Per-run movie/strip/caption/mechanics exist; nothing assembles the round
   montage, the progress reel, or the (χ,γ) phase-diagram panel that Cedric asked for.
4. **The Proposer call blocks the round** — 8 GPUs idle for minutes. Should overlap N+1 with N.
5. Analyst/Interpreter/Meta-review/Evolution/Judge are written and parse but have **never been
   invoked** either.

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
