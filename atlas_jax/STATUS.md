# Atlas — jax-morph: status

*(This directory was `atlas_jax_morph/` until 2026-08-02; it is now `atlas_jax/`, with `config/atlas_jax/`, `log/atlas_jax/` and `graphs_data/atlas_jax/` alongside.)*

*Written 2026-08-01 at the close of Phase 6; updated 2026-08-02. Programme-level view:
`../ATLAS_STATUS.md`. A second campaign (`atlas_cc3d`) now exists and the transfer question is
answered — see `TRANSFER.md` there.*

The full narrative is `atlas_note.pdf` (28 pages). This file is the operational summary: what was
done, what it cost, what is reusable, and what is not.

---

## 1. What the atlas is for

Two ways to grow the Plexus vocabulary. **Discovery** (`discovery/`) searches for a mechanism
nobody wrote down. **Atlas** (here) takes a published system *whose code exists*, decomposes it
into the Plexus operator algebra, and measures whether that algebra is converging.

The measurement is the deliverable, and it is falsifiable (plexus2.tex App. "Building the Plexus operator atlas"): if decomposing a
broad collection of frameworks converges on a compact reusable vocabulary, the algebra is a real
intermediate representation; if every framework yields new operator families, the language is
incomplete.

**Target:** `fmottes/jax-morph` @ `ace08b87` — Deshpande, Mottes et al., *Engineering morphogenesis
of cell clusters with differentiable programming*, Nat Comput Sci 2025. Apache-2.0, pure Python,
and — the point — it runs.

---

## 2. Where it stands

| phase | state | what it produced |
|---|---|---|
| 0 — instruments, frozen baseline | done | 52-contract baseline, oracle in its own interpreter, 12 validator rules, guarded driver |
| 1 — read 24 mechanisms | done | 24 excavations, 36 min, equations + surprises per mechanism |
| 2 — normalize + skeptic | done | the first ledger; 5 of 23 verdicts challenged, all 5 upheld-or-changed on evidence |
| 3 — implement | done | 16 operator modules in the anti-chamber, each with a property test |
| 4 — differential validation | done | **16/16 validated**, metric + threshold written down *before* each run |
| 5 — the forward figure | done | both engines on one canvas; morphology matches, gyration to 9% |
| 6 — differentiability | done | engine `grad=True`; variance settled; inverse design closes |
| 7 — promotion | **not started** | 16 mechanisms due at curator; deliberately deferred |

**Ledger:** 0 validator violations · 0 blocked · 16 validated · 8 normalized · **0 promoted**.

---

## 3. The measurement

24 mechanisms → **8 genuinely new contracts**, 6 further implementations of those same eight, 2
aliases, 2 refinements, 6 out of scope. **Yield 0.44 new contracts per scored mechanism.**

New: `adhere`, `agitate`, `apoptose`, `mechanosense`, `morphogen`, `regulate`, `relax`, `reorient`.

The number that matters is the *distinction*: 14 record entries argue `new`; 8 contracts are new.
Four gene-network steps with four unrelated equations are one word (`regulate`); four pair laws at
four exponents are one word (`adhere`). Counting rows instead of contracts would have inflated the
headline by 36% on the one measurement the atlas exists to make.

**One repository cannot show saturation.** The curve becomes an argument at the second and third.
That is the whole reason for the question this file is being written to answer.

---

## 4. Results worth carrying forward

- **Forward figure.** Their model in both engines: morphology matches, radius of gyration
  0.651→3.228 vs 0.686→3.518 (9%). Cell counts differ (82 vs 124) — independent random streams, so
  that compares luck; the division differ compared pooled *hazard* instead and got 1.1e-3 against
  a 5.2e-3 three-sigma bar.
- **Four code-vs-paper contradictions**, code wins in each: growth law (saturating flow, not
  constant increment + clamp), gene-network input coupling (inside the sigmoid, not added
  outside), Morse cutoff (present in code, absent from the paper), stress (Irving–Kirkwood virial,
  not the paper's taxicab sum). *None of these is visible without the code.*
- **Two mechanisms in the code and nowhere in the paper**: cell death, active Brownian motion.
- **Engine differentiability.** `engine.run(grad=True)` — one physics, one tape. Verified
  end-to-end: `d loss/d max_radius = 0.168` through 12 frames and 9 division events.
- **The variance question, settled.** Straight-through gradient through a discrete division:
  σ ∝ K^-0.73 (Monte-Carlo rate), and the apparent low-K bias was Adam's overshoot sampled at a
  fixed 20-step budget (peak at step 22–26), not estimator bias. So the trace/replay/score contract
  is an optimisation, not a prerequisite.
- **Inverse design closes.** Asked for the rim to grow fastest; gradient descent through 24 frames
  of real physics crossed zero and turned the activator into an inhibitor
  (`W_in` +0.35 → −0.253, corr(dist, growth) −0.944 → +0.959).
- **`sense → regulate → grow` loop runs**, with an ablation: cut one weight and everything still
  runs while growth collapses to a uniform 0.3034 — exactly σ(b)/γ.

---

## 5. Defects this campaign found that a forward run could not

Recorded because they are the argument for the whole apparatus.

1. **Silent zero divisions.** `cell_divide` read the division rate as a per-node buffer while the
   spec and `grow_radius` kept it as a state block. Run completed, movie looked plausible, 4 cells
   after 40 frames. Caught by the **acted ledger** asking "did this operator ever do anything?"
2. **In-place occupancy write** killed the tape at the first division. Fixed by clone-and-publish.
   Invisible to any forward run.
3. **In-place `state` write inside `cell_divide`** — `state` is itself a registered per-node
   buffer, so the mother→daughter copy loop wrote into the tensor the operator had just cloned.
   **Only surfaced in a composition** where an upstream operator held a live gradient on a slice of
   `state` across a division.
4. **`torch.full(shape, param)`** rejects a tensor fill value — six operators ran fine with floats
   and refused a learnable parameter. Not "non-differentiable"; *mechanically refusing to be fit*.
5. **Constructor coercion.** `self.x = float(params.get(...))` silently cast away the tape on every
   operator in the library. Fixed by `Operator.tunable()`.
6. **The acted ledger is not enough.** In the ablated loop run, every operator still acts and the
   ledger is unchanged while the coupling is gone. *Acted is not coupled.*

---

## 6. What is reusable, and what is not

**Target-agnostic — copy as-is:**

| file | what it is |
|---|---|
| `record.py` | schema, status ladder, the 12 rules, self-test |
| `registry_view.py` | frozen baseline of the promoted language |
| `saturation.py` | the ledger; alias / refinement / new / implementation / out_of_scope |
| `atlas.py` | guarded driver — blast radius, rung, rules, commit-or-revert |
| `agents/atlas_agents.py` | the six roles and their prompts |
| `run_spec.py` | evidence folder + **the acted ledger** |
| `verify_impl.py`, `record_clean.py` | import-all-together check, record hygiene |
| `cluster6.py` | LSF submission (reuses `discovery/cluster.py` primitives) |
| `src/plexus/operators/diff/{audit,certify}.py` | the two-route differentiability measurement |
| `render_movies.py`, `render_observables.py` | evidence rendering without re-simulating |

**Target-specific — must be rebuilt per repository:**

- `oracle.py` + `_oracle/venv` — the reference in its own interpreter, with provenance. **This is
  the load-bearing part.**
- `inventory.py` — the mechanism scan, *plus* the architectural contracts a scan cannot see (4 of
  jax-morph's 24 were added by hand and were among the most interesting).
- `paper.py` — PDF → greppable text.
- `config/atlas_jax/*.yaml` (22 specs), `_oracle/scripts/*` (differential tests), `campaign/notes/*`.

**Rough split:** ~60% of the Python is instrument and transfers; ~40% is target-specific.

---

## 7. What is deliberately not done

- **Phase 7 / promotion.** 16 mechanisms sit at `validated`, none promoted — and this is now
  programme policy rather than an open task. `plexus2.tex` requires *"evidence that the mechanism
  is reusable beyond its originating prototype"* before promotion, which is exactly what
  `../catalog.py` measures; of these 16 contracts only `adhere` and `morphogen` have so far been
  sighted in a second repository. The curator should also not live in this folder — it writes into
  `src/plexus/operators/` and belongs with the language, not with one repository's atlas.
- **The `regulate × growth` composition is not a fourth contract**, and must not be registered as
  one — that would be exactly the inflated `new` the ledger exists to prevent.
- **Figure 5 proper**: the inverse result is two parameters and a response target, not a gene
  network and a spatial objective.
- **10 operators remain `UNCONFIRMED`** in the differentiability audit — single-source, so not
  actionable.

---

## 8. Cost

Roughly one working day of wall-clock for Phases 0–4 (agent time: 36 min for 24 excavations,
73 min normalize, 56 min implement, ~107 min differential), plus the Phase 5–6 work. ~5.9k lines
of Python here, 22 specs, 26 oracle runs, 24 mechanism notes, 43 commits.

The expensive parts were **not** the agent calls. They were: standing up the oracle, and the four
harness defects that each invented a fake finding before being caught.

---

## 9. The open question this file exists for

Does the procedure transfer to a second paper + code? See `TRANSFER.md` for the prediction made
*before* running it.
