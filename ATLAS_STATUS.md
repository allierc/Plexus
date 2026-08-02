# The Plexus operator atlas — programme status

*Written 2026-08-02. Read this first to resume; the per-campaign detail is in
`atlas_jax/STATUS.md` and `atlas_cc3d/STATUS.md`, and the narratives are the two
`atlas_note.pdf`s.*

---

## 1. What this is, in one paragraph

Two ways to grow the Plexus vocabulary. **Discovery** searches for a mechanism nobody wrote down.
The **atlas** takes published systems *whose code exists*, decomposes each into the Plexus operator
algebra, and measures whether that algebra is converging. The measurement is the deliverable and it
is falsifiable — `paper/plexus2.tex`, App. *"Building the Plexus operator atlas"*: convergence
toward a compact reusable vocabulary supports the algebra; repeated discovery of new operator
families means the language is incomplete.

**One repository cannot answer that.** The claim is about a curve. That is why there are now two
campaigns and a catalog, and why nothing has been promoted.

---

## 2. Where the programme stands

| | `atlas_jax` (jax-morph) | `atlas_cc3d` (CompuCell3D) |
|---|---|---|
| Phase 0 instruments + oracle | done | done |
| Phase 1 read every mechanism | done (24) | done (35) |
| Phase 2 normalize + skeptic | done | done |
| Phase 3 implement | done (16 candidates) | **deferred by policy** |
| Phase 4 differential validation | done (16/16) | deferred |
| Phase 5 forward figure | done | deferred |
| Phase 6 differentiability | done | likely N/A (Metropolis has no pathwise derivative) |
| Phase 7 promotion | **not started** | not started |

**Nothing is promoted. `src/plexus/operators/` still holds the same 52 contracts both campaigns
were scored against**, which is what makes the two measurements comparable.

### The catalog (`catalog.py` → `log/catalog.{json,png}`)

| repository | mech | scored | new | impl | alias | refine | o-o-s | yield |
|---|---|---|---|---|---|---|---|---|
| `atlas_jax` | 24 | 18 | 8 | 6 | 2 | 2 | 6 | 0.44 |
| `atlas_cc3d` | 35 | 24 | 8 | 6 | 4 | 6 | 11 | **0.33** |
| **total** | 59 | 42 | **16** | | | | | |

**The number that matters is not 16, it is 2.** The paper's promotion rule requires *"evidence
that the mechanism is reusable beyond its originating prototype"* — and only `adhere` and
`morphogen` have so far been sighted in a repository other than the one that introduced them.
`morphogen` is the strongest case: a JAX point-particle library and a C++ lattice framework
solving the same `D∇²c − kc + S = 0`, a match the normalizer found on its own.

---

## 3. The decision that shapes everything downstream

**Extract many, catalog, then promote.** Phases 1–2 only read and classify — they write a record
and nothing else. Phase 3 writes into `src/plexus/operators/candidates/`; Phase 7 promotes into the
language. Implementing CompuCell3D's candidates now would shape the anti-chamber around *one*
framework before we know what the next three need.

This is not just prudence, it is the paper's stated policy: *"Prototype freely; promote biological
mechanisms conservatively."*

**What it costs, stated plainly:** every CC3D record sits at `normalized`, and no CC3D verdict has
been checked against the reference by a differential test. **A verdict of `new` that no
differential test has checked is a claim about our reading of the source, not about the source.**
The catalog's 16 should be read with that caveat attached.

---

## 4. Next action

**Extract the next framework.** The corpus is already specified in the paper (App. *"Reference
corpus for the operator atlas"*): Chaste, Morpheus, Tissue Forge, VirtualLeaf, PhysiCell.
CompuCell3D was on that list too. They do not need choosing; they need doing.

Cost per framework: roughly one working day of wall-clock for Phases 0–2, of which **the expensive
part is standing up the oracle**, not the agent calls.

The recipe (see §5 for the pitfalls):

```bash
mkdir atlas_<target> && cp atlas_cc3d/{record,registry_view,saturation,atlas,verify_impl,record_clean,run_spec,cluster6}.py atlas_<target>/
cp -r atlas_cc3d/agents atlas_<target>/          # then RETARGET the prompts — see §5
cd atlas_<target>
python registry_view.py --json                   # freeze the SAME 52-contract baseline
# write oracle.py + inventory.py for this target  (the only genuinely new code)
python oracle.py verify                          # MUST pass before anything else
python inventory.py --write
python atlas.py phase --role excavator  --all --jobs 4
python atlas.py phase --role normalizer --all --jobs 4 --skeptic
cd .. && python catalog.py --order atlas_jax atlas_cc3d atlas_<target> --plot
```

**Also worth doing before the catalog grows:** the paper asks each atlas entry to carry
*"originating publications and reference repositories"*. Our records carry `paper_section` and
`code_path`, which is close but not the cross-framework provenance specified.

---

## 5. Pitfalls, all of them paid for once already

**Environment**

- Plexus python: `/workspace/.conda_envs/neural-graph-linux/bin/python`, `PYTHONPATH=src`.
- The CC3D oracle is its own env: `/workspace/.conda_envs/cc3d-oracle` (4.10.0, py312). **A Plexus
  process must never be able to import a reference implementation** — `oracle.py verify` checks it.
- The agent wrapper lives in `discovery*/agents/llm.py`. `discovery/` was reorganised mid-session
  into `discovery_okuda/` + `discovery_cardio_mpm/`, leaving an empty `discovery/`; the hard-coded
  path broke **both** atlases with an ImportError that looked unrelated. Both now glob for it.

**Paths** — uniform since 2026-08-02: `atlas_<t>/` ↔ `config/atlas_<t>/`, `log/atlas_<t>/`,
`graphs_data/atlas_<t>/`. `atlas_jax_morph` was renamed to `atlas_jax`.

**Forking a campaign** — five of seven instruments need *zero* edits (`record.py`,
`saturation.py`, `atlas.py`, `registry_view.py`, `agents/`). But **`agents/atlas_agents.py`'s
PREAMBLE is target-specific** and describes the previous target: retarget it or 24 agents go
looking for the wrong repository. Only `run_spec.py`'s output paths need touching.

**The driver**

- `--skeptic` belongs *during* the normalizing pass. Afterwards it re-runs the normalizer first,
  and a call that changes nothing counts as a failure — use `--role skeptic` directly.
- `R3_code_path` requires a resolvable `file:Lnn`. Dotted module names are rejected, correctly.
- Blocks are sticky. A *stale* block is its own kind of lie; clear it only with evidence and log
  the clearance (`_state/unblocked.jsonl`).

**CompuCell3D specifically** — three upstream defects, all routed around in `oracle.py`:
`PyCoreSpecs → service_cc3d` passes the specs *list* where a file is expected; CC3D `exec`s the
project script with its own globals so a steppable class must live in a separate module; the output
directory may not sit inside the project. The way through is to *generate* CC3DML from
`PyCoreSpecs` and run `cc3d/run_script.py`.

**Diagnosis** — piping an unbuffered process through `grep` buffers its output, which is lost when
the process is killed on timeout. Two healthy runs looked like hangs at import. Capture to a file.

**Citations** — I cited "plexus2.tex App. E.1" for an entire campaign; there is no E.1, and when I
corrected it I got the appendix *number* wrong twice. Cite the appendix by **name**. Read the
paper before quoting it.

---

## 6. What each campaign produced beyond the ledger

**`atlas_jax`** — the forward figure (both engines on one canvas, gyration to 9%); a
differentiable engine (`engine.run(grad=True)`, one physics and one tape); the variance question
settled (σ ∝ K^−0.73, and the apparent low-K bias was Adam's overshoot sampled at a fixed 20-step
budget, not estimator bias); and inverse design closing — asked for the rim to grow fastest, the
optimiser crossed zero and turned the activator into an inhibitor.

**`atlas_cc3d`** — six mechanisms with reference runs **and ablations** in `log/atlas_cc3d/`; the
finding that every CPM mechanism is an *energy term*, not an update (nothing writes state; they
change only the probability a pixel copy is accepted); and five architectural mechanisms no scan
could see, of which `cell_as_lattice_domain` (`occupy`) is genuinely new and the Metropolis rule,
energy-sum composition and MCS time unit all came back *out of scope* — the same verdict
jax-morph's four architectural entries got. **Two very different architectures, and both scored
zero new biological vocabulary.**
