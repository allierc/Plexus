# SMG phase-field loop (`pf/`) — the re-based, literature-grounded forward model

This replaces the original sparse-agent SMG loop (`../search/`), whose substrate was **wrong for the
biology** and could not be fixed by tuning the reward.

## Why the old loop was out of scope

The real SMG (see `_real/real_targets.png`) is a **dense, fully-connected lobular epithelium** that
branches by **cleft formation**: a solid bud is subdivided by narrow indentations into daughter lobules
while the whole tissue stays one connected mass (real data: area grows only ~33 % while lobule count
~doubles → subdivision-dominated). The old substrate was **600 self-propelled active-matter agents**;
they can only migrate apart into thin **fragments** (`../search/_bootstrap/montage_specs.png`). The
tightened readout confirmed it: re-scoring every old "duct=1.0" winner gave `duct≈0.05 / fragment`,
0/28 survived. No mechanism search over a fragmenting substrate can produce a dense clefting gland.

## Foundation (literature + repos)

Grounded in `../../papers/organs_genesis_review.pdf` (Andrews & Priya 2024) + `Tissue_active_matter.pdf`
(Brückner & Hannezo 2025) and the cloned repos:

- **Mechanism** = focal-ECM cleft-subdivision of a solid bud (Yamada lab: Harunaga 2011/2014, Wang 2021):
  fibronectin deposited at surface indentations pins/deepens clefts that partition the bud.
- **Forward-model class** = **phase-field on a grid + light mechanics** — the only representation that is
  a *dense connected domain by construction*. Mirrors **Chaste immersed-boundary** (growth = a local
  source term; tissue = a field with interfacial-tension mechanics) and **SimuCell3D** (cortical
  tension vs adhesion, growth via evolving target volume). *Where* clefts form is the searchable
  hypothesis, per **reaction-diffusion / CompuCell3D** (Turing morphogen sets lobule spacing).

## The model — `pf_sim.py`

Dense continuum, connected by construction:
- `phi(x,t) ∈ [0,1]` tissue indicator; Allen–Cahn surface tension keeps it smooth + connected and sets
  lobule size (`kappa`).
- **Volume-controlled growth** (Chaste-IB): a global pressure pulls area toward a slowly-growing target
  (`beta`, `growth_frac`) — the tissue is maintained + gently grown, never curvature-annihilated.
- **Self-limiting clefts**: a fibronectin/ECM field `F` pinches the boundary inward but only advances
  while local tissue is thick and only nucleates in a surface band (`thick_gate`, `thick_hi`) — so
  clefts stop at duct width and **lobules stay connected** (no fragmentation, no interior holes).
- **Cleft position = the hypothesis** (`cleft_mode`): `curvature` (focal-ECM positive feedback) or
  `turing` (Gray-Scott prepattern → regular lobule spacing).

Validated in `_real/forward_model_validation.png`: from the real t=0 shape the model produces dense,
connected, solid lobes that cleft-subdivide, matching the real gland and reading branch-like / low
cluster under the honest readout. ~0.8 s/spec on one GPU (≈50× faster than the MPM substrate).

## The loop

- `pf_tree.py` — 4 hypothesis branches sharing the substrate: `focal_ecm` (Yamada),
  `turing_prepattern` (Menshykau–Iber), `differential_adhesion` (Steinberg/Wang–Yamada),
  `confined_growth` (Varner–Nelson). Branch fixes cleft mode + param ranges; `encode` → surrogate features.
- `pf_bootstrap.py` — runs the forward model over the tree, scores each spec with the **tightened**
  readout (`../search/smg_reward.py`: sigma 2.5 + skeleton tissue-support; calibration gate passes,
  incl. real-gland-shattered-against-itself control) + **`target_distance`** to the real morphology
  vector. 120 specs / 1.6 min: **82 branch-like, 38 no-growth, 0 fragments/clusters**; best
  `turing_prepattern` td=0.012 (gen 9 = real). Dataset → `_boot/dataset.jsonl` + `encodings.npy`.
- `pf_montage_winners.py` — best spec per hypothesis vs the real target row (`_boot/winners.png`).
- `pf_explore.py` / `real_targets.py` — regime sweep + real target rasterization.

## Next

Surrogate (predict the morphology vector from `encodings.npy`) + **UCB over the four hypothesis
branches** to decide which mechanism deserves compute, minimizing `target_distance` to real. Then wrap
`pf_sim` as Plexus grid operators (`pf_diffuse`/`pf_growth`/`pf_cleft` on a `frame: grid` field — see the
`embryo_gray_scott` operator template) so the loop runs inside the engine.
