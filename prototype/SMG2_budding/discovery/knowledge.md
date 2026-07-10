# Knowledge ledger (distilled from the RunRecord archive) — round 0

> Distilled interpretation of the evidence. The archive of RunRecords is the source of truth; this ledger is revised as evidence accumulates.

## Bootstrap-ladder status  (metric_v0, frozen)
- **Rung 1 reached** — first composition robustly in the real regime across seeds + parameter basin: `cleft_induce+confine+react_rd` (turing-like (Menshykau-Iber), rate 1.00). 9 compositions are in-regime.

## Established  (sufficient ∧ robust ∧ has a necessary operator)
- **focal-ECM under confinement** — composition `cleft_induce+confine` emerges (rate 1.00); necessary operator(s): `cleft_induce`.
- **focal-ECM (Yamada)** — composition `cleft_induce+interface_relax+tissue_grow` emerges (rate 1.00); necessary operator(s): `cleft_induce`, `tissue_grow`.

## Structural limitation  (a composition that CANNOT produce the phenotype)
- `interface_relax+tissue_grow` → **no cleft operator (grows but cannot subdivide)** (class branch-like, rate 0.00).
- `confine+interface_relax` → **no growth operator (cannot develop)** (class no-growth, rate 0.00).
- `confine+tissue_grow` → **no surface-tension operator (tissue fragments, not connected)** (class cluster, rate 0.00).
- `confine+interface_relax+tissue_grow` → **no cleft operator (grows but cannot subdivide)** (class blob, rate 0.00).
- `cleft_induce+confine+interface_relax` → **no growth operator (cannot develop)** (class branch-like, rate 0.00).
- `cleft_induce+confine+interface_relax+tissue_grow` → **no-growth phenotype, not in real regime** (class no-growth, rate 0.00).
- `cleft_induce+interface_relax+react_rd+tissue_grow` → **branch-like phenotype, not in real regime** (class branch-like, rate 0.00).
- `cleft_induce+confine+interface_relax+react_rd` → **no growth operator (cannot develop)** (class branch-like, rate 0.00).
- `cleft_induce+confine+react_rd+tissue_grow` → **no surface-tension operator (tissue fragments, not connected)** (class no-growth, rate 0.00).

## Refuted  (hypothesis contradicted across seeds)
- _(none yet)_

## Open
- `cleft_induce` — partial (rate 1.00, class branch-like).
- `cleft_induce+interface_relax` — partial (rate 0.50, class branch-like).
- `cleft_induce+tissue_grow` — partial (rate 0.50, class branch-like).
- `cleft_induce+confine+tissue_grow` — partial (rate 0.50, class branch-like).
- `cleft_induce+react_rd` — partial (rate 1.00, class branch-like).
- `cleft_induce+interface_relax+react_rd` — partial (rate 0.25, class branch-like).
- `cleft_induce+react_rd+tissue_grow` — partial (rate 1.00, class branch-like).
- `cleft_induce+confine+react_rd` — partial (rate 1.00, class branch-like).
- `cleft_induce+react_rd` — partial (rate 1.00, class branch-like).
- `cleft_induce+interface_relax+react_rd` — partial (rate 0.50, class branch-like).
- `cleft_induce+react_rd+tissue_grow` — partial (rate 0.75, class branch-like).
- `cleft_induce+confine+react_rd` — partial (rate 1.00, class branch-like).

## Composition → phenotype map
| composition | region | emergence | class | topology |
| --- | --- | --- | --- | --- |
| `cleft_induce` | focal-ECM (Yamada) | 1.00 | branch-like | duct 0.931 / gen 7 |
| `cleft_induce+confine` | focal-ECM under confinement | 1.00 | branch-like | duct 0.802 / gen 7 |
| `cleft_induce+interface_relax+tissue_grow` | focal-ECM (Yamada) | 1.00 | branch-like | duct 0.886 / gen 6 |
| `cleft_induce+react_rd` | turing-like (Menshykau-Iber) | 1.00 | branch-like | duct 0.877 / gen 6 |
| `cleft_induce+react_rd+tissue_grow` | turing-like (Menshykau-Iber) | 1.00 | branch-like | duct 0.721 / gen 6 |
| `cleft_induce+confine+react_rd` | turing-like (Menshykau-Iber) | 1.00 | branch-like | duct 1.0 / gen 8 |
| `cleft_induce+react_rd` | turing-like (Menshykau-Iber) | 1.00 | branch-like | duct 0.877 / gen 6 |
| `cleft_induce+confine+react_rd` | turing-like (Menshykau-Iber) | 1.00 | branch-like | duct 1.0 / gen 8 |
| `cleft_induce+react_rd+tissue_grow` | turing-like (Menshykau-Iber) | 0.75 | branch-like | duct 0.721 / gen 6 |
| `cleft_induce+interface_relax` | focal-ECM (Yamada) | 0.50 | branch-like | duct 0.414 / gen 5 |
| `cleft_induce+tissue_grow` | focal-ECM (Yamada) | 0.50 | branch-like | duct 0.702 / gen 6 |
| `cleft_induce+confine+tissue_grow` | focal-ECM under confinement | 0.50 | branch-like | duct 0.846 / gen 8 |
| `cleft_induce+interface_relax+react_rd` | turing-like (Menshykau-Iber) | 0.50 | branch-like | duct 0.684 / gen 4 |
| `cleft_induce+interface_relax+react_rd` | turing-like (Menshykau-Iber) | 0.25 | branch-like | duct 0.684 / gen 4 |
| `interface_relax` | unnamed | 0.00 | branch-like | duct 0.867 / gen 7 |
| `tissue_grow` | unnamed | 0.00 | branch-like | duct 0.816 / gen 6 |
| `confine` | confined-growth buckling (Varner-Nelson) | 0.00 | branch-like | duct 0.899 / gen 7 |
| `interface_relax+tissue_grow` | unnamed | 0.00 | branch-like | duct 0.788 / gen 6 |
| `confine+interface_relax` | confined-growth buckling (Varner-Nelson) | 0.00 | no-growth | duct 0.516 / gen 4 |
| `confine+tissue_grow` | confined-growth buckling (Varner-Nelson) | 0.00 | cluster | duct 0.429 / gen 3 |
| `confine+interface_relax+tissue_grow` | confined-growth buckling (Varner-Nelson) | 0.00 | blob | duct 0.292 / gen 2 |
| `cleft_induce+confine+interface_relax` | focal-ECM under confinement | 0.00 | branch-like | duct 0.914 / gen 6 |
| `cleft_induce+confine+interface_relax+tissue_grow` | focal-ECM under confinement | 0.00 | no-growth | duct 0.431 / gen 3 |
| `cleft_induce+interface_relax+react_rd+tissue_grow` | turing-like (Menshykau-Iber) | 0.00 | branch-like | duct 0.782 / gen 6 |
| `cleft_induce+confine+interface_relax+react_rd` | turing-like (Menshykau-Iber) | 0.00 | branch-like | duct 0.746 / gen 7 |
| `cleft_induce+confine+react_rd+tissue_grow` | turing-like (Menshykau-Iber) | 0.00 | no-growth | duct 0.419 / gen 4 |
