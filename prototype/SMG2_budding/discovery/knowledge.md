# Knowledge ledger (distilled from the RunRecord archive) — round 0

> Distilled interpretation of the evidence. The archive of RunRecords is the source of truth; this ledger is revised as evidence accumulates.

## Bootstrap-ladder status  (metric_v0, frozen)
- **Rung 1 reached** — first composition robustly in the real regime across seeds + parameter basin: `cleft_induce+interface_relax+tissue_grow` (focal-ECM (Yamada), rate 1.00). 7 compositions are in-regime.
- **Named failure → Loop III (measurement discovery):** metric_v0 cannot separate the 7 in-regime compositions, nor prove any operator *necessary* — the initial condition is the real t=0 gland (already branch-like), so the topology readout saturates and cannot measure developmental subdivision. A new observable (subdivision / cleft-spacing over time) is required.

## Established  (sufficient ∧ robust ∧ has a necessary operator)
- _(none yet)_

## Structural limitation  (a composition that CANNOT produce the phenotype)
- `confine+interface_relax` → **no growth operator (cannot develop)** (class no-growth, rate 0.00).
- `confine+tissue_grow` → **no surface-tension operator (tissue fragments, not connected)** (class cluster, rate 0.00).
- `confine+interface_relax+tissue_grow` → **no cleft operator (grows but cannot subdivide)** (class blob, rate 0.00).

## Refuted  (hypothesis contradicted across seeds)
- _(none yet)_

## Open
- `interface_relax` — partial (rate 1.00, class branch-like).
- `tissue_grow` — partial (rate 1.00, class branch-like).
- `cleft_induce` — partial (rate 1.00, class branch-like).
- `confine` — partial (rate 1.00, class branch-like).
- `interface_relax+tissue_grow` — partial (rate 1.00, class branch-like).
- `cleft_induce+interface_relax` — partial (rate 0.50, class branch-like).
- `cleft_induce+tissue_grow` — partial (rate 0.50, class branch-like).
- `cleft_induce+confine` — partial (rate 1.00, class branch-like).
- `cleft_induce+interface_relax+tissue_grow` — partial (rate 1.00, class branch-like).

## Composition → phenotype map
| composition | region | emergence | class | topology |
| --- | --- | --- | --- | --- |
| `interface_relax` | unnamed | 1.00 | branch-like | duct 0.867 / gen 7 |
| `tissue_grow` | unnamed | 1.00 | branch-like | duct 0.816 / gen 6 |
| `cleft_induce` | focal-ECM (Yamada) | 1.00 | branch-like | duct 0.931 / gen 7 |
| `confine` | confined-growth buckling (Varner-Nelson) | 1.00 | branch-like | duct 0.899 / gen 7 |
| `interface_relax+tissue_grow` | unnamed | 1.00 | branch-like | duct 0.788 / gen 6 |
| `cleft_induce+confine` | focal-ECM (Yamada) | 1.00 | branch-like | duct 0.802 / gen 7 |
| `cleft_induce+interface_relax+tissue_grow` | focal-ECM (Yamada) | 1.00 | branch-like | duct 0.886 / gen 6 |
| `cleft_induce+interface_relax` | focal-ECM (Yamada) | 0.50 | branch-like | duct 0.414 / gen 5 |
| `cleft_induce+tissue_grow` | focal-ECM (Yamada) | 0.50 | branch-like | duct 0.702 / gen 6 |
| `confine+interface_relax` | confined-growth buckling (Varner-Nelson) | 0.00 | no-growth | duct 0.516 / gen 4 |
| `confine+tissue_grow` | confined-growth buckling (Varner-Nelson) | 0.00 | cluster | duct 0.429 / gen 3 |
| `confine+interface_relax+tissue_grow` | confined-growth buckling (Varner-Nelson) | 0.00 | blob | duct 0.292 / gen 2 |
