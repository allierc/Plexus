# Campaign 2026-08-02 — nine rounds, the run where the logic did not emerge

Clean start 13:02, resume 17:32, clean stop 20:35. Rounds 1–5 and 7–10 recorded
(round 6 was a recon round deliberately killed). **79 posed, 58 admitted, 21 refused,
66 simulations.** Predictions: **30 refuted, 23 confirmed, 5 inconclusive.**

Ten defects fixed while it ran, across 26 commits — chief among them the reading loop
recording ONE chimeric run per round, and a ranking key that read `premises_broken` from
the wrong dict so invalid specimens became the frontier for five rounds.

## The conclusion

The apparatus is sound; the epistemics are not. Across `records/memory.md` and
`records/analysis.md` there is **not one** hedge (*only tested once*, *may not generalise*,
*confounder*, *insufficient evidence*, *cannot conclude*) against **fifteen** assertions of
closure (6 INERT, 3 NECESSARY, 2 FUNDAMENTAL, 1 EXHAUSTED, 3 prohibitions).

**The logic did not emerge.** The agents were exactly as rigorous as the structure they were
given: positives carry falsifiers because the template has a `Falsifiable by:` field; negatives
carry nothing because it has no field for their conditions and no bucket for the untested.

## The six specimens, and why each is here

| specimen | what it shows |
|---|---|
| `bud-with-fission-neck-VALID` (r008c_02) | a clean bud with a constricting neck **after `divide_3d` was removed** — refuted "division is NECESSARY", a claim still asserted in memory.md two rounds later |
| `best-valid-1.295-2tubes` (r007c_02) | the campaign ceiling on a sound specimen. Set in round 3, never exceeded in seven later rounds |
| `multilobe-coarse-spots-VALID` (r009c_01) | a few coarse chemical domains → a multi-lobed body. Recorded as `morphology=unclear, tubes=1` — the metrics undersell it |
| `finest-Turing-pattern-but-PERFECT-SPHERE` (r009c_05) | the best pattern in the campaign and **zero** deformation. Recorded as a null (`sphere`, 1.003). The instrument cannot see the pattern, so it reported absence of effect |
| `INVALID-1.529-30tubes-dead-chemistry` (r005c_04) | the highest numbers ever measured, and worthless: activator blown to 1.4e6, gone negative, extinct; sheet folded five layers through itself |
| `RESERVOIR-CAPPED-36749cells` (r001n_08) | growth stopped by an array, not by biology — a censored measurement, i.e. a lower bound |

Read the first, fourth and third together: **pattern localisation controls budding.** A fine
pattern spreads growth evenly and keeps a sphere spherical; one localised patch makes a bud.
Nothing in the loop measures wavelength, domain count or contrast, so the governing variable was
not merely unmeasured — it was unrepresentable.

## Contents

- `records/` — every campaign `.md` and `.jsonl`, the frontier, hypotheses, round records, terminal log
- `specimens/` — six runs: `movie.mp4`, `strip.png`, `metrics.png`, `diag.json`, `spec_run.yaml`, composition
