# embryogenesis3 — Turing / Gray-Scott reaction-diffusion morphogenesis, in Plexus

A strict-Plexus reproduction of the reaction-diffusion morphogenesis model — **A. M. Turing,
*The Chemical Basis of Morphogenesis* (Phil. Trans. R. Soc. B, 1952)**, in the two-species
**Gray-Scott** form of **J. E. Pearson, *Complex Patterns in a Simple System* (Science, 1993)**.
Reference implementation vendored at [`papers/reaction-diffusion/`](../../papers/reaction-diffusion/)
(B. Maier, MIT). Morphogenesis from **pure local chemistry** — no cells, no mechanics, no
learned rule: a homogeneous field spontaneously develops a stationary/dynamic **pattern**.

## The idea

Two morphogens — substrate **A** and autocatalyst **B** — diffuse and react by
`A + 2B → 3B`:

```
∂A/∂t = D_A ∇²A − A·B²  + f·(1 − A)          A fed in at rate f, consumed by the reaction
∂B/∂t = D_B ∇²B + A·B²  − (f + k)·B           B produced by the reaction, removed at rate f+k
```

With **differential diffusion** (D_A > D_B) this is a **Turing instability**: the uniform state
destabilises into a pattern whose *class* — spots, stripes, mazes, self-replicating "mitosis"
spots, moving solitons — is set entirely by the two rates **(f, k)** (the Pearson map). This is
the original mechanism proposed for how a developing embryo breaks symmetry into structure.

## Operators ([`embryo_gray_scott_ops.py`](embryo_gray_scott_ops.py))

| operator | what it does |
|---|---|
| `gray_scott` | one reaction-diffusion tick on a 2-channel `grid` field (ch0=A, ch1=B): periodic Laplacian + reaction + feed/kill, with `substeps` inner Euler steps per frame; `kind=field` |
| `rd_seed` | frame-0 IC (`before_frame: 1`): A≈1, B≈0 with a small central square seeded (A=0.5, B=0.25) + a little noise (the symmetry-breaking perturbation) |

The two morphogens are a Plexus **2-channel `grid` field**; the PDE is a Plexus **field operator**
stepped by the engine (fields persist across frames). A 1-node dummy set satisfies the engine's
≥1-set requirement; the field does all the work. Boundaries are periodic (the Laplacian wraps).

## Run

```bash
python prototype/embryo_gray_scott/run_embryo_gray_scott.py --rank 0 --nproc 2 --device cuda:0 &
python prototype/embryo_gray_scott/run_embryo_gray_scott.py --rank 1 --nproc 2 --device cuda:1 &
wait; python prototype/embryo_gray_scott/run_embryo_gray_scott.py --montage
```

Each Pearson preset → `archive/<name>/` (`movie.mp4` morphogen-B, inferno on black; `strip.png`
development stages; `spec.yaml`; `diag.json`), plus `_montage.png` + `_summary.md`.

## Findings (`archive/_summary.md`)

- **The (f, k) map controls the morphogenetic class** — the single sweep reproduces the full
  Pearson zoo from one operator: `mitosis`/`spots` (self-replicating and stable spot lattices),
  `maze`/`stripes` (labyrinths), `coral`/`worms` (branching), `holes` (negative spots),
  `chaos`/`waves` (spatiotemporal dynamics), `uskate` (filling bubbles). 11/12 presets live.
- **There is an extinction region.** Low feed at the original `waves` point (f=0.014, k=0.045)
  the reaction dies before it can nucleate — patterns only exist inside a bounded tongue of
  (f, k). Low-feed presets (`waves`, `stripes`, `maze`) need a **stronger ignition** (a larger
  central seed + more noise) to reach the pattern basin; with it they self-organise robustly.
- **Development is progressive:** all patterns grow *outward from the seed* (see `strip.png`), a
  reaction front spreading and differentiating — morphogenesis as an expanding instability.

## Notes

- 200×200 grid, 2 channels, 20 substeps/frame (~14 000 PDE steps over 700 frames); ~6 s/preset
  on an A6000. `DA=0.16, DB=0.08, dt=1` (the reference values); Euler-stable at this `dt`.
- The next Plexus step would **couple the morphogen to tissue** (a real embryo): gate `cell_grow`
  or cell fate by the B field via `sample`/`gather`, turning the Turing pattern into a body plan.
