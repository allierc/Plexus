# `well/` — The Well, implemented in Plexus

A prototype that expresses simulations from **The Well** (Ohana et al., *NeurIPS
2024*; 15 TB / 16 PDE families; `github.com/PolymathicAI/the_well`, cloned at
`papers/the_well`) inside the Plexus framework — **spec → registered operators**,
with the framework's categorical separation (set vs field; state / relations /
entities).

It is the companion to `water/`. Where `water/` exercised the **set** half of
Plexus (Lagrangian MPM particles), this exercises the **field** half (Eulerian grid
PDEs) — and closes the loop with one set-based dataset. **Read `notes.md` for the
full "what it takes" write-up.**

## Three operators, three Well families (simplest first)

| operator | kind | Well dataset | what it is |
|---|---|---|---|
| `reaction_diffusion` | exchange (FIELD) | `gray_scott_reaction_diffusion` | Gray-Scott: Laplacian diffusion + local reaction on a 2-channel grid |
| `wave_acoustic` | exchange (FIELD) | `acoustic_scattering_*` | first-order acoustics `[p,u,v]` in a heterogeneous medium `ρ(x,y)` |
| `active_matter` + `radius_graph` | lateral + rewire (SET) | `active_matter` | Vicsek self-propelled particles (the micro-model the Well coarse-grains) |

Reaction-diffusion follows ParticleGraph's `generators/RD_Gray_Scott.py` (same
du/dv law, cited in the code); the field machinery generalizes the prototyped MPM's
grid transfer.

## 15 scenarios (the "experiment" is only the YAML)

- **Gray-Scott zoo** — `rd_spots rd_gliders rd_bubbles rd_maze rd_worms` (one
  operator; only `(f,k)` + initial condition change between patterns).
- **Acoustic scattering** — `wave_inclusions wave_split wave_lens wave_gradient
  wave_double_slit wave_maze` (random inclusions, discontinuous interface, focusing
  lens, refracting gradient, double-slit diffraction, connected sound-hard maze).
- **Active matter** — `am_flock am_disorder am_bands am_swirl` (the Vicsek
  order/disorder transition: φ→0.99 ordered vs φ≈0.53 disordered).

## Run

```bash
cd prototype/well
PYTHONPATH=/workspace/Plexus/src python well_suite.py           # all 15 -> gifs, gallery.png, results.md, run.log
PYTHONPATH=/workspace/Plexus/src python render_well.py rd_maze  # one field sim
PYTHONPATH=/workspace/Plexus/src python render_am.py   am_flock # one particle sim
```

Outputs: `<name>.gif`, `<name>_montage.png`, `gallery.png`, `results.md`
(per-family intent checks — 15/15 pass), `run.log`.

## Files

`well_fields.py` (MultiField) · `well_ops.py` (operators) · `well_engine.py`
(generic build+run) · `well_schema.py` (typed, validated spec) · `render_well.py` /
`render_am.py` · `well_suite.py` · `scenarios/*.yaml`.

No engine edits per scenario: a new Well family is **one registered operator + one
channel layout + a YAML**.
