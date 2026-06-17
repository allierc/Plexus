# The Well in Plexus -- exhaustive test suite

Each row is one scenario: a spec.yaml driving the generic engine, no per-
scenario code. `ok` is the per-family intent check (behaviour occurred, not
just 'it ran'). Cite: The Well, Ohana et al., NeurIPS 2024 (github.com/PolymathicAI/the_well).


## Gray-Scott reaction-diffusion (FIELD self-update)

name | frames | grid/N | metric | value | extra | ok | sec
---|---|---|---|---|---|---|---
rd_spots | 101 | 128x128 grid | contrast | 0.167 | B_area=0.233 | OK | 18.3
rd_gliders | 101 | 128x128 grid | contrast | 0.1783 | B_area=0.081 | OK | 32.1
rd_bubbles | 101 | 128x128 grid | contrast | 0.1585 | B_area=0.090 | OK | 17.8
rd_maze | 101 | 128x128 grid | contrast | 0.1418 | B_area=0.421 | OK | 35.8
rd_worms | 101 | 128x128 grid | contrast | 0.1978 | B_area=0.358 | OK | 28.4

## Acoustic scattering (FIELD wave, heterogeneous medium)

name | frames | grid/N | metric | value | extra | ok | sec
---|---|---|---|---|---|---|---
wave_inclusions | 151 | 256x256 grid | max_amp | 2.369 | E_end/E_0=0.074 | OK | 9.2
wave_split | 151 | 256x256 grid | max_amp | 1.264 | E_end/E_0=0.072 | OK | 8.8
wave_lens | 131 | 256x256 grid | max_amp | 1.628 | E_end/E_0=0.058 | OK | 7.4
wave_gradient | 151 | 256x256 grid | max_amp | 1.488 | E_end/E_0=0.030 | OK | 8.4
wave_double_slit | 131 | 256x256 grid | max_amp | 2.085 | E_end/E_0=0.069 | OK | 9.1
wave_maze | 151 | 256x256 grid | max_amp | 2.062 | E_end/E_0=0.656 | OK | 11.4

## Active matter (SET: Vicsek particles + radius graph)

name | frames | grid/N | metric | value | extra | ok | sec
---|---|---|---|---|---|---|---
am_flock | 201 | 4000 particles | order_phi | 0.992 | phi_0=0.10 | OK | 37.5
am_disorder | 201 | 4000 particles | order_phi | 0.526 | phi_0=0.06 | OK | 46.8
am_bands | 201 | 4000 particles | order_phi | 0.939 | phi_0=0.08 | OK | 43.5
am_swirl | 201 | 4000 particles | order_phi | 0.977 | phi_0=0.05 | OK | 39.2

**15/15 scenarios passed their intent check.**