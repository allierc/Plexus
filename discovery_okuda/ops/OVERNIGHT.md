# Overnight plan

Two questions, run in parallel, plus the ladder continuing behind them.

## A. The corset, and the claim that it is not needed

`Ku & Bilder (2023) Dev. Cell 58:522` is cited in `bm_secrete`. The "molecular corset"
idea is that a basement membrane which resists CIRCUMFERENTIAL expansion more than meridional
expansion squeezes the girth and pushes growth into the ends, elongating the organ.

**The prediction has the opposite sign to every shape result in this prototype so far.** A polar-dense
MATRIX resists at the poles and gives an OBLATE tissue, aspect `r_eq/r_ax = 1.43`. A corset resists at
the equator, so it should give a PROLATE one, aspect < 1. Same gate, same pipeline, opposite outcome --
which is what makes it a test rather than a demo. If a corset map produces an oblate tissue, the gate
is not reading what we believe it reads.

Two stages, because the coupling is one-way: the epithelium is a replay, so the only way a membrane
property reaches the tissue is to record a map in pass A and rebuild pass B gated on it.

| run | aniso | what it asks |
|---|---|---|
| 102 | 1.0 | isotropic control -- the aspect with no corset at all |
| 103 | 3.0 | does a mild corset elongate? |
| 104 | 10.0 | the strong corset |
| 105 | 0.1 | **reversed**: stiffer along the meridian. Should push the aspect the OTHER way. A sign control, because an effect that does not reverse when the cause reverses is not the effect. |

**And the competing explanation, which is the interesting half.** Elongation can be generated inside
the epithelium by tension-keyed junctional myosin, with a completely isotropic membrane. Measured
earlier in this prototype: tension-keyed myosin doubles the T1 rate (0.0213 against 0.0089 per cell per
frame), and T1s are how an epithelium elongates without changing cell number.

| run | what it asks |
|---|---|
| 106 | polarised myosin, isotropic membrane. **If this elongates as much as 104, the corset is not necessary.** |
| 107 | both together -- additive, redundant, or antagonistic? |

The honest scope statement: this is a Drosophila egg-chamber mechanism being run on a mammalian-style
spheroid, so it is a demonstration that the mechanism CAN produce the shape in this model, not a claim
about mammalian tissue. And the corset needs anisotropic connection strength, which only the SPRING
membrane can express -- the continuum's Lame constants are isotropic, so a corset there would need a
transversely isotropic material model that does not exist yet.

## B. The continuum ladder (M3)

91-98 established the sheet grows with the spheroid and carries the right strain, that secretion keeps
it whole, and fixed the standoff. M3 asks whether the integrin anchor is needed at all now that the
grid boundary carries the sheet.

| run | what it asks |
|---|---|
| 99 | anchor on, nominal `k_adh` -- does the sheet strain MORE when tethered? |
| 100 | anchor x5 -- does a stiff anchor tear the sheet instead of holding it? |

## What to read in the morning

For A: `aspect_end` in each run's `pass1.json`, against 102's isotropic baseline and against the
matrix's 1.43. The sign of 105 relative to 103/104 is the control that says the gate works.
For B: `strain_end` and `coverage` against 91 and 92.
