# Turing x vertex study — predictions recorded BEFORE the runs

Posed 2026-07-30 ~17:40 EDT, from reading the growth law only. Scored by `predict.score`.

## The mechanism claim under test

`grow_3d` has two branches:

    s <- s * (1 + rate*(rho + hillv))          hillv = a^h/(a_sw^h + a^h)  in [0,1]
    rho > 0 :  s <- min(s, (vth_frac*v_ref/V0f_init)^(1/3))     # CAPPED  -- divide, don't bulge
    rho == 0:  s <- min(s, cap)                                 # activator-only bulge

**Claim:** at `rho > 0` the per-cell cap pins every cell to the same target volume, so a
high-activator cell reaches the division threshold SOONER but never grows BIGGER. Shape is
therefore activator-independent by construction, and lowering `a_sw` into the activator's range
cannot make the pattern shape the sheet -- it can only change division timing.

## Wave A — sweep `a_sw` at the archived `rho = 1.0`

| a_sw | intent | prediction |
|---|---|---|
| 50.0 (archived) | control | `protr 1.0-1.15` — the minisite operating point |
| 0.60 | confirmatory | `protr 1.0-1.15` |
| 0.45 | confirmatory | `protr 1.0-1.15` |
| 0.30 | confirmatory | `protr 1.0-1.15` |
| 0.15 | confirmatory | `protr 1.0-1.15` |

Also expected but not the scored clause: `|corr_act_rad| < 0.15` throughout, and `cells_end`
RISING as a_sw falls (the activator advances the division clock even though it cannot bulge).
**Refuted if** protr exceeds 1.15 at any a_sw — that would mean the cap does not do what I think.

## Wave B — sweep `rho` at `a_sw = 0.30`

| rho | intent | prediction |
|---|---|---|
| 1.0 | control | `protr 1.0-1.15` (same as wave A) |
| 0.5 | confirmatory | `protr 1.0-1.15` — still capped |
| 0.2 | confirmatory | `protr 1.0-1.15` — still capped |
| 0.05 | adversarial | `protr 1.0-1.15` — still capped, but close to the branch switch |
| 0.0 | adversarial | `protr >= 1.3` — the cap is GONE; activator-driven bulging |

**The discriminating point is rho = 0.0.** If protr does not rise there, the cap is not what
holds the shape flat and my whole reading of this operator is wrong.

---

## Wave C result and what it forces (recorded 2026-07-30 ~17:50)

`conserve_amount` on/off, one edit, everything else held:

| conserve_amount | act_max @200 | act_max @400 | act_frac_high | corr_act_rad | protr |
|---|---|---|---|---|---|
| 1 (archived default) | 6.5e-8 | **3.0e-19** | 0.000 | undefined | 1.041 |
| 0 | 0.421 | **0.431** | 0.340 | **+0.292** | 1.033 |

**The activator is diluted to extinction by `conserve_amount`.** It divides `cell.chem` by the
volume growth ratio EVERY tick; over 400 frames that is a geometric decay of ~0.88/frame.

**Therefore waves A and B were null BY CONSTRUCTION** — both swept knobs (`a_sw`, `rho`) that
multiply an activator which was already 1e-19. Ten runs, all `cells_end = 1778`. That also
retro-explains why the two minisite movies are identical.

Waves D and E repeat A and B with `conserve_amount = 0`, i.e. against a live pattern.

## Wave D — `a_sw` at `conserve_amount = 0`

Activator now lives at ~0.43 peak / 0.14 mean, so the Hill switch is in range for a_sw ~ 0.1-0.4.

| a_sw | intent | prediction |
|---|---|---|
| 50.0 | control | `protr 1.0-1.10` — switch off, uniform growth |
| 0.60 | confirmatory | `protr 1.0-1.15` — above the activator peak, barely fires |
| 0.40 | confirmatory | `protr 1.0-1.20` |
| 0.25 | confirmatory | `protr 1.0-1.25` |
| 0.10 | adversarial | `protr >= 1.15` — switch saturated ON, i.e. near-uniform again |

Expectation not scored: `corr_act_rad` should PEAK at intermediate a_sw (~0.25-0.40) and fall at
both ends — off at 50, saturated at 0.10. A monotone corr would refute the Hill-switch reading.

## Wave E — `rho` at `conserve_amount = 0`, `a_sw = 0.30`

This is the CAP hypothesis, now unconfounded (wave B tested it against a dead field, so its
refutation was uninformative — my error, recorded).

| rho | intent | prediction |
|---|---|---|
| 1.0 | control | `protr 1.0-1.15` |
| 0.5 | confirmatory | `protr 1.0-1.20` |
| 0.2 | confirmatory | `protr 1.0-1.30` |
| 0.05 | confirmatory | `protr >= 1.2` |
| 0.0 | confirmatory | `protr >= 1.3` — cap branch gone, activator-only bulge |

**Discriminating point is again rho = 0.0**, but now with a live activator behind it.

---

## Waves F/G — Okuda's (chi, gamma) axes, at the only setting where shaping happens

Run at `conserve_amount=0, rho=0.05, a_sw=0.30` — the one regime with real coupling (corr +0.63).
Caveat recorded up front: `hollow_frac` is already 0.197 there, so morphology read from these
waves is measured on a straining mesh and must be reported with the integrity number.

Okuda's targets: thin tube (chi .01, gamma 100) · thick tube (.1, 1) · undulation (.1, 100) ·
branching (.01, .01). Here chi = `cell_diffuse.chi`, gamma = `cell_react.rate`.

### Wave F — `chi` (domain size). Okuda: tube diameter ∝ chi^(1/4)

| chi | intent | prediction |
|---|---|---|
| 0.3 | confirmatory | `protr >= 1.10` — small domains, many fine lobes |
| 0.65 | confirmatory | `protr >= 1.10` |
| 1.3 (base) | control | `protr 1.10-1.30` |
| 3.0 | confirmatory | `protr 1.0-1.30` — larger, broader domains |
| 8.0 | adversarial | `protr 1.0-1.15` — domains approach the whole vesicle; nothing localised left to push |

### Wave G — `rate` (gamma; chemistry speed vs mechanics)

**Mechanism claim:** `corr_act_rad` should FALL as `rate` rises. The shell relaxes on a mechanical
timescale; if the pattern reorganises faster than the shell can follow, the bulges lag the bands
and the correlation decays. Low `rate` = quasi-static pattern = shell tracks it.

| rate | intent | prediction |
|---|---|---|
| 0.1 | confirmatory | `protr >= 1.10` |
| 0.3 | confirmatory | `protr >= 1.10` |
| 1.0 (base) | control | `protr 1.10-1.30` |
| 3.0 | confirmatory | `protr 1.0-1.25` |
| 10.0 | adversarial | `protr 1.0-1.15` — chemistry outruns mechanics |

Scored on `protr`; the `corr` trend is the mechanism claim and is reported alongside.
**Refuted if** corr is flat or rising across `rate` — that would mean the timescale separation
does not matter here, and the Okuda gamma axis is not doing what the paper says it does.
