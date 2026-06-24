# Phase 2 — Inverse Fit Anchoring Specification

## Timing Alignment (VERIFIED 2026-06-24)

All numbers derived from `check_timing_alignment.py` and `cardio_real.npz` analysis.

### Real GT Beat (what Phase 2 must fit)
- **Beat onsets detected:** [2, 51, 101, 152, 204] (in frames, from `cardio_real.npz`)
- **Period:** 50 frames (mean inter-onset interval)
- **Fit beat:** beat index **3** (the -2 beat in trainer convention: `fit_beat=-2`)
- **Fit beat onset frame:** **152**
- **Fit beat end frame:** **204** (next onset)
- **Fit beat duration:** **53 frames** (frames [152:205), frame 152 to 204 inclusive)
- **Closed loop:** starts at rest (frame 152), returns to rest (frame 204)

### Atlas Phase 1 Morphology Window (ALIGNED with GT)
- **Measurement frames:** [50:103] = **53 frames**
- **Matches GT duration:** ✓ YES
- **Note:** Atlas measures a synthetic forward beat; Phase 2 will compare it to the real GT beat on the same 52-frame temporal scale

## Phase 2 Trainer Setup (cardio_mpm_train.py)

When `cardio_mpm_train.py` is adapted to use parametric patterns (Phase 2), use these args:

```bash
python cardio_mpm_train.py <label> \
  --fit_beat 3 \
  --warmup 0 \
  --grad 0 \
  --bwidth 0.06 \
  --n_iter 300 \
  <other args for learning rate, amplitude, drag, etc.>
```

### Explanation
- **`--fit_beat 3`** → fit the 4th beat (onset index 3 in the detected list) = frames [152:205]
- **`--warmup 0`** → settle for one beat (PERIOD frames) before starting gradient
- **`--grad 0`** → differentiate over the full 52-frame beat duration
- **`--bwidth 0.06`** → Dirichlet outer band (distance 0.06 from domain edge)

## Boundary Anchoring (Outer Band)

The outer band (Dirichlet-anchored nodes) should be pinned to real GT trajectory:

```python
real_disp, bnd, onsets, period = load_real(rest_pos, bwidth=0.06)
fit_beat_idx = 3
onset = onsets[fit_beat_idx]
end = onsets[fit_beat_idx + 1]  # = 204
for t in range(onset, end + 1):
    sim_pos[bnd, t] = rest_pos[bnd] + real_disp[t, bnd]
```

**Key:** Each frame `t ∈ [152:205]` of the sim should have outer nodes pinned to the corresponding frame `t` in `cardio_real.npz`. **No offset or resampling** — the sim and real data are both in sheet domain `[0.15, 0.85]²`, indexed to the same MPM particles.

## Interior Fit Target

- **Interior nodes:** all nodes where `~bnd` (not in the outer band)
- **Target:** match interior trajectories `real_disp[152:205, ~bnd]` over the full 52-frame beat
- **Loss:** R² (per-node motion-normalized interior fit, same as `cardio_mpm_train.py:275-279`)
- **Morphology loss:** optional, can weight openness/aspect/angle to match loop shape metrics

## Sanity Checks

Before launching Phase 2, verify:

1. ✓ `real_disp[152:205].shape[0] == 53` (beat is 53 frames)
2. ✓ `morphology()` measures atlas sim over exactly 53 frames
3. ✓ Boundary nodes at onset frame 152: `sim[bnd, 152] == rest[bnd] + real_disp[152, bnd]` (pinned to start of beat)
4. ✓ Boundary nodes at final frame 204: `sim[bnd, 204] ≈ rest[bnd] + real_disp[204, bnd]` (pinned to end of beat, should close the loop)

## Files to Modify / Create

- [ ] Adapt `cardio_mpm_train.py` to load parametric pattern params (not UNet)
- [ ] Update loss function to include loop-morphology terms
- [ ] Implement boundary anchoring in forward loop (call `anchor(lvl, rest, real_disp[t], bnd)` each frame)
- [ ] Record both R² and loop metrics (openness, aspect, angle) to dashboard

## Reference

- Real data: `/groups/saalfeld/home/allierc/Graph/Plexus/prototype/cardio/cardio_real.npz`
- Real data loader: `cardio_mpm_data.load_real(rest_pos, bwidth=0.06)`
- Atlas morphology: `cardio_mpm_atlas.morphology(name)` returns per-node loop metrics
- Old trainer (reference): `cardio_mpm_train.py` (has the `anchor()` and loss logic, just needs parametric pattern swap)
